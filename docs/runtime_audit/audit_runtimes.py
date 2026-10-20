"""Reconstruct recorded computation for EXP001--EXP015, without running models.

Usage: python audit_runtimes.py --sources ORIGINAL_FILES [MORE_FOLDERS ...] --output AUDIT
Sources are original ZIPs/notebooks/JSONL records. No file dates are used as timers.
Torch tensor data are never loaded or executed.
"""
import argparse, collections, csv, hashlib, io, json, pickle, re, statistics, zipfile
from pathlib import Path


class TensorPlaceholder:
    pass


def discard_tensor(*args, **kwargs):
    return TensorPlaceholder()


class MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return collections.OrderedDict
        if module == 'torch._utils' and name in {'_rebuild_tensor_v2', '_rebuild_tensor', '_rebuild_parameter', '_rebuild_tensor_v3'}:
            return discard_tensor
        if module == 'torch' and name.endswith('Storage'):
            return TensorPlaceholder
        if (module, name) == ('builtins', 'set'):
            return set
        raise pickle.UnpicklingError(f'Unsupported global {module}.{name}')

    def persistent_load(self, pid):
        return TensorPlaceholder()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}

    def source(name):
        for root in args.sources:
            path = root/name
            if path.is_file():
                if name not in hashes:
                    with path.open('rb') as stream:
                        hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
                return path
        raise FileNotFoundError(name)

    def archive_bytes(name, member):
        with zipfile.ZipFile(source(name)) as z:
            return z.read(member)

    def obj(name, member):
        return json.loads(archive_bytes(name, member))

    def csv_bytes(data):
        return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))

    rows = []
    details = {}

    def add(i, hardware, seconds, basis, scope, evidence, **extra):
        entry = dict(experiment=f'EXP{i:03d}', hardware=hardware,
                     measured_seconds=seconds, basis=basis, scope=scope,
                     evidence=evidence, **extra)
        rows.append(entry)
        return entry

    # Original CPU checkpoints, excluding smoke jobs. The same channel/seed
    # may occur for two EXP005 candidates, so candidate identity is retained.
    early = []
    with zipfile.ZipFile(source('checkpoints.zip')) as z:
        for name in z.namelist():
            if not name.startswith('checkpoints/') or not re.search(r'_seed(17|29|43)\.pt$', name):
                continue
            with zipfile.ZipFile(io.BytesIO(z.read(name))) as tensor_zip:
                pk = next(n for n in tensor_zip.namelist() if n.endswith('/data.pkl') or n == 'data.pkl')
                value = MetadataUnpickler(io.BytesIO(tensor_zip.read(pk))).load()
            result = value['result']
            clean = {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool)) or v is None}
            clean['checkpoint_member'] = name
            early.append(clean)
    for i in range(1, 6):
        rr = [r for r in early if f'exp{i:03d}' in r['experiment_id']]
        expected = 486 if i == 5 else 243
        assert len(rr) == expected
        assert len({r['channel'] for r in rr}) == 81
        assert {r['seed'] for r in rr} == {17, 29, 43}
        assert len({r['checkpoint_member'] for r in rr}) == expected
        add(i, 'CPU; processor not recorded', sum(r['elapsed_seconds'] for r in rr),
            'sum of measured training-loop elapsed timers',
            f'{expected} confirmation fits; screening additional' if i == 5 else '243 channel/seed fits',
            'checkpoints.zip: checkpoints/v2_exp%03d*/...pt -> result.elapsed_seconds' % i,
            count=expected, whole_experiment_wall_seconds=None)
        details[f'EXP{i:03d}'] = {mission: {'fits': len(q),
            'measured_seconds': sum(r['elapsed_seconds'] for r in q)}
            for mission in ['MSL', 'SMAP'] if (q := [r for r in rr if r['mission'] == mission])}
    cpu_source = archive_bytes('src.zip', 'src/v2_train_exp002.py').decode()
    assert re.search(r'torch\.device\(\s*[\"\']cpu', cpu_source)
    keys = sorted(set().union(*(r.keys() for r in early)))
    with (out/'early_fit_timers.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(early)

    # EXP004 has a separate direct wall timer, but only for the resumed launch.
    log = source('full_parallel_run.log').read_text()
    assert 'V2 EXP004 PARALLEL CPU RUNNER' in log
    wall = float(re.search(r'Wall time:\s+([0-9.]+) s', log)[1])
    assert int(re.search(r'New runs completed:\s+(\d+)', log)[1]) == 239
    details['EXP004']['resumed_launch'] = dict(new_fits=239, reused_fits=4,
                                              measured_wall_seconds=wall, workers=2)
    # The screening log only prints the 251 newly completed screen fits;
    # 37 were already complete. Rounded display times cannot recover those 37.
    log = source('full_tuning_run.log').read_text()
    screen = [float(v) for v in re.findall(r'^\[DONE\] screen .* time=([0-9.]+)s$', log, re.M)]
    assert len(screen) == 251 and 'Already complete: 37' in log
    details['EXP005']['screening_partial_log'] = dict(new_fits=251, reused_fits=37,
        sum_rounded_display_seconds=sum(screen), complete_screening_timer_available=False)

    # Complete final GPU job tables; use synchronized fit-level wall timers,
    # not a sum of interrupted notebook sessions or file timestamps.
    for i, outer, member, expected in [
        (6, 'exports-20261004T114850Z-1-001.zip', 'exports/EXP006_GPU_V1_results.zip', 1209),
        (7, 'exports-20261004T114932Z-1-001.zip', 'exports/EXP007_GPU_V1_results.zip', 1116)]:
        data = archive_bytes(outer, member)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            rr = csv_bytes(z.read('tables/all_runs.csv'))
            env = json.loads(z.read('RUN_FREEZE.json'))['environment']
        assert len(rr) == expected and len({r['job_id'] for r in rr}) == expected
        assert 'A100' in env['gpu'] and '40GB' in env['gpu']
        total = sum(float(r['fit_wall_seconds']) for r in rr)
        add(i, env['gpu'], total, 'sum of synchronized per-fit wall timers',
            f'{expected} unique fits in the final GPU study; setup and earlier attempts excluded',
            f'{outer} -> {member}: tables/all_runs.csv, fit_wall_seconds',
            count=expected, whole_experiment_wall_seconds=None)
        details[f'EXP{i:03d}'] = {stage: {'fits': len(q),
            'measured_seconds': sum(float(r['fit_wall_seconds']) for r in q)}
            for stage in sorted({r['stage'] for r in rr}) if (q := [r for r in rr if r['stage'] == stage])}

    rr = csv_bytes(archive_bytes('EXP008_results.zip', 'tables/replay_results.csv'))
    assert len(rr) == 534 and len({r['job_id'] for r in rr}) == 534
    add(8, 'NVIDIA A100-SXM4-40GB', sum(float(r['elapsed_seconds']) for r in rr),
        'sum of measured replay-loop timers',
        '534 two-pass replays; checkpoint/data loading, nominal diagnostics, and export excluded',
        'EXP008_results.zip: tables/replay_results.csv, elapsed_seconds', count=534,
        whole_experiment_wall_seconds=None)

    name = 'EXP009_DETECTOR_V1_E01_results_20261004T160429Z.zip'
    close = obj(name, 'results/STUDY_CLOSE.json')
    prior = obj(name, 'engineering_E01/PRE_E01_COMMIT_INVENTORY.json')
    assert close['units'] == 4194 and len(prior) == 936
    add(9, 'CPU NumPy detector; processor not recorded', close['elapsed_seconds'],
        'measured elapsed time of resumed detector run',
        'E01 resumed stage, with 936 earlier units reused; initial failed launch excluded',
        f'{name}: results/STUDY_CLOSE.json, elapsed_seconds; PRE_E01_COMMIT_INVENTORY.json',
        count=4194, reused_units=936, whole_experiment_wall_seconds=None)

    # Check every EXP010 unit record for a computational timing field.
    timing_key = re.compile(r'elapsed|wall|runtime|seconds|timestamp|\butc\b', re.I)
    with zipfile.ZipFile(source('EXP010_MECHANISMS_V1_results.zip')) as z:
        units = [n for n in z.namelist() if n.startswith('units/') and n.endswith('.zip')]
        assert len(units) == 3078
        for n in units:
            with zipfile.ZipFile(io.BytesIO(z.read(n))) as u:
                record = json.loads(u.read('record.json'))
            assert not any(timing_key.search(k) for k in record)
    add(10, 'NVIDIA A100-SXM4-40GB', None, 'not recoverable',
        '3078 committed diagnostic units; no computational timer in unit or closure records',
        'EXP010_MECHANISMS_V1_results.zip: all units/*/record.json, STATUS.json, STUDY_CLOSE.json', count=3078)

    name = 'EXP011_FUSION_PILOT_V1_summary_20261005T070729Z.zip'
    status = obj(name, 'results/STATUS.json')
    assert status['detector_fits'] == 440 and status['forecasting_fits'] == 0
    def assert_no_timers(v):
        if isinstance(v, dict):
            assert not any(timing_key.search(k) for k in v)
            for vv in v.values(): assert_no_timers(vv)
        elif isinstance(v, list):
            for vv in v: assert_no_timers(vv)
    with zipfile.ZipFile(source(name)) as z:
        for n in z.namelist():
            if n.endswith('.json') and not n.startswith('__MACOSX/'):
                assert_no_timers(json.loads(z.read(n)))
    add(11, 'CPU environment; processor not recorded', None, 'not recoverable',
        '440 classifier fits; exported summary and log contain no computational elapsed time',
        f'{name}: results/STATUS.json, STUDY_CLOSE.json, logs/run_20261005T070729Z.log', count=440)

    name = 'EXP012_execution_provenance.zip'
    nb = obj(name, 'EXP012_ACTUALLY_EXECUTED.ipynb')
    cell = nb['cells'][5]
    assert 'src/run_exp012.py' in ''.join(cell['source'])
    elapsed = cell['metadata']['executionInfo']['elapsed']/1000
    assert cell['metadata']['executionInfo']['status'] == 'ok'
    prep = obj('EXP012_PREPARATION_results.zip', 'RUN_FREEZE.json')
    assert '80GB' in prep['environment']['gpu']
    add(12, prep['environment']['gpu'], elapsed, 'measured Colab execution-cell elapsed time',
        'complete preparation cell, including 243 LSTM fits and qualification; setup and later TEST evaluation excluded',
        f'{name}: EXP012_ACTUALLY_EXECUTED.ipynb, cell 5, executionInfo.elapsed (milliseconds)',
        count=243, whole_experiment_wall_seconds=None)

    # Heartbeats carry accumulated elapsed epoch time across exact resumes.
    # Taking the largest value per job/epoch prevents counting heartbeat and
    # FIT/STOP updates multiple times. Each terminal heartbeat must be STOP.
    progress = [json.loads(l) for l in source('EXECUTION_PROGRESS.jsonl').read_text().splitlines() if l.strip()]
    epochs = {}
    for o in progress:
        if o.get('phase') not in ['FIT', 'STOP'] or 'elapsed_epoch_seconds' not in o:
            continue
        key = (o['job'], o['epoch'])
        if key not in epochs or o['elapsed_epoch_seconds'] > epochs[key]['elapsed_epoch_seconds']:
            epochs[key] = o
    assert len(epochs) == 254 and len({job for job, ep in epochs}) == 22
    assert all(o['phase'] == 'STOP' and o['batch'] == o['batches'] for o in epochs.values())
    calls = [o for o in progress if o.get('phase') == 'SCORE_CALL_COMMITTED']
    assert len(calls) == 49781
    assert sum(o.get('phase') == 'SCORE_CALL_START' for o in progress) == len(calls)
    epoch_seconds = sum(o['elapsed_epoch_seconds'] for o in epochs.values())
    score_seconds = sum(o['seconds'] for o in calls)
    add(13, 'NVIDIA A100-SXM4-40GB', epoch_seconds + score_seconds,
        'sum of recorded epoch and scoring-call elapsed timers',
        '22 mission-level fits (254 epochs), plus 49781 uncached scoring calls; not an end-to-end run timer',
        'EXECUTION_PROGRESS.jsonl: max elapsed_epoch_seconds per job/epoch and SCORE_CALL_COMMITTED.seconds',
        count=22, epoch_seconds=epoch_seconds, score_call_seconds=score_seconds,
        whole_experiment_wall_seconds=None)
    details['EXP013'] = {job: {'epochs': len(q), 'measured_epoch_seconds': sum(o['elapsed_epoch_seconds'] for o in q)}
        for job in sorted({j for j, ep in epochs}) if (q := [o for (j, ep), o in epochs.items() if j == job])}
    with (out/'epoch_timers_EXP013.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['job', 'epoch', 'elapsed_epoch_seconds', 'UTC'])
        writer.writeheader()
        writer.writerows({k: o[k] for k in writer.fieldnames} for key, o in sorted(epochs.items()))

    source14 = archive_bytes('EXP014_LITERATURE_COMPARISON_V1_input_bundle.zip', 'src/run.py').decode()
    source15 = archive_bytes('EXP015_EXP013_GCN_HUNDMAN_SUPPLEMENT_V1_input_bundle.zip', 'src/run.py').decode()
    assert 'seconds=new_times[-1]' in source14
    assert 'inference_seconds=time.monotonic()-begin' in source15
    assert 'detector_seconds=time.monotonic()-begin' in source15
    for i, name, expected in [(14, 'EXP014_FINAL_RESULTS_392b1c7bdcd08967.zip', 81),
                              (15, 'EXP015_FINAL_RESULTS_c5f0eceb421e3e9d.zip', 729)]:
        progress = [json.loads(l) for l in archive_bytes(name, 'PROGRESS.jsonl').decode().splitlines() if l.strip()]
        units = [o for o in progress if o['action'] == 'UNIT_COMMITTED']
        assert len(units) == expected and len({o['job_id'] for o in units}) == expected
        env = obj(name, 'RUN_FREEZE.json')['environment']
        code_archive = ('EXP014_LITERATURE_COMPARISON_V1_input_bundle.zip' if i == 14
                        else 'EXP015_EXP013_GCN_HUNDMAN_SUPPLEMENT_V1_input_bundle.zip')
        code_manifest = archive_bytes(code_archive, 'MANIFEST.json')
        run_identity = obj(name, 'RUN_FREEZE.json')['identity']
        assert hashlib.sha256(code_manifest).hexdigest() == run_identity['code_manifest_sha256']
        declared_code = json.loads(code_manifest)['files']['src/run.py']
        assert hashlib.sha256(archive_bytes(code_archive, 'src/run.py')).hexdigest() == declared_code
        assert '40GB' in env['GPU']
        if i == 14:
            seconds = sum(o['seconds'] for o in units)
            extra = {}
            scope = '81 frozen-GDN scoring units, including replay/residual/detector work; report generation and setup additional'
        else:
            inf = sum(o['inference_seconds'] for o in units)
            det = sum(o['detector_seconds'] for o in units)
            seconds = inf + det
            extra = dict(inference_seconds=inf, detector_seconds=det)
            scope = '729 scoring units; model loading, validation, reports, setup, and exports additional'
        add(i, env['GPU'], seconds, 'sum of measured scoring-unit timers', scope,
            f'{name}: PROGRESS.jsonl, UNIT_COMMITTED', count=expected,
            whole_experiment_wall_seconds=None, runner_source_manifest_matches_final_run=True, **extra)

    assert len(rows) == 15
    payload = dict(method='Recorded timers only; no creation-date or hardware-scaled duration inference.',
                   experiments=rows, stage_details=details, source_sha256=hashes)
    (out/'timing_evidence.json').write_text(json.dumps(payload, indent=2)+'\n')
    fields = ['experiment', 'hardware', 'measured_seconds', 'basis', 'scope', 'count', 'evidence']
    with (out/'timing_summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(row['experiment'], row['measured_seconds'], row['basis'], flush=True)


if __name__ == '__main__':
    main()
