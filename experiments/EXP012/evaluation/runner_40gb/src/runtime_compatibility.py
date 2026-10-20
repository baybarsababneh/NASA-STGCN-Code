"""Runtime-only amendment. A real, complete nominal 40GB replay is mandatory."""
import json,os
from pathlib import Path
from common import sha,digest

def check_environment(actual,reference,amendment):
    allowed=set(amendment['changed_runtime_fields'])
    if allowed!={'gpu','gpu_bytes'}:raise RuntimeError('Runtime amendment exceeds GPU name/capacity scope')
    if set(actual)!=set(reference):raise RuntimeError('Runtime environment schema differs')
    unexpected=[k for k in reference if k not in allowed and actual[k]!=reference[k]]
    if unexpected:raise RuntimeError('Non-GPU runtime fields differ: '+', '.join(unexpected))
    target=amendment['target_runtime_fields']
    if set(target)!=allowed or any(actual[k]!=target[k] for k in allowed):raise RuntimeError('Require exact amended A100-SXM4-40GB name and capacity')
    if actual['capability']!=[8,0]:raise RuntimeError('A100 capability differs')
    return {'changed_fields':{k:{'original':reference[k],'actual':actual[k]} for k in sorted(allowed)},'all_other_fields_exact':True}

def identity(environment,results,package,runner):
    results=Path(results);package=Path(package);runner=Path(runner)
    return {'mode':'TRAIN_ONLY_FROZEN_WEIGHT_REPLAY_NO_FITS','environment':environment,'runner_manifest_sha256':sha(runner/'RUNNER_MANIFEST.json'),'binding_sha256':sha(runner/'EVALUATION_BINDING.json'),'amendment_sha256':sha(runner/'COMPATIBILITY_AMENDMENT.json'),'reference_run_freeze_sha256':sha(results/'RUN_FREEZE.json'),'results_manifest_sha256':sha(results/'OUTPUT_MANIFEST.json'),'package_manifest_sha256':sha(package/'PACKAGE_MANIFEST.json')}

def verify_receipt(directory,expected,roster,weights):
    directory=Path(directory);pass_path=directory/'COMPATIBILITY_PASS.json'
    if not pass_path.exists():raise RuntimeError('40GB use is NOT AUTHORISED: complete nominal replay first; missing '+str(pass_path))
    receipt=json.loads(pass_path.read_text())
    if receipt['status']!='PASS_40GB_NOMINAL_BITWISE_REPLAY' or receipt['identity']!=expected:raise RuntimeError('40GB replay identity/status differs')
    counts={'streams':1053,'checkpoint_states':972,'nominal_split_streams':2106,'detector_split_streams':4212,'adaptive_windows':2080,'mismatches':0,'scientific_fits':0}
    if receipt['counts']!=counts or receipt['NASA_TEST_opened'] or receipt['NASA_labels_opened'] or receipt['scientific_settings_changed']:raise RuntimeError('40GB replay incomplete or changed science')
    if len(receipt['roster'])!=1053 or set(receipt['roster'])!=set(roster):raise RuntimeError('40GB replay roster differs')
    actual={p.relative_to(directory).as_posix() for p in (directory/'units').rglob('*') if p.is_file() and '.tmp-' not in p.name}
    if actual!=set(receipt['files']):raise RuntimeError('40GB replay evidence roster differs')
    for n,h in receipt['files'].items():
        if sha(directory/n)!=h:raise RuntimeError('Changed 40GB replay evidence: '+n)
    if sha(directory/'COMPATIBILITY_RUN_FREEZE.json')!=receipt['run_freeze_sha256'] or json.loads((directory/'COMPATIBILITY_RUN_FREEZE.json').read_text())!=expected:raise RuntimeError('Changed compatibility run freeze')
    for jid in roster:
        rec=json.loads((directory/'units'/jid/'REPLAY_RECORD.json').read_text())
        if rec['job_id']!=jid or not rec['exact_pass'] or rec['selected_state_sha256']!=(weights[jid]['state_sha256'] if jid in weights else None):raise RuntimeError('Invalid replay record '+jid)
    return sha(pass_path)

def require_pass(directory,environment,results,package,runner):
    results=Path(results)
    cf=json.loads((results/'CALIBRATION_FREEZE.json').read_text());mf=json.loads((results/'MODEL_FREEZE.json').read_text())
    roster={r['job_id'] for r in cf['thresholds']};weights={r['job_id']:r for r in mf['weights']}
    return verify_receipt(directory,identity(environment,results,package,runner),roster,weights)
