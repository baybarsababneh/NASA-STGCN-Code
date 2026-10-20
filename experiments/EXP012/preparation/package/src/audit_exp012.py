"""Read-only completed preparation audit. No training or NASA evaluation."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import torch
from integrity import sha,digest,read_commit
from models_io import tensor_hash,read_parent
from lstm_baseline import ControlledLSTM
from hundman_detector import detect
from detector_metrics_v2 import higher_threshold,burden

def audit(out,package=None):
    out=Path(out);events_before=[json.loads(s) for s in (out/'DATA_ACCESS_EVENTS.jsonl').read_text().splitlines()];package=Path(package) if package else Path(__file__).resolve().parents[1]
    manifest=json.loads((package/'PACKAGE_MANIFEST.json').read_text())
    for f,h in manifest['files'].items():
        if sha(package/f)!=h:raise RuntimeError('Input bundle integrity failure '+f)
    rows=json.loads((package/'data_manifest.json').read_text());plan=json.loads((package/'parent_plan.json').read_text())
    cfg=json.loads((package/'config.json').read_text());run=json.loads((out/'RUN_FREEZE.json').read_text());rh=digest(run)
    mf=json.loads((out/'MODEL_FREEZE.json').read_text());cf=json.loads((out/'CALIBRATION_FREEZE.json').read_text());bf=json.loads((out/'BENCHMARK_FREEZE.json').read_text())
    if run['package_manifest_sha256']!=sha(package/'PACKAGE_MANIFEST.json') or bf['run_hash']!=rh or mf['run_hash']!=rh or cf['run_hash']!=rh:
        raise RuntimeError('Freeze identity failure')
    assert bf['model_freeze_sha256']==sha(out/'MODEL_FREEZE.json') and bf['calibration_freeze_sha256']==sha(out/'CALIBRATION_FREEZE.json')
    assert bf['evaluation_contract']==json.loads((package/'evaluation_contract.json').read_text())
    expected=set();records={};parent={(m['job']['channel'],m['job']['seed'],m['job']['remove_messages']):m for m in plan}
    weight_roster={r['job_id']:r for r in mf['weights']};assert len(weight_roster)==972
    thresholds={r['job_id']:r for r in cf['thresholds']};assert len(thresholds)==1053
    verified=0;windows=0;reports=[]
    for row in rows:
        for arm,seed in [(a,s) for a in ['gcn','removed','tcn','lstm'] for s in [17,29,43]]+[('persistence',None)]:
            jid=f"{arm}__{row['channel']}"+(f"__{seed}" if seed is not None else '')
            expected.add(jid);base=read_commit(out,jid,{'run_hash':rh,'job_id':jid});assert base is not None
            rec=json.loads((base/'record.json').read_text());records[jid]=rec
            with np.load(base/'validation.npz',allow_pickle=False) as z:val={k:z[k] for k in z.files}
            assert np.array_equal(val['target_index'],np.arange(row['fit_end'],row['stop_end']))
            assert all(np.isfinite(v).all() for v in val.values())
            error=val['target_standardized'].astype(float)-val['prediction_standardized'].astype(float)
            assert np.isclose(np.mean(error**2),rec['validation_mse'],rtol=5e-6,atol=1e-12) # independent reduction-rounding check, not replay tolerance
            if arm in ['gcn','removed']:
                meta=parent[(row['channel'],seed,arm=='removed')];m,ck,rr=read_parent(package,meta)
                assert rec['parent_sha256']==meta['archive_sha256'] and rec['state_sha256']==tensor_hash(m.state_dict())
                assert rec['validation_mse']==rr['early_stop_mse'] and rec['parent_exact_mse'] and rec['exact_repeat'];del m,ck
            elif arm=='lstm':
                ck=torch.load(base/'checkpoint.pt',map_location='cpu',weights_only=True)
                m=ControlledLSTM(len(row['incoming_features']));m.load_state_dict(ck['state_dict'],strict=True);del m
                assert tensor_hash(ck['state_dict'])==rec['state_sha256'] and ck['identity']=={'run_hash':rh,'job_id':jid}
                assert ck['settings']==cfg['lstm'] and ck['result']['validation_mse']==rec['validation_mse']
                h=json.loads((base/'history.json').read_text());selected=[v for v in h if v['selected']]
                assert selected[-1]['epoch']==rec['best_epoch'] and selected[-1]['validation_mse']==rec['validation_mse'] and len(h)==rec['epochs_ran']
            elif arm=='tcn':
                p=out/'parent_weights/exp005'/f"{row['channel']}_seed{seed}.pt"
                ck=torch.load(p,map_location='cpu',weights_only=True)
                assert sha(p)==rec['parent_sha256'] and tensor_hash(ck['state_dict'])==rec['state_sha256']
            if seed is not None:
                wr=weight_roster[jid];assert wr['commit_sha256']==sha(base/'COMMIT.json')
                assert {k:v for k,v in wr.items() if k not in ['job_id','commit_sha256']}==rec
            qid='qualification__'+jid;expected.add(qid)
            qbase=read_commit(out,qid,{'run_hash':rh,'job_id':qid,'model_freeze_sha256':sha(out/'MODEL_FREEZE.json')});assert qbase is not None
            qr=json.loads((qbase/'record.json').read_text())
            with np.load(qbase/'calibration_and_diagnostics.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            assert np.array_equal(arrays['target_index'],np.arange(row['stop_end'],row['rows']))
            residual=np.abs(arrays['target_standardized'].astype(float)-arrays['prediction_standardized'].astype(float))
            assert qr['threshold']==higher_threshold(residual,.99)
            frozen=thresholds[jid];assert {k:v for k,v in qr.items() if k not in ['detector_reports','adaptive_window_audit']}==frozen
            for split,stream in [('validation',val),('calibration',arrays)]:
                errors=np.abs(stream['target_standardized'].astype(float)-stream['prediction_standardized'].astype(float))
                flags=errors>qr['threshold']['threshold'];d=detect(stream['target'],stream['prediction'],stream['target_index'],cfg['detector'])
                assert np.array_equal(flags,arrays[split+'_raw_flags']) and np.array_equal(d['flags'],arrays[split+'_adaptive_flags'])
                assert np.array_equal(np.where(flags,stream['target_index'],-1),arrays[split+'_raw_issue_index'])
                assert np.array_equal(d['issue_index'],arrays[split+'_issue_index']) and np.array_equal(d['smoothed'],arrays[split+'_smoothed_raw_residual'])
                assert qr['adaptive_window_audit'][split]['windows']==d['windows'];windows+=len(d['windows'])
                for name,flags0 in [('raw_q99',flags),('hundman_adapt_v1',d['flags'])]:
                    r=next(r for r in qr['detector_reports'] if r['split']==split and r['detector']==name)
                    for k,v in burden(flags0).items():assert r[k]==v
                    reports.append(r)
            verified+=1
    actual={p.name for p in (out/'units').iterdir() if p.is_dir() and (p/'COMMIT.json').exists()}
    if actual!=expected:raise RuntimeError('Missing/extra/duplicate completed unit identity')
    orphan_dirs=[p.name for p in (out/'units').iterdir() if p.is_dir() and not (p/'COMMIT.json').exists()]
    if orphan_dirs:raise RuntimeError('Unresolved orphan unit directories '+str(orphan_dirs))
    assert len(expected)==2106 and verified==1053 and len(reports)==4212
    with (out/'tables/nominal_diagnostics.csv').open() as f:
        table=list(csv.DictReader(f));assert len(table)==4212
        for a,b in zip(table,reports):
            for k,v in b.items():assert a[k]==('' if v is None else str(v))
    from summaries import forecast_rows,nominal_rows,resolution_rows
    for filename,expected_rows in [
        ('forecasting_summary.csv',forecast_rows(list(records.values()))),
        ('nominal_detector_summary.csv',nominal_rows(reports)),
        ('calibration_resolution_summary.csv',resolution_rows(list(thresholds.values())))]:
        with (out/'tables'/filename).open() as f:
            actual_rows=list(csv.DictReader(f));assert len(actual_rows)==len(expected_rows)
            for a,b in zip(actual_rows,expected_rows):
                for k,v in b.items():assert a[k]==('' if v is None else str(v))
    events=[json.loads(s) for s in (out/'DATA_ACCESS_EVENTS.jsonl').read_text().splitlines()]
    assert events and all(e['path'].startswith('development/') if e['phase']=='DEVELOPMENT' else e['path'].startswith(('calibration/','development/')) for e in events)
    assert cf['no_detector_selected_from_diagnostics'] and bf['status']=='PREPARED_AWAITING_INDEPENDENT_REVIEW_NO_TEST_AUTHORIZATION'
    log=json.loads((out/'DATA_ACCESS_LOG.json').read_text());assert not log['NASA_TEST_arrays_accessed'] and not log['anomaly_labels_accessed'] and log['accesses']==events_before
    if (out/'OUTPUT_MANIFEST.json').exists():
        # A completed rerun legitimately updates session/access logs before a new export;
        # manifest verification is required for standalone closed exports, not the in-process audit.
        pass
    return {'status':'PASS','committed_units':len(expected),'model_units':972,'persistence_validation_units':81,'qualification_units':verified,
      'new_LSTM_fits':243,'frozen_GCN_checkpoints_byte_and_state_verified':486,'historical_TCN_checkpoints_byte_and_state_verified':243,
      'detector_diagnostic_rows_recomputed':len(reports),'adaptive_windows_recomputed':windows,'no_NASA_TEST_or_labels':True,
      'decision':'PREPARED_AWAITING_REVIEW_NO_TEST_AUTHORIZATION','requires_original_input_bundle':True}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--results',type=Path,required=True);a.add_argument('--package',type=Path);args=a.parse_args()
    out=args.results
    if (out/'OUTPUT_MANIFEST.json').exists():
        for n,h in json.loads((out/'OUTPUT_MANIFEST.json').read_text())['files'].items():
            if sha(out/n)!=h:raise RuntimeError('Output export bytes changed '+n)
    print(json.dumps(audit(out,args.package),indent=2))
