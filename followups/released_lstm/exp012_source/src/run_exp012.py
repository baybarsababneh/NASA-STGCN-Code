"""TRAIN-only benchmark preparation. No TEST loader/evaluator or detector selector."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse,csv,hashlib,io,json,shutil,time,zipfile
from pathlib import Path
import numpy as np
import torch
from integrity import *
from models_io import *
from hundman_detector import detect,VERSION as DETECTOR_VERSION
from detector_metrics_v2 import higher_threshold,burden,VERSION as METRIC_VERSION
from engineering import run_checks,resume_check

ROOT=Path(__file__).resolve().parents[1]

def check_package():
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_MANIFEST.json'}
    if actual!=set(manifest['files']):raise RuntimeError('Package roster differs')
    for n,h in manifest['files'].items():
        if sha(ROOT/n)!=h:raise RuntimeError('Package bytes differ '+n)
    return manifest

def sidecar(path,out,rows):
    """Only indexed .pt entries plus manifest can be unpacked; reject path traversal."""
    expected={f"{r['channel']}_seed{s}.pt":(r,s) for r in rows for s in [17,29,43]}
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise RuntimeError('Duplicate ZIP entries')
        m=json.loads(z.read('TRANSFER_MANIFEST.json'))
        if m['purpose']!='EXP012_SELECTED_EXP005_WEIGHTS' or set(m['files'])!=set(expected):raise RuntimeError('Wrong or incomplete EXP005 sidecar')
        if set(z.namelist())!=set(expected)|{'TRANSFER_MANIFEST.json'}:raise RuntimeError('Unexpected sidecar payload')
        for name,h in m['files'].items():
            blob=z.read(name)
            if hashlib.sha256(blob).hexdigest()!=h:raise RuntimeError('Sidecar checkpoint hash mismatch '+name)
            dst=out/name
            if dst.exists():
                if sha(dst)!=h:raise RuntimeError('Changed sidecar after transfer '+name)
            else:atomic(dst,blob)
    return {'archive_sha256':sha(path),'manifest':m}

def bytes_pt(value):
    f=io.BytesIO();torch.save(value,f);return f.getvalue()
def bytes_npz(value):
    f=io.BytesIO();np.savez_compressed(f,**value);return f.getvalue()
def record_json(value):return json.dumps(normal(value),sort_keys=True,allow_nan=False).encode()

def trace(raw,original,row,pred,start,end,offset=0):
    k=np.arange(start,end);local=k-offset
    return {'target_index':k,'target_standardized':raw[local,0],'prediction_standardized':pred,
            'target':original[local,0].astype(np.float64),
            'prediction':pred.astype(np.float64)*row['target_scale']+row['target_mean']}

def write_table(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for r in records for k in r))
    import csv
    f=io.StringIO();w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(normal(records));atomic(path,f.getvalue().encode())

def static_preflight(root,out,rows,plan,expected_tcn):
    report=[]
    for row in rows:
        a=np.load(root/'development'/f"{row['channel']}.npy",allow_pickle=False)
        if not np.isfinite(a).all() or a.shape!=(row['stop_end'],row['columns']) or row['fit_end']<=64:
            raise RuntimeError('Invalid development array or no fit windows')
        mu=float(a[:row['fit_end'],0].mean());sd=float(a[:row['fit_end'],0].std());sd=sd if np.isfinite(sd) and sd>=1e-8 else 1.
        if (mu,sd)!=(row['target_mean'],row['target_scale']):raise RuntimeError('TRAIN-fit scaler changed')
        for meta in [m for m in plan if m['job']['channel']==row['channel']]:
            model,ck,r=read_parent(root,meta)
            if r['graph_sources']!=row['incoming_features'][1:] or (r['fit_end'],r['stop_end'])!=(row['fit_end'],row['stop_end']):raise RuntimeError('Frozen input/split mismatch')
            report.append({'kind':'frozen_gcn','job_id':meta['job_id'],'archive_sha256':meta['archive_sha256']});del model,ck
        for seed in [17,29,43]:
            p=out/'parent_weights/exp005'/f"{row['channel']}_seed{seed}.pt";model,ck,r=read_tcn(p,row,row['incoming_features'])
            er=expected_tcn[(row['channel'],seed)]
            if r['seed']!=seed:raise RuntimeError('Mislabeled EXP005 seed')
            for field in ['candidate_id','hidden_size','best_epoch','epochs_ran','parameters','selected_feature_columns']:
                if str(r[field])!=str(er[field]):raise RuntimeError('EXP005 historical result identity differs '+field)
            if not np.isclose(r['validation_mse'],float(er['validation_mse']),atol=1e-12,rtol=1e-10):raise RuntimeError('EXP005 source result differs')
            report.append({'kind':'historical_tcn','channel':row['channel'],'seed':seed,'checkpoint_sha256':sha(p)});del model,ck
    freeze(out/'PREFLIGHT.json',{'status':'PASS','parents_verified':729,'items':report})

def export(out):
    dest=out/'exports';dest.mkdir(exist_ok=True)
    entries=[p for p in out.rglob('*') if p.is_file() and 'exports' not in p.relative_to(out).parts
             and 'work' not in p.relative_to(out).parts and not p.name.startswith('ACTIVE_RUN') and '.tmp-' not in p.name]
    manifest={p.relative_to(out).as_posix():sha(p) for p in sorted(entries) if p.name!='OUTPUT_MANIFEST.json'}
    write_json(out/'OUTPUT_MANIFEST.json',{'files':manifest})
    entries.append(out/'OUTPUT_MANIFEST.json')
    for full,name in [(False,'EXP012_PREPARATION_summary.zip'),(True,'EXP012_PREPARATION_results.zip')]:
        tmp=dest/(name+'.tmp')
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for p in sorted(set(entries)):
                rel=p.relative_to(out)
                if full or rel.parts[0] not in ['units','parent_weights']:z.write(p,rel)
        os.replace(tmp,dest/name);atomic(dest/(name+'.sha256'),(sha(dest/name)+'  '+name+'\n').encode())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--exp005-sidecar',type=Path,required=True);ap.add_argument('--provisioning-record',type=Path,required=True)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if out==ROOT or ROOT in out.parents:raise ValueError('Output must be outside immutable package')
    check_package();rows=json.loads((ROOT/'data_manifest.json').read_text());cfg=json.loads((ROOT/'config.json').read_text())
    plan=json.loads((ROOT/'parent_plan.json').read_text())
    if len(rows)!=81 or len(plan)!=486:raise RuntimeError('Incorrect study inventory')
    side=sidecar(args.exp005_sidecar,out/'parent_weights/exp005',rows)
    env=configure_cuda()
    with Lease(out) as lease:
        identity={'experiment':'EXP012_ANOMALY_BENCHMARK_PREPARATION_V1','environment':env,
                  'package_manifest_sha256':sha(ROOT/'PACKAGE_MANIFEST.json'),'EXP005_input':side,'configuration':cfg}
        freeze(out/'RUN_FREEZE.json',identity);rh=digest(identity)
        guard=DataGuard(ROOT,out);guard.install()
        freeze(out/'ENGINEERING_GATE.json',{'standard':run_checks(),'CUDA_epoch_resume':resume_check('cuda')})
        # Provenance copied once, strictly byte-identical on resume.
        for p in sorted(ROOT.rglob('*')):
            if p.is_file() and p.suffix in {'.py','.json','.md','.txt','.csv'} and '__pycache__' not in p.parts:
                dest=out/'reproducibility'/p.relative_to(ROOT)
                if dest.exists() and sha(dest)!=sha(p):raise RuntimeError('Provenance copy changed')
                if not dest.exists():atomic(dest,p.read_bytes())
        provisioning=json.loads(args.provisioning_record.read_text())
        write_json(out/'sessions'/('provisioning_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json'),provisioning)
        session={'run_hash':rh,'utc':time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()),
                 'gpu_uuid':__import__('subprocess').check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()}
        write_json(out/'sessions'/(session['utc']+'.json'),session)
        parent={(m['job']['channel'],m['job']['seed'],m['job']['remove_messages']):m for m in plan}
        expected_tcn={(r['channel'],int(r['seed'])):r for r in csv.DictReader((ROOT/'provenance/EXP005_selected_results.csv').open())}
        static_preflight(ROOT,out,rows,plan,expected_tcn)
        roster=[];qualification=[]
        # Stage 1: audit/replay frozen parents and train only the controlled LSTM.
        for arm in ['gcn','removed','tcn','lstm']:
            for row in rows:
                original=np.load(ROOT/'development'/f"{row['channel']}.npy",allow_pickle=False)
                if original.shape!=(row['stop_end'],row['columns']):raise RuntimeError('Development shape changed')
                raw=standardize(original,row['target_mean'],row['target_scale']);features=row['incoming_features']
                for seed in [17,29,43]:
                    jid=f"{arm}__{row['channel']}__{seed}";uid={'run_hash':rh,'job_id':jid}
                    saved=read_commit(out,jid,uid)
                    if saved is None:
                        lease.check();clock=time.perf_counter();history=[];ref=None
                        if arm in ['gcn','removed']:
                            meta=parent[(row['channel'],seed,arm=='removed')];model,ck,ref=read_parent(ROOT,meta)
                            lookback=64;inputs=ref['graph_features'];origin_hash=meta['archive_sha256']
                            if ref['target_mean']!=row['target_mean'] or ref['target_scale']!=row['target_scale']:raise RuntimeError('Parent scaler mismatch')
                        elif arm=='tcn':
                            p=out/'parent_weights/exp005'/f"{row['channel']}_seed{seed}.pt";model,ck,ref=read_tcn(p,row,features)
                            lookback=ref['lookback'];inputs=features;origin_hash=sha(p)
                            er=expected_tcn[(row['channel'],seed)]
                            for field in ['candidate_id','hidden_size','best_epoch','epochs_ran','parameters','selected_feature_columns']:
                                if str(ref[field])!=str(er[field]):raise RuntimeError('EXP005 historical CSV identity differs '+field)
                            if not np.isclose(ref['validation_mse'],float(er['validation_mse']),atol=1e-12,rtol=1e-10):raise RuntimeError('EXP005 historical CSV value differs')
                        else:
                            model,ck,history,_=train_lstm(raw,row,features,seed,cfg['lstm'],out/'work'/jid/'CURRENT.json',uid)
                            lookback=cfg['lstm']['lookback'];inputs=features;origin_hash=None
                        model=model.to('cuda');state_before=tensor_hash(model.state_dict())
                        pred,mse=predict(model,raw,row,inputs,row['fit_end'],row['stop_end'],lookback)
                        repeated,mse2=predict(model,raw,row,inputs,row['fit_end'],row['stop_end'],lookback)
                        if not np.array_equal(pred,repeated) or mse!=mse2:raise RuntimeError('Prediction repeatability gate failed')
                        if arm in ['gcn','removed'] and mse!=ref['early_stop_mse']:raise RuntimeError('Frozen GCN numerical replay gate failed; no tolerance relaxation')
                        if tensor_hash(model.state_dict())!=state_before:raise RuntimeError('Frozen state changed')
                        ceiling=cfg['lstm']['max_epochs'] if arm=='lstm' else (ck['experiment_config']['training_common']['max_epochs'] if arm=='tcn' else ck['config']['training_common']['max_epochs'])
                        rec={'arm':arm,'channel':row['channel'],'mission':row['mission'],'seed':seed,'validation_mse':mse,
                             'historical_validation_mse':ref['validation_mse'] if ref else None,'parent_sha256':origin_hash,
                             'lookback':lookback,'features':inputs,'state_sha256':state_before,'parameters':sum(p.numel() for p in model.parameters()),
                             'exact_repeat':True,'parent_exact_mse':arm in ['gcn','removed'],'elapsed_seconds':time.perf_counter()-clock,
                             'best_epoch':(ref or ck['result'])['best_epoch'],'epochs_ran':(ref or ck['result'])['epochs_ran'],
                             'epoch_ceiling':ceiling,'hit_epoch_ceiling':(ref or ck['result'])['epochs_ran']==ceiling}
                        files={'record.json':record_json(rec),'validation.npz':bytes_npz(trace(raw,original,row,pred,row['fit_end'],row['stop_end']))}
                        if arm=='lstm':files.update({'checkpoint.pt':bytes_pt(ck),'history.json':record_json(history)})
                        saved=commit(out,jid,uid,files)
                        del model;torch.cuda.empty_cache()
                    rec=json.loads((saved/'record.json').read_text());roster.append({'job_id':jid,**rec,'commit_sha256':sha(saved/'COMMIT.json')})
                    print(f"[weights {len(roster)}/972] verified {jid}",flush=True)
                    write_json(out/'STATUS.json',{'status':'MODEL_PREPARATION','verified':len(roster),'total':972})
                if arm!='lstm':continue
                # Deterministic persistence is calibrated once, not independent seed replications.
                jid=f"persistence__{row['channel']}";uid={'run_hash':rh,'job_id':jid}
                saved=read_commit(out,jid,uid)
                if saved is None:
                    a=row['fit_end'];b=row['stop_end'];pred=raw[a-1:b-1,0].copy()
                    commit(out,jid,uid,{'record.json':record_json({'arm':'persistence','channel':row['channel'],'mission':row['mission'],'seed':None,
                            'lookback':1,'features':[0],'validation_mse':float(np.mean((pred-raw[a:b,0])**2))}),
                            'validation.npz':bytes_npz(trace(raw,original,row,pred,a,b))})
        if len(roster)!=972:raise RuntimeError('Model inventory incomplete')
        freeze(out/'MODEL_FREEZE.json',{'run_hash':rh,'forecasting_fits_new':243,'frozen_gcn_parents':486,'historical_tcn_parents':243,
               'weights':roster,'persistence_channels':81,'source_manifest_sha256':sha(ROOT/'PACKAGE_MANIFEST.json')})
        write_table(out/'tables/model_inventory.csv',roster)
        guard.enable_calibration()
        # Stage 2: one qualification/calibration pass, never a selector or retuning loop.
        thresholds=[]
        for row in rows:
            orig=np.load(ROOT/'calibration'/f"{row['channel']}.npy",allow_pickle=False);offset=row['calibration_offset']
            if orig.shape!=(row['rows']-offset,row['columns']):raise RuntimeError('Calibration shape differs')
            raw=standardize(orig,row['target_mean'],row['target_scale'])
            for arm,seed in [(a,s) for a in ['gcn','removed','tcn','lstm'] for s in [17,29,43]]+[('persistence',None)]:
                stem=f"{arm}__{row['channel']}"+(f"__{seed}" if seed is not None else '')
                jid='qualification__'+stem;uid={'run_hash':rh,'job_id':jid,'model_freeze_sha256':sha(out/'MODEL_FREEZE.json')}
                saved=read_commit(out,jid,uid)
                if saved is None:
                    lease.check();a=row['stop_end'];b=row['rows'];model=None
                    if arm in ['gcn','removed']:
                        model,ck,r=read_parent(ROOT,parent[(row['channel'],seed,arm=='removed')]);inputs=r['graph_features'];lookback=64
                    elif arm=='tcn':model,ck,r=read_tcn(out/'parent_weights/exp005'/f"{row['channel']}_seed{seed}.pt",row,row['incoming_features']);inputs=row['incoming_features'];lookback=64
                    elif arm=='lstm':
                        ck=torch.load(out/'units'/stem/'checkpoint.pt',map_location='cpu',weights_only=True)
                        model=ControlledLSTM(len(row['incoming_features']));model.load_state_dict(ck['state_dict'],strict=True);inputs=row['incoming_features'];lookback=cfg['lstm']['lookback']
                    if model is None:pred=raw[a-offset-1:b-offset-1,0].copy()
                    else:
                        model=model.to('cuda');before=tensor_hash(model.state_dict());pred,_=predict(model,raw,row,inputs,a,b,lookback,offset=offset)
                        again,_=predict(model,raw,row,inputs,a,b,lookback,offset=offset)
                        if not np.array_equal(pred,again) or tensor_hash(model.state_dict())!=before:raise RuntimeError('Calibration replay changed state/results')
                    cal=trace(raw,orig,row,pred,a,b,offset)
                    residual=np.abs(cal['target_standardized'].astype(np.float64)-pred.astype(np.float64))
                    threshold=higher_threshold(residual,.99)
                    with np.load(out/'units'/stem/'validation.npz',allow_pickle=False) as zz:val={k:zz[k] for k in zz.files}
                    reports=[];arrays=dict(cal);window_records={}
                    for split,stream in [('validation',val),('calibration',cal)]:
                        errors=np.abs(stream['target_standardized'].astype(np.float64)-stream['prediction_standardized'].astype(np.float64))
                        raw_flags=errors>threshold['threshold'];d=detect(stream['target'],stream['prediction'],stream['target_index'],cfg['detector'])
                        for detector,flags in [('raw_q99',raw_flags),('hundman_adapt_v1',d['flags'])]:
                            reports.append({'arm':arm,'channel':row['channel'],'mission':row['mission'],'seed':seed,'split':split,'detector':detector,
                                            **burden(flags),'detector_status':d['status'] if detector=='hundman_adapt_v1' else 'OUTPUT_COMPLETE'})
                        arrays[split+'_adaptive_flags']=d['flags'];arrays[split+'_issue_index']=d['issue_index'];arrays[split+'_raw_flags']=raw_flags
                        arrays[split+'_raw_issue_index']=np.where(raw_flags,stream['target_index'],-1)
                        arrays[split+'_smoothed_raw_residual']=d['smoothed']
                        arrays[split+'_target_index']=stream['target_index']
                        window_records[split]={'windows':d['windows'],'status':d['status'],'startup_averaging_lookahead':d['startup_averaging_lookahead']}
                    record={'job_id':stem,'arm':arm,'channel':row['channel'],'mission':row['mission'],'seed':seed,
                            'threshold':threshold,'detector_reports':reports,'threshold_units':'fit-standardized absolute telemetry residual',
                            'calibration_target_start':a,'calibration_target_end_exclusive':b,'adaptive_window_audit':window_records}
                    saved=commit(out,jid,uid,{'record.json':record_json(record),'calibration_and_diagnostics.npz':bytes_npz(arrays)})
                    if model is not None:del model;torch.cuda.empty_cache()
                r=json.loads((saved/'record.json').read_text());thresholds.append({k:v for k,v in r.items() if k not in ['detector_reports','adaptive_window_audit']});qualification+=r['detector_reports']
                print(f"[qualification {len(thresholds)}/1053] verified {stem}",flush=True)
                write_json(out/'STATUS.json',{'status':'CALIBRATION_QUALIFICATION','verified':len(thresholds),'total':1053})
        if len(thresholds)!=1053 or len(qualification)!=4212:raise RuntimeError('Qualification roster incomplete')
        write_table(out/'tables/nominal_diagnostics.csv',qualification)
        write_table(out/'tables/thresholds.csv',[{**{k:v for k,v in r.items() if k!='threshold'},**r['threshold']} for r in thresholds])
        from summaries import forecast_rows,nominal_rows,resolution_rows
        persistence_records=[json.loads((out/'units'/f"persistence__{r['channel']}"/'record.json').read_text()) for r in rows]
        write_table(out/'tables/forecasting_summary.csv',forecast_rows(roster+persistence_records))
        write_table(out/'tables/nominal_detector_summary.csv',nominal_rows(qualification))
        write_table(out/'tables/calibration_resolution_summary.csv',resolution_rows(thresholds))
        freeze(out/'CALIBRATION_FREEZE.json',{'run_hash':rh,'thresholds':thresholds,'detector_settings':cfg['detector'],
               'nominal_diagnostics_sha256':sha(out/'tables/nominal_diagnostics.csv'),'no_detector_selected_from_diagnostics':True})
        freeze(out/'BENCHMARK_FREEZE.json',{'status':'PREPARED_AWAITING_INDEPENDENT_REVIEW_NO_TEST_AUTHORIZATION','run_hash':rh,
               'model_freeze_sha256':sha(out/'MODEL_FREEZE.json'),'calibration_freeze_sha256':sha(out/'CALIBRATION_FREEZE.json'),
               'evaluation_contract':json.loads((ROOT/'evaluation_contract.json').read_text()),'metric_version':METRIC_VERSION,
               'detector_version':DETECTOR_VERSION,'data_manifest_sha256':sha(ROOT/'data_manifest.json')})
        write_json(out/'DATA_ACCESS_LOG.json',{'accesses':[json.loads(line) for line in (out/'DATA_ACCESS_EVENTS.jsonl').read_text().splitlines()],'NASA_TEST_arrays_accessed':False,'anomaly_labels_accessed':False,
               'final_TRAIN_calibration_used_after_model_freeze':True,'final_TRAIN_is_not_globally_virgin':True})
        from audit_exp012 import audit
        close={'status':'COMPLETE_PREPARED_AWAITING_REVIEW','new_LSTM_fits':243,'new_GCN_fits':0,'model_units':972,
               'qualification_units':1053,'nominal_diagnostic_rows':4212,'NASA_TEST_accessed':False,'anomaly_labels_accessed':False,
               'detector_selected_by_performance':False,'thresholds_locked':True,'TEST_evaluation_authorized':False}
        freeze(out/'STUDY_CLOSE.json',close);write_json(out/'STATUS.json',close)
        atomic(out/'SUMMARY.md',b'# EXP012 preparation complete\n\n243 controlled LSTM fits; zero GCN refits. Frozen parents and all calibration units verified.\n\nBoth prespecified detectors are retained, independent of nominal alarm rates. No candidate was promoted by these diagnostics. No NASA TEST evaluation occurred. Final TRAIN calibration is historically exposed.\n\nSTOP: independent methodological review and a separately authorised, label-blind inference stage are required. This package contains no TEST runner.\n')
        write_json(out/'AUDIT_RESULT.json',audit(out,ROOT))
        write_json(out/'DATA_ACCESS_LOG.json',{'accesses':[json.loads(line) for line in (out/'DATA_ACCESS_EVENTS.jsonl').read_text().splitlines()],
               'NASA_TEST_arrays_accessed':False,'anomaly_labels_accessed':False,'final_TRAIN_calibration_used_after_model_freeze':True,'final_TRAIN_is_not_globally_virgin':True})
        lease.check();export(out)
        print(json.dumps(close,indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        print('SAFE STOP: preserve committed units and epoch progress; never relax gates or retune against diagnostics to resume.',flush=True)
        raise
