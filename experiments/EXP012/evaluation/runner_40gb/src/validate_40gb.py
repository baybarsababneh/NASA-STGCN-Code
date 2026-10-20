"""No NASA TEST or labels. Exact frozen nominal replay, never training/refitting."""
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1');sys.dont_write_bytecode=True
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from common import *
from runtime_compatibility import check_environment,identity,require_pass
ROOT=Path(__file__).resolve().parents[1]

class NominalOnlyGuard:
    def __init__(self,package,results,out):
        self.allowed={Path(package).resolve()/'development',Path(package).resolve()/'calibration'};self.references=Path(results).resolve()/'units';self.out=Path(out)
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.suffix.lower()=='.csv' or 'labeled_anomal' in p.name.lower() or 'labelled_anomal' in p.name.lower() or p.name.lower()=='archive.zip':raise PermissionError('Labels/NASA archive forbidden during compatibility validation')
        if p.suffix.lower()=='.npy':
            if p.parent not in self.allowed:raise PermissionError('Only frozen nominal arrays allowed during compatibility validation')
            with open(self.out/'NOMINAL_ACCESS_EVENTS.jsonl','a') as f:f.write(json.dumps({'path':str(p),'mode':'TRAIN_ONLY_COMPATIBILITY_REPLAY'})+'\n');f.flush();os.fsync(f.fileno())
        if p.suffix.lower()=='.npz' and self.references not in p.parents:raise PermissionError('Only saved nominal trace references allowed')
    def install(self):sys.addaudithook(self.hook)

def array_hash(a):
    a=np.asarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes(order='C'));return h.hexdigest()

def exact_array(actual,expected,name):
    a=np.asarray(actual);b=np.asarray(expected)
    if a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes(order='C')!=b.tobytes(order='C'):raise RuntimeError('Bitwise nominal replay mismatch: '+name+'; zero tolerance; no authorization')
    return array_hash(a)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--package',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--validate-nominal-only',action='store_true');a=ap.parse_args()
    if not a.validate_nominal_only:raise RuntimeError('Explicit --validate-nominal-only required')
    r=a.results.resolve();p=a.package.resolve();out=a.output.resolve()
    if out==r or r in out.parents or out==p or p in out.parents or out in r.parents or out in p.parents:raise RuntimeError('Keep compatibility evidence separate from frozen inputs')
    verify_manifest(ROOT,'RUNNER_MANIFEST.json');binding=json.loads((ROOT/'EVALUATION_BINDING.json').read_text());amendment=json.loads((ROOT/'COMPATIBILITY_AMENDMENT.json').read_text())
    verify_manifest(r,'OUTPUT_MANIFEST.json');verify_manifest(p,'PACKAGE_MANIFEST.json')
    for key,file in [('results_manifest_sha256',r/'OUTPUT_MANIFEST.json'),('model_freeze_sha256',r/'MODEL_FREEZE.json'),('calibration_freeze_sha256',r/'CALIBRATION_FREEZE.json'),('benchmark_freeze_sha256',r/'BENCHMARK_FREEZE.json'),('package_manifest_sha256',p/'PACKAGE_MANIFEST.json')]:
        if sha(file)!=binding[key]:raise RuntimeError('Frozen preparation binding differs '+key)
    if json.loads((p/'evaluation_contract.json').read_text())!=binding['evaluation_contract']:raise RuntimeError('Frozen scientific contract differs')
    sys.path.insert(0,str(r/'reproducibility/src'))
    import torch
    from integrity import configure_cuda
    from models_io import read_parent,read_tcn,standardize,predict,tensor_hash
    from lstm_baseline import ControlledLSTM
    from hundman_detector import detect
    from detector_metrics_v2 import burden
    env=configure_cuda();reference=json.loads((r/'RUN_FREEZE.json').read_text())['environment'];check_environment(env,reference,amendment)
    cfg=json.loads((p/'config.json').read_text());rows=json.loads((p/'data_manifest.json').read_text());plan=json.loads((p/'parent_plan.json').read_text());parents={(v['job']['channel'],v['job']['seed'],v['job']['remove_messages']):v for v in plan}
    mf=json.loads((r/'MODEL_FREEZE.json').read_text());cf=json.loads((r/'CALIBRATION_FREEZE.json').read_text());weights={v['job_id']:v for v in mf['weights']};thresholds={v['job_id']:v['threshold']['threshold'] for v in cf['thresholds']}
    if len(rows)!=81 or len(weights)!=972 or len(thresholds)!=1053:raise RuntimeError('Original scientific roster incomplete')
    out.mkdir(parents=True,exist_ok=True);run=identity(env,r,p,ROOT);freeze(out/'COMPATIBILITY_RUN_FREEZE.json',run)
    if (out/'COMPATIBILITY_PASS.json').exists():require_pass(out,env,r,p,ROOT);print('EXISTING COMPLETE 40GB REPLAY VERIFIED; no repeated replay');return
    with Lease(out):
        NominalOnlyGuard(p,r,out).install();roster=[];window_count=0
        for row in rows:
            dev=np.load(p/'development'/f"{row['channel']}.npy",allow_pickle=False);cal=np.load(p/'calibration'/f"{row['channel']}.npy",allow_pickle=False)
            if dev.shape!=(row['stop_end'],row['columns']) or cal.shape!=(row['rows']-row['calibration_offset'],row['columns']) or not np.isfinite(dev).all() or not np.isfinite(cal).all():raise RuntimeError('Invalid frozen nominal input')
            zd=standardize(dev,row['target_mean'],row['target_scale']);zc=standardize(cal,row['target_mean'],row['target_scale'])
            for arm,seed in [(arm,s) for arm in ['gcn','removed','tcn','lstm'] for s in [17,29,43]]+[('persistence',None)]:
                stem=f"{arm}__{row['channel']}"+(f'__{seed}' if seed is not None else '');uid={'run_hash':digest(run),'job_id':stem};saved=reuse(out,stem,uid)
                if saved is None:
                    model=None;before=None;features=[0];lookback=1
                    torch.cuda.reset_peak_memory_stats()
                    if arm in ['gcn','removed']:model,ck,ref=read_parent(p,parents[(row['channel'],seed,arm=='removed')]);features=ref['graph_features'];lookback=64
                    elif arm=='tcn':model,ck,ref=read_tcn(r/'parent_weights/exp005'/f"{row['channel']}_seed{seed}.pt",row,row['incoming_features']);features=row['incoming_features'];lookback=64
                    elif arm=='lstm':
                        ck=torch.load(r/'units'/stem/'checkpoint.pt',map_location='cpu',weights_only=True);model=ControlledLSTM(len(row['incoming_features']));model.load_state_dict(ck['state_dict'],strict=True);features=row['incoming_features'];lookback=64
                    if model is not None:
                        model=model.to('cuda');before=tensor_hash(model.state_dict())
                        if before!=weights[stem]['state_sha256'] or features!=weights[stem]['features'] or lookback!=weights[stem]['lookback']:raise RuntimeError('Frozen state/features/context changed')
                    v=loadnpz(r/'units'/stem/'validation.npz');qref=loadnpz(r/'units'/('qualification__'+stem)/'calibration_and_diagnostics.npz');qr=json.loads((r/'units'/('qualification__'+stem)/'record.json').read_text());records={};windows=0
                    for split,original,raw,offset,start,end,reftrace in [('validation',dev,zd,0,row['fit_end'],row['stop_end'],v),('calibration',cal,zc,row['calibration_offset'],row['stop_end'],row['rows'],qref)]:
                        if model is None:
                            pred=raw[start-offset-1:end-offset-1,0].copy();mse=float(np.mean((pred-raw[start-offset:end-offset,0])**2));again=pred.copy();mse2=mse
                        else:
                            pred,mse=predict(model,raw,row,features,start,end,lookback,offset=offset);again,mse2=predict(model,raw,row,features,start,end,lookback,offset=offset)
                        exact_array(pred,again,stem+' '+split+' repeat')
                        if mse!=mse2 or model is not None and tensor_hash(model.state_dict())!=before:raise RuntimeError('Repeatability/state mismatch')
                        if split=='validation':
                            expected_mse=weights[stem]['validation_mse'] if model is not None else json.loads((r/'units'/stem/'record.json').read_text())['validation_mse']
                            if mse!=expected_mse:raise RuntimeError('Exact saved validation MSE mismatch '+stem)
                        ix=np.arange(start,end,dtype=np.int64);local=ix-offset
                        trace={'target_index':ix,'target_standardized':raw[local,0],'prediction_standardized':pred,'target':original[local,0].astype(np.float64),'prediction':pred.astype(np.float64)*row['target_scale']+row['target_mean']}
                        checks={k:exact_array(trace[k],reftrace[k],stem+' '+split+' '+k) for k in trace}
                        errors=np.abs(trace['target_standardized'].astype(np.float64)-pred.astype(np.float64));locked_flags=errors>thresholds[stem]
                        # Thresholds are consumed unchanged; never recalculated or selected.
                        expected_errors=np.abs(reftrace['target_standardized'].astype(np.float64)-reftrace['prediction_standardized'].astype(np.float64));checks['raw_residual']=exact_array(errors,expected_errors,stem+' '+split+' residual')
                        d=detect(trace['target'],trace['prediction'],ix,cfg['detector'])
                        for suffix,value in [('raw_flags',locked_flags),('raw_issue_index',np.where(locked_flags,ix,-1)),('adaptive_flags',d['flags']),('issue_index',d['issue_index']),('smoothed_raw_residual',d['smoothed']),('target_index',ix)]:checks[suffix]=exact_array(value,qref[split+'_'+suffix],stem+' '+split+' '+suffix)
                        window_audit={'windows':d['windows'],'status':d['status'],'startup_averaging_lookahead':d['startup_averaging_lookahead']}
                        if digest(window_audit)!=digest(qr['adaptive_window_audit'][split]):raise RuntimeError('Adaptive window/timing audit mismatch')
                        for detector,flags in [('raw_q99',locked_flags),('hundman_adapt_v1',d['flags'])]:
                            record=next(v for v in qr['detector_reports'] if v['split']==split and v['detector']==detector)
                            if any(record[k]!=value for k,value in burden(flags).items()):raise RuntimeError('Saved detector diagnostics mismatch')
                        windows+=len(d['windows']);records[split]={'exact_array_hashes':checks,'mse':mse,'window_audit_sha256':digest(window_audit),'target_start':start,'target_end_exclusive':end,'exact_pass':True}
                    rec={'job_id':stem,'exact_pass':True,'selected_state_sha256':before,'splits':records,'adaptive_windows':windows,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'threshold_used_unchanged':thresholds[stem],'NASA_TEST_opened':False,'NASA_labels_opened':False,'scientific_fits':0}
                    saved=commit(out,stem,uid,{'REPLAY_RECORD.json':json.dumps(normal(rec),sort_keys=True,allow_nan=False).encode()})
                    if model is not None:del model;torch.cuda.empty_cache()
                rec=json.loads((saved/'REPLAY_RECORD.json').read_text());window_count+=rec['adaptive_windows'];roster.append(stem);print(f'[40GB nominal replay {len(roster)}/1053] {stem}',flush=True)
        if set(roster)!=set(thresholds) or len(roster)!=1053 or window_count!=2080 or {v.name for v in (out/'units').iterdir() if v.is_dir()}!=set(roster):raise RuntimeError('Replay coverage incomplete')
        files={v.relative_to(out).as_posix():sha(v) for v in sorted((out/'units').rglob('*')) if v.is_file() and '.tmp-' not in v.name}
        freeze(out/'COMPATIBILITY_PASS.json',{'status':'PASS_40GB_NOMINAL_BITWISE_REPLAY','identity':run,'run_freeze_sha256':sha(out/'COMPATIBILITY_RUN_FREEZE.json'),'roster':roster,'files':files,'counts':{'streams':1053,'checkpoint_states':972,'nominal_split_streams':2106,'detector_split_streams':4212,'adaptive_windows':window_count,'mismatches':0,'scientific_fits':0},'NASA_TEST_opened':False,'NASA_labels_opened':False,'scientific_settings_changed':False})
        require_pass(out,env,r,p,ROOT)
        print('40GB NOMINAL COMPATIBILITY PASS. This matching runtime is authorised for the unchanged Phase A computation. No NASA TEST/labels were opened.')

if __name__=='__main__':main()
