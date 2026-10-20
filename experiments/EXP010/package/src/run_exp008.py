"""EXP008: saved-weight replay then descriptive TRAIN-only nominal feasibility.
No optimiser, fitting, final calibration, model selector or TEST evaluator exists here.
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse,hashlib,io,json,sys,time,uuid,zipfile,subprocess
from pathlib import Path
import numpy as np
import torch
from common import sha,digest,atomic_bytes,write_json,freeze_json
from execution import configure_cuda,Lease
from models_exp007 import MessageGatedGCN
from detector_metrics_v2 import VERSION
from test_exp008 import tests
from nominal_study import records_for_trace,summarise
from replay_io import commit,read_commit
ROOT=Path(__file__).resolve().parents[1]

def check_package():
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
    found={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name!='PACKAGE_MANIFEST.json' and '__pycache__' not in p.parts}
    assert found==set(manifest['files']),'Package roster mismatch'
    for rel,h in manifest['files'].items():assert sha(ROOT/rel)==h,rel
    assert len(list((ROOT/'train_prefix').glob('*.npy')))==81
    return manifest

def guard():
    # No labels/other data arrays may be read. Final20% values are physically absent.
    allow=(ROOT/'train_prefix').resolve()
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0]));n=p.name.lower()
        if n.endswith('.npy') and p.resolve().parent!=allow:raise PermissionError('Only truncated TRAIN inputs allowed')
        if 'labeled_anomal' in n or 'labelled_anomal' in n:raise PermissionError('Label access denied')
    sys.addaudithook(hook)

def load_arrays(meta,row):
    r=meta['reference_result'];raw=np.load(ROOT/row['prefix_file'],allow_pickle=False).astype(np.float32)
    assert len(raw)==row['prefix_rows'] and r['num_values']==row['rows']
    mu=float(raw[:r['fit_end'],0].mean());sd=float(raw[:r['fit_end'],0].std());sd=sd if np.isfinite(sd) and sd>=1e-8 else 1.
    assert mu==r['target_mean'] and sd==r['target_scale'],'Scaler mismatch'
    raw[:,0]=(raw[:,0]-mu)/sd
    return raw

@torch.no_grad()
def predict(model,raw,features,start,end,lookback=64):
    assert lookback<=start<end<=len(raw)
    times=np.arange(start,end,dtype=np.int64)
    windows=np.lib.stride_tricks.sliding_window_view(raw,lookback,axis=0)
    pred=[];total=0.;model.eval();loss_fn=torch.nn.MSELoss()
    for at in range(0,len(times),256):
        ts=times[at:at+256]
        xb=np.ascontiguousarray(windows[ts-lookback][:,features,:],dtype=np.float32)
        yb=np.ascontiguousarray(raw[ts,0],dtype=np.float32)
        p=model(torch.from_numpy(xb).to('cuda'));loss=loss_fn(p,torch.from_numpy(yb).to('cuda'))
        if not torch.isfinite(p).all() or not torch.isfinite(loss):raise RuntimeError('Nonfinite prediction')
        total+=float(loss.item())*len(ts);pred.append(p.cpu().numpy())
    return dict(target_index=times,prediction=np.concatenate(pred),target=raw[times,0].copy(),persistence=raw[times-1,0].copy()),total/len(times)

def replay(meta,row,graphrows):
    blob=(ROOT/meta['archive']).read_bytes();assert hashlib.sha256(blob).hexdigest()==meta['archive_sha256']
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        assert set(z.namelist())=={'checkpoint.pt','result.json','job.json','history.json'}
        ck=torch.load(io.BytesIO(z.read('checkpoint.pt')),map_location='cpu',weights_only=True)
        r=json.loads(z.read('result.json'));j=json.loads(z.read('job.json'))
    assert r==meta['reference_result']==ck['result'] and j==meta['job']==ck['job'] and ck['run_hash']==meta['parent_run_hash']
    assert ck['config']==json.loads((ROOT/'provenance/effective_config.json').read_text())
    spec=ck['spec'];fraction={'canonical':.6,'R1':.4,'R2':.5}[j['fold']];g=graphrows[(j['channel'],fraction)]
    assert r['graph_features']==g['features'] and digest(ck['state_dict']['adjacency'].tolist())==g['adjacency_sha256']
    assert spec==ck['config']['baseline_by_mission'][j['mission']]
    model=MessageGatedGCN(r['graph_num_nodes'],r['hidden_size'],ck['state_dict']['adjacency'].numpy(),spec['kernel_size'],spec['dilations'],j['gate_mode'],j['remove_messages'])
    model.load_state_dict(ck['state_dict'],strict=True);model=model.to('cuda');raw=load_arrays(meta,row)
    intervals=[('stop',r['fit_end'],r['stop_end'],r['early_stop_mse'])]
    if j['stage']=='robust':intervals.append(('assessment',r['stop_end'],r['assess_end'],r['assessment_mse']))
    all_arrays=[];metrics={};start_clock=time.perf_counter()
    for name,a,b,reference in intervals:
        first,mse=predict(model,raw,r['graph_features'],a,b);second,mse2=predict(model,raw,r['graph_features'],a,b)
        if mse!=reference:raise RuntimeError(f"Numerical replay mismatch {meta['job_id']} {name}: {mse!r} versus {reference!r}. No tolerance relaxation or downstream scoring permitted.")
        if mse2!=mse or any(not np.array_equal(first[k],second[k]) for k in first):raise RuntimeError('Repeat prediction mismatch')
        metrics[name]={'mse':mse,'reference_mse':reference,'exact_match':True,'repeat_prediction_exact':True}
        all_arrays.append(first)
    # State tensors cannot be updated by this read-only workflow.
    assert all(torch.equal(v.cpu(),ck['state_dict'][k]) for k,v in model.state_dict().items())
    arrays={k:np.concatenate([a[k] for a in all_arrays]) for k in all_arrays[0]}
    record=dict(job_id=meta['job_id'],job=j,parent_archive_sha256=meta['archive_sha256'],metrics=metrics,
                elapsed_seconds=time.perf_counter()-start_clock,target_start=int(arrays['target_index'][0]),target_end_exclusive=int(arrays['target_index'][-1])+1)
    del model;torch.cuda.empty_cache()
    return record,arrays


def export(out):
    folder=out/'exports';folder.mkdir(exist_ok=True)
    items=[p for p in out.rglob('*') if p.is_file() and 'exports' not in p.relative_to(out).parts and not p.name.startswith('ACTIVE_RUN') and '.tmp-' not in p.name]
    for full,name in [(False,'EXP008_summary.zip'),(True,'EXP008_results.zip')]:
        tmp=folder/(name+'.tmp');final=folder/name
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
            for p in sorted(items):
                if full or 'replays' not in p.relative_to(out).parts:z.write(p,p.relative_to(out))
        os.replace(tmp,final);atomic_bytes(folder/(name+'.sha256'),(sha(final)+'  '+name+'\n').encode())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if out==ROOT or ROOT in out.parents:raise ValueError('Output must be outside immutable input package')
    manifest=check_package();guard();env=configure_cuda();parent=json.loads((ROOT/'provenance/RUN_FREEZE.json').read_text())
    if env!=parent['environment']:
        differences={k:{'expected':parent['environment'].get(k),'actual':env.get(k)} for k in set(env)|set(parent['environment']) if env.get(k)!=parent['environment'].get(k)}
        write_json(out/'ENVIRONMENT_MISMATCH.json',differences);raise RuntimeError('Environment differs from governing GPU execution; preserve mismatch record. No automatic fallback.')
    with Lease(out) as lease:
        config=json.loads((ROOT/'config.json').read_text());freeze=dict(experiment='EXP008_READINESS_V1',environment=env,package_manifest_sha256=sha(ROOT/'PACKAGE_MANIFEST.json'),protocol_sha256=sha(ROOT/'PROTOCOL.md'),config=config,parent_run_hash=digest(parent))
        freeze_json(out/'RUN_FREEZE.json',freeze);rh=digest(freeze);plan=json.loads((ROOT/'plan.json').read_text());assert len(plan)==534
        freeze_json(out/'PLAN.json',plan);freeze_json(out/'metric_synthetic_checks.json',tests())
        import shutil
        for rel in ['PROTOCOL.md','config.json','PACKAGE_MANIFEST.json','data_prefix_manifest.json']:
            dst=out/'reproducibility'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dst)
        for src in (ROOT/'src').glob('*.py'):
            dst=out/'reproducibility/src'/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        data={r['channel']:r for r in json.loads((ROOT/'data_prefix_manifest.json').read_text())};graphs={(g['channel'],g['fit_fraction']):g for g in json.loads((ROOT/'provenance/graph_manifest.json').read_text())}
        session={'utc':time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()),'run_hash':rh,'gpu_unit':subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv'],text=True)}
        write_json(out/'sessions'/(session['utc']+'.json'),session)
        replay_rows=[]
        for i,meta in enumerate(plan):
            lease.check();saved=read_commit(out,meta,rh)
            if saved is None:
                record,arrays=replay(meta,data[meta['job']['channel']],graphs);commit(out,meta,rh,record,arrays,lease)
                print(f"[replay {i+1}/534] committed {meta['job_id']}",flush=True)
            else:record,arrays=saved;print(f"[replay {i+1}/534] verified resume {meta['job_id']}",flush=True)
            replay_rows.append(record);write_json(out/'STATUS.json',dict(status='REPLAYING',committed=i+1,total=534,run_hash=rh))
        expected={m['job_id'] for m in plan};actual={p.name.split('.')[0] for p in (out/'replays').glob('*.done.json')};assert expected==actual
        # Nominal calculations cannot begin until ALL exact replay gates passed.
        freeze_json(out/'REPLAY_GATE.json',dict(status='PASS',jobs=534,exact_mse_and_repeat_predictions=True,run_hash=rh))
        rows=[];persistent_seen=set()
        for meta in plan:
            if meta['job']['stage']!='robust':continue
            record,arrays=read_commit(out,meta,rh);rows+=records_for_trace(meta,arrays,'gcn');key=(meta['job']['channel'],meta['job']['fold'])
            if key not in persistent_seen:rows+=records_for_trace(meta,arrays,'persistence');persistent_seen.add(key)
        assert len(persistent_seen)==24;summarise(rows,out/'tables')
        import pandas as pd
        pd.DataFrame([dict(job_id=r['job_id'],**r['job'],elapsed_seconds=r['elapsed_seconds'],**{k+'_mse':v['mse'] for k,v in r['metrics'].items()}) for r in replay_rows]).to_csv(out/'tables/replay_results.csv',index=False)
        freeze_json(out/'STUDY_CLOSE.json',dict(status='COMPLETE_NO_DETECTOR_SELECTED',checkpoint_replay_jobs=534,new_model_fits=0,nominal_rows=288,
            scorer_family={'spans':[1,10],'q':[.99,.995],'quantile':'higher','comparison':'strict >'},
            final_calibration_accessed=False,NASA_TEST_accessed=False,anomaly_labels_accessed=False,
            architecture_reopened=False,automatic_promotion=False,metric_version=VERSION,
            next_gate='Independent reviewer sign-off and bounded detector/external-baseline protocol; final TEST is not authorised by this script.'))
        (out/'SUMMARY.md').write_text('# EXP008 readiness study complete\n\n534 saved-weight replay jobs passed exact recorded-MSE and repeat-prediction checks. Zero training fits.\n\nThe four scorer/threshold combinations are descriptive nominal TRAIN diagnostics, not selected anomaly detectors. Persistence has one record per channel/origin, not duplicated seeds. No anomaly recall, precision, F-score or AP was computed on NASA data.\n\nSee tables/ for occupancy, alarm runs, quantile resolution and conditional paired intervals. Small blocks and reused origins limit inference. Neither quietness nor a narrow bootstrap interval establishes anomaly sensitivity.\n\nSTOP: submit this addendum for review. Final calibration, detector selection, external baselines and retrospective TEST evaluation require separately frozen stages.\n')
        lease.check();write_json(out/'STATUS.json',dict(status='COMPLETE_NO_DETECTOR_SELECTED',committed=534,total=534,run_hash=rh));export(out)
        print((out/'SUMMARY.md').read_text(),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        print('Stopped safely. Preserve committed replays. Do not relax gates, change environments or start TEST evaluation to bypass failure.',flush=True)
        raise
