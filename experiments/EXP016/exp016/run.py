import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, contextlib, csv, fcntl, io, json, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
from .io import ROOT,sha,digest,atomic,read_json,save_json,save_npz,freeze,commit,committed,package_identity,Mirror,export_run
from .data import stage_data,audit_data
from .train import configure,environment,train_one,predict,BudgetReached
from .benchmark import benchmark
from .verify_inputs import verify_released
from .gpu_gate import run_gate
from .detector import detect
from .evaluate import evaluate

class Tee:
    def __init__(self,*s):self.streams=s
    def write(self,v):
        for s in self.streams:s.write(v);s.flush()
    def flush(self):
        for s in self.streams:s.flush()

def gate_time(deadline,mirror):
    if time.monotonic()>=deadline:
        mirror.sync(force=True);raise BudgetReached('Session budget reached at a committed stage boundary.')

def run_core(root,out,c,rows,identity,device,mirror,deadline,label_path=None,fixture=False):
    for seed in c['seeds']:
        for r in rows:
            ch=r['channel'];folder=out/'fits'/f'{ch}_{seed}';gate_time(deadline,mirror)
            tr=np.load(root/'train'/(ch+'.npy'),allow_pickle=False)
            fitid=digest(dict(run=identity,channel=ch,seed=seed))
            train_one(tr,c,seed,folder,fitid,device,deadline,mirror)
    fits={str(p.relative_to(out)):sha(p) for p in sorted((out/'fits').glob('*/COMMIT.json'))}
    if len(fits)!=len(rows)*len(c['seeds']):raise RuntimeError('Incomplete fit roster')
    freeze(out/'TRAINING_SEAL.json',dict(identity=identity,files=fits))
    mirror.sync(force=True)
    # Forecast all channels with all seeds BEFORE accessing any new annotation intervals.
    for r in rows:
        ch=r['channel'];te=np.load(root/'test'/(ch+'.npy'),allow_pickle=False)
        for role,seed in [('released_lstm',0)]+[('stgcn',s) for s in c['seeds']]:
            gate_time(deadline,mirror);folder=out/'predictions'/f'{role}_{seed}_{ch}'
            jid=digest(dict(run=identity,role=role,seed=seed,channel=ch))
            if not committed(folder,jid):
                if role=='released_lstm':
                    first=np.load(root/'2018-05-19_15.00.10/y_hat'/(ch+'.npy'));all_horizons=first[:,None]
                else:
                    all_horizons=predict(te,c,out/'fits'/f'{ch}_{seed}'/'final.pt',device);first=all_horizons[:,0]
                save_npz(folder/'prediction.npz',prediction=first,all_horizons=all_horizons,target_index=np.arange(250,len(te)-10))
                commit(folder,['prediction.npz'],jid)
            z=np.load(folder/'prediction.npz');first=z['prediction']
            for mode in ('uniform','native_C2'):
                gate_time(deadline,mirror);sf=out/'scores'/f'{role}_{seed}_{ch}_{mode}';sid=digest(dict(job=jid,mode=mode))
                if committed(sf,sid):continue
                try:d=detect(te,first,ch,mode)
                except Exception as ex:
                    if mode=='uniform':raise
                    save_json(sf/'receipt.json',dict(failure=type(ex).__name__+': '+str(ex)))
                    commit(sf,['receipt.json'],sid);continue
                save_npz(sf/'stream.npz',flags=d['flags'],issue_index=d['issue_index'],raw_error=d['raw_error'],
                    smoothed=d['smoothed'],target_index=np.arange(250,len(te)-10))
                save_json(sf/'receipt.json',{k:d[k] for k in ('intervals','warnings','degeneracies','windows','anomaly_scores')})
                commit(sf,['stream.npz','receipt.json'],sid)
            mirror.sync()
        print('Forecast/detector outputs committed:',ch,flush=True)
        mirror.sync(force=True)
    scores={str(p.relative_to(out)):sha(p) for base in ('predictions','scores') for p in sorted((out/base).rglob('*')) if p.is_file() and not p.name.startswith('.')}
    freeze(out/'SCORE_SEAL.json',dict(identity=identity,files=scores,labels_used_for_fitting=False))
    gate_time(deadline,mirror)
    evaluation=evaluate(out,c,rows,label_path,fixture)
    save_json(out/'FINAL_AUDIT.json',dict(status='FIXTURE_COMPLETE' if fixture else (('COMPLETE' if evaluation['primary_degenerate_branches']==0 else 'COMPLETE_WITH_RECORDED_ABSTENTIONS') if evaluation['detector_failures']==0 else 'COMPLETE_WITH_NATIVE_DETECTOR_FAILURES'),
        identity=identity,fit_jobs=len(fits),**evaluation,
        scientific_GPU_run=not fixture,fixture=fixture,
        limitations=['retrospective TEST exposure','inherited TEST-extrema preprocessing','chronological purged split differs from released LSTM',
                     'one released LSTM fit per channel; original training seed/history unavailable','native detector uses retrospective lookahead']))
    save_json(out/'STATUS.json',dict(status='COMPLETE',fixture=fixture));mirror.sync(force=True)
    return evaluation

def fixture_inputs(root):
    rng=np.random.default_rng(42);rows=[]
    for ch,nodes in [('A-1',25),('C-1',55)]:
        for split,n in [('train',540),('test',950)]:
            t=np.arange(n);a=np.zeros((n,nodes),np.float64);a[:,0]=.2*np.sin(t/11)+.01*rng.normal(size=n)
            a[:,1:]=(rng.random((n,nodes-1))>.8).astype(float)
            if split=='test':a[510:570,0]+=1.3
            (root/split).mkdir(parents=True,exist_ok=True);np.save(root/split/(ch+'.npy'),a)
        p=root/'2018-05-19_15.00.10/y_hat';p.mkdir(parents=True,exist_ok=True)
        test=np.load(root/'test'/(ch+'.npy'));np.save(p/(ch+'.npy'),test[249:-11,0])
    labels=root/'fixture_labels.csv'
    labels.write_text('chan_id,spacecraft,anomaly_sequences,class,num_values\nA-1,SMAP,"[[510, 569]]",[point],950\nC-1,MSL,"[[510, 569]]",[point],950\n')
    return labels

def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['preflight','run','fixture','export'])
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--data',type=Path,default=Path('/content/EXP016_data'))
    ap.add_argument('--mirror',type=Path);ap.add_argument('--hours',type=float,default=5.5);ap.add_argument('--cpu',action='store_true')
    a=ap.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if a.hours<=0:raise ValueError('--hours must be positive')
    mirror=Mirror(a.out,a.mirror);mirror.restore()
    lock=open(a.out/'.RUN_LOCK','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('Another runner is already using this output directory')
    if a.command=='export':print(export_run(a.out));return
    device='cpu' if a.command=='fixture' or a.cpu else 'cuda'
    if device=='cpu' and a.command=='run':raise RuntimeError('CPU is engineering-only. Full scientific run requires CUDA.')
    configure(device);package_hash=package_identity();c=read_json(ROOT/'config.json')
    if a.command=='fixture':
        c.update(hidden=4,epochs=2,seeds=[17],training_channels=2,labelled_channels=2,bootstrap_repetitions=100,checkpoint_steps=2)
        root=a.data.resolve();label_path=fixture_inputs(root)
    else:root=stage_data(a.data);label_path=None
    identity=digest(dict(package=package_hash,config=c,environment=environment(device),fixture=a.command=='fixture'))
    freeze(a.out/'RUN_FREEZE.json',dict(identity=identity,package=package_hash,config=c,environment=environment(device)))
    # Preserve executable source/config/test evidence with every exported scientific run.
    if not (a.out/'source').exists():
        shutil.copytree(ROOT/'exp016',a.out/'source'/'exp016',ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('config.json','PACKAGE_MANIFEST.json','PROTOCOL.md','requirements.txt'):shutil.copy2(ROOT/name,a.out/'source'/name)
        shutil.copytree(ROOT/'vendor',a.out/'source'/'vendor')
    import subprocess
    atomic(a.out/'requirements_resolved.txt',subprocess.check_output([sys.executable,'-m','pip','freeze']))
    log=open(a.out/'execution.log','a',buffering=1)
    with contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
        deadline=time.monotonic()+a.hours*3600
        try:
            rows=audit_data(root,c,a.out)
            if a.command=='preflight':
                run_gate(c,a.out,device);verify_released(root,a.out);benchmark(root,rows,c,a.out,device)
                save_json(a.out/'PREFLIGHT.json',dict(status='PASS',identity=identity,device=device,full_GPU_run_tested=False))
            elif a.command=='fixture':
                run_core(root,a.out,c,rows,identity,device,mirror,deadline,label_path,True)
            else:
                if not (a.out/'PREFLIGHT.json').exists():raise RuntimeError('Run the preflight cell first on this GPU.')
                if read_json(a.out/'PREFLIGHT.json')['identity']!=identity:raise RuntimeError('Preflight identity mismatch')
                run_core(root,a.out,c,rows,identity,device,mirror,deadline)
        except BudgetReached as e:
            save_json(a.out/'STATUS.json',dict(status='PAUSED_RESUMABLE',reason=str(e)));print(str(e),flush=True)
        except BaseException as e:
            save_json(a.out/'STATUS.json',dict(status='INTERRUPTED_RESUMABLE' if isinstance(e,KeyboardInterrupt) else 'FAILED',reason=repr(e)))
            traceback.print_exc();raise
        finally:
            mirror.sync(force=True)
            result=export_run(a.out);print('Result/recovery archive:',result,flush=True)
            if a.mirror:
                atomic(a.mirror.parent/result.name,result.read_bytes())
if __name__=='__main__':main()
