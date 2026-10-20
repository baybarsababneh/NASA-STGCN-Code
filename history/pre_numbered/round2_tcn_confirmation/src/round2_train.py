"""Validation-only training. Never loads TEST, labels or calibration observations.

All candidates have identical fit targets starting at index 64. Longer contexts
use zero (fit-standardized mean) padding only before the start of the array.
The last 20% remains sealed: no calibration errors or thresholds are calculated.
"""
from pathlib import Path
import json,time,copy,os,hashlib,sys,fcntl
import numpy as np
import pandas as pd
import torch
from round2_models import make,adjacency

ROOT=Path(__file__).resolve().parents[1]
BASE=Path(os.environ.get('SMAP_MSL_ROOT',str(ROOT.parent/'smap_msl_research')))
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)))

def prepare(ch,L):
    path=BASE/'data/data/data/train'/f'{ch}.npy'
    # Memory mapping permits shape access followed by fit/validation slices only.
    raw=np.load(path,mmap_mode='r');n=len(raw);f=int(.6*n);v=int(.8*n)
    used=np.array(raw[:v],dtype=np.float32);active=np.r_[0,np.flatnonzero(used[:f,1:].std(0)>0)+1]
    x=used[:,active].copy();mu=float(x[:f,0].mean());sd=float(x[:f,0].std());constant=sd<1e-7
    if constant:sd=1.
    x[:,0]=(x[:,0]-mu)/sd
    padded=np.pad(x,((L,0),(0,0)))
    windows=np.lib.stride_tricks.sliding_window_view(padded,L,axis=0)[64:v].copy()
    t=np.arange(64,v);y=x[t,0].copy()
    meta=dict(active=active.tolist(),mean=mu,scale=sd,fit_end=f,validation_end=v,total_train_rows=n,constant_target=constant,first_fit_target=64,padded_fit_targets=max(0,min(L,f)-64),calibration_accessed=False)
    return x,torch.tensor(windows),torch.tensor(y),t,meta

@torch.no_grad()
def predict(model,x):
    model.eval();return torch.cat([model(b) for b in x.split(256)]).numpy()

def _train_job(job):
    ch,mission,cfg,seed=job
    folder=ROOT/'checkpoints'/cfg['name'];folder.mkdir(parents=True,exist_ok=True)
    out=folder/f'{ch}_s{seed}.json'
    if out.exists():
        saved=json.loads(out.read_text());assert saved['configuration']==cfg
        return saved['result']
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    x,X,y,t,meta=prepare(ch,cfg['lookback']);a=adjacency(x[:meta['fit_end']],cfg['graph'])
    model=make(cfg,x.shape[1],a);tr=np.flatnonzero(t<meta['fit_end']);va=np.flatnonzero(t>=meta['fit_end'])
    assert len(tr)>0 and len(va)>0
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.get('lr',.001),weight_decay=.0001)
    best=float('inf');best40=float('inf');state=None;bad=0;history=[];start=time.perf_counter()
    val0=float(np.mean((predict(model,X[va])-y[va].numpy())**2))
    for epoch in range(cfg.get('epochs',80)):
        model.train();loss_sum=0
        for idx in np.array_split(rng.permutation(tr),max(1,int(np.ceil(len(tr)/256)))):
            opt.zero_grad();loss=(model(X[idx])-y[idx]).square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();loss_sum+=float(loss.detach())*len(idx)
        val=float(np.mean((predict(model,X[va])-y[va].numpy())**2));history.append(dict(epoch=epoch+1,fit_mse=loss_sum/len(tr),validation_mse=val))
        if val<best-1e-7:
            best=val;best_epoch=epoch+1;state=copy.deepcopy(model.state_dict());bad=0
        else:bad+=1
        if epoch<40:best40=best;epoch40=best_epoch
        if bad>=6:break
    model.load_state_dict(state);vp=predict(model,X[va]);yv=y[va].numpy()
    persistence=X[va,0,-1].numpy();pmse=float(np.mean((persistence-yv)**2))
    result=dict(channel=ch,spacecraft=mission,configuration=cfg['name'],kind=cfg['kind'],seed=seed,lookback=cfg['lookback'],hidden=cfg['hidden'],graph=cfg['graph'],kernel_size=cfg.get('kernel_size',0),receptive_field=1+(cfg.get('kernel_size',3)-1)*sum(cfg.get('dilations',[1,2])) if cfg['kind']=='tcn_gcn' else cfg['lookback'],val_mse=best,val_raw_mse=best*meta['scale']**2,val_mae=float(np.abs(vp-yv).mean()),persistence_val_mse=pmse,initial_val_mse=val0,best40_mse=best40,best40_epoch=epoch40,best_epoch=best_epoch,epochs=len(history),ceiling=cfg.get('epochs',80),ceiling_hit=best_epoch==cfg.get('epochs',80),early_stopped=len(history)<cfg.get('epochs',80),seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),nodes=x.shape[1],edges=int((a>0).sum()),fit_targets=len(tr),validation_targets=len(va),padded_fit_targets=meta['padded_fit_targets'],constant_target=meta['constant_target'])
    torch.save(dict(schema='round2_tcn_v1',configuration=cfg,metadata=meta,seed=seed,adjacency=a,state=state),out.with_suffix('.pt'))
    np.savez_compressed(out.with_suffix('.npz'),times=t[va],prediction=vp,truth=yv,persistence=persistence)
    dump(out,dict(configuration=cfg,metadata=meta,history=history,result=result))
    return result


def train_job(job):
    """Coordinate overlapping queues without duplicate writes or training."""
    ch,mission,cfg,seed=job
    folder=ROOT/'checkpoints'/cfg['name'];folder.mkdir(parents=True,exist_ok=True)
    with (folder/f'{ch}_s{seed}.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        return _train_job(job)
