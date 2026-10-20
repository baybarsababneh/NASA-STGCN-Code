"""Frozen EXP013 command-reference forward, unchanged t64/256 batch boundaries."""
import time,gc,resource
from pathlib import Path
import numpy as np
import torch
from integrity import ROOT,read,sha,checked,rng,restore_rng,rng_equal,state_hash,event,arrays
from checkpoint import load_torch
from vendor.exp013.architectures import CommandReferenceGCN

def model_job(mission,role,seed):return f'{mission}__{role}__{seed}'

def load_model(source,mission,role,seed,root,device):
    d=Path(source)/'models'/model_job(mission,role,seed);st=read(d/'COMPLETE.json');ck=load_torch(d/'selected.pt')
    checked(d/'selected.pt',st['checkpoint_sha256']);identity=read(d/'IDENTITY.json')
    if st['status']!='COMPLETE' or ck['identity']!=identity:raise RuntimeError('Selected checkpoint/identity differs')
    channels=sorted(ch for ch,r in read(Path(root)/'inputs/TEST_ROSTER.json').items() if r['mission']==mission)
    nf=55 if mission=='MSL' else 25
    cfg=ck['configuration']
    if ck['channels']!=channels or ck['n_features']!=nf or ck['scaler'] is not None:raise RuntimeError('Frozen command-reference roster/scaler differs')
    if cfg['lookback']!=64 or cfg['batch_size']!=256 or identity['role']!=role or identity['mission']!=mission or identity['seed']!=seed or identity['configuration']!=cfg or identity['scaler'] is not None:raise RuntimeError('Frozen model role/seed/configuration differs')
    th=read(Path(source)/'calibration'/model_job(mission,role,seed)/'thresholds.json')
    if th['model_state_sha256']!=st['state_sha256']:raise RuntimeError('Calibration state differs')
    saved=rng()
    try:
        model=CommandReferenceGCN(nf,len(channels),role,h=32)
        model.load_state_dict(ck['model'],strict=True);model.to(device).eval()
    finally:restore_rng(saved)
    if state_hash(model)!=st['state_sha256'] or any(p.dtype!=torch.float32 for p in model.parameters()):raise RuntimeError('Frozen FP32 state differs')
    return model,ck,st

def transform(original,meta,scaler=None):
    if scaler is not None:raise RuntimeError('EXP013 graph/control checkpoints must have no pooled scaler')
    # Preserve the original intermediate cast and command slots.
    a=original.astype(np.float32,copy=True)
    a[:,0]=(original[:,0].astype(np.float64)-meta['mean'])/meta['scale']
    return a

def inverse_reference(reference,meta,scaler=None):
    if scaler is not None:raise RuntimeError('Unexpected pooled scaler')
    if meta['scale']<=0 or not np.isfinite([meta['scale'],meta['mean']]).all():raise RuntimeError('Invalid frozen entity scale')
    return np.asarray(reference,np.float64)*meta['scale']+meta['mean']

def gather(z,lo,hi):
    if lo<64 or not lo<hi<=len(z):raise ValueError('Invalid reference context')
    x=np.stack([z[t-64:t+1] for t in range(lo,hi)])
    x[:,-1,1:]=0
    return np.ascontiguousarray(x)

def ewma(v,span):
    v=np.asarray(v,np.float64)
    if span==1:return v.copy()
    alpha=2/(span+1);out=np.empty_like(v);out[0]=v[0]
    for i in range(1,len(v)):out[i]=alpha*v[i]+(1-alpha)*out[i-1]
    return out

def memory_record(device):
    host={'process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    if device!='cuda':return {'device':'cpu','GPU_validation':False,**host}
    torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
    return {'device':'cuda','GPU_validation':True,'free_bytes':free,'total_bytes':total,
            'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),**host}

def host_budget(n,nf):return int(n*nf*24+max(0,n-64)*192+256*65*nf*8)

def gpu_budget(model,nf):
    # Bound windows/GEMM columns and both live encoder tensors; measured gate follows.
    state=sum(v.numel()*v.element_size() for v in model.state_dict().values())
    nodes=256*max(1,nf-1)
    return int(2**30+4*state+8*nodes*65*32*4+4*nodes*64*96*4)

def verify_reference(source,mission,role,seed,ch,logits,root):
    p=Path(source)/'scores/units'/(model_job(mission,role,seed)+'__'+ch)
    ref=arrays(p/'scores.npz');row=read(Path(root)/'inputs/TEST_ROSTER.json')[ch]
    ix=np.arange(250,row['rows'],dtype=np.int64)
    th=read(Path(source)/'calibration'/model_job(mission,role,seed)/'thresholds.json')['channels'][ch]
    n=np.asarray(logits,np.float64);smooth=ewma(n,th['span'])
    n=n[250-64:];smooth=smooth[250-64:];flags=smooth>th['threshold']
    for key,value in [('target_index',ix),('native_score',n),('detector_score',smooth),('flags',flags),('issue_index',np.where(flags,ix,-1))]:
        if ref[key].shape!=value.shape or not np.array_equal(ref[key],value):
            delta=None if ref[key].shape!=value.shape else float(np.max(np.abs(ref[key].astype(float)-value.astype(float))))
            raise RuntimeError(f'Frozen EXP013 {key} replay failed for {mission}/{role}/{seed}/{ch}; max abs difference {delta}. No tolerance or substitution is authorised.')
    return {'frozen_native_score_replay':'BITWISE_PASS','native_flags_issue_replay':'BITWISE_PASS',
            'max_abs_native_difference':0.,'reference_sha256':sha(p/'scores.npz')}

@torch.no_grad()
def predict(model,z,entity,device,cfg,work,ch,tick=None):
    n=len(z);start=64
    if n<=250 or host_budget(n,z.shape[1])>cfg['maximum_single_channel_host_bytes']:raise RuntimeError('Empty support or channel host budget exceeded')
    references=np.empty(n-start,np.float32);logits=np.empty(n-start,np.float32)
    before=state_hash(model);saved=rng();beg=last=time.monotonic()
    for lo in range(start,n,256):
        hi=min(n,lo+256);x=torch.from_numpy(gather(z,lo,hi)).to(device)
        entities=torch.full((len(x),),entity,dtype=torch.long,device=device)
        ref,logit=model(x,entities);ref=ref.cpu().numpy();logit=logit.cpu().numpy()
        if ref.shape!=(hi-lo,) or logit.shape!=ref.shape or ref.dtype!=np.float32 or logit.dtype!=np.float32 or not np.isfinite(ref).all() or not np.isfinite(logit).all():raise RuntimeError('Invalid frozen native reference/logit vectors')
        references[lo-start:hi-start]=ref;logits[lo-start:hi-start]=logit
        if device=='cuda' and torch.cuda.memory_reserved()>cfg['maximum_gpu_reserved_fraction']*torch.cuda.get_device_properties(0).total_memory:raise RuntimeError('GPU inference budget exceeded')
        del x,entities,ref,logit
        if time.monotonic()-last>10:
            elapsed=time.monotonic()-beg;remaining=elapsed*(n-hi)/(hi-start)
            print(f'INFERENCE {ch} {hi-start}/{n-start} · current-channel ETA {remaining/60:.1f} min from completed batches',flush=True);last=time.monotonic()
        if tick:tick()
    if state_hash(model)!=before or not rng_equal(saved,rng()):raise RuntimeError('Inference mutated model state or RNG')
    return references,logits

def full_shape_gate(model,ck,st,source,mission,role,seed,root,device,work):
    cfg=read(Path(root)/'config.json');saved=rng();before=state_hash(model)
    ch=ck['channels'][0];row=read(Path(root)/'inputs/TEST_ROSTER.json')[ch]
    if row['rows']<320:raise RuntimeError('No full 256-target native batch in the gate channel')
    if host_budget(row['rows'],row['columns'])>cfg['maximum_single_channel_host_bytes']:raise RuntimeError('Gate host budget exceeded')
    if device=='cuda':
        free,total=torch.cuda.mem_get_info()
        if free<max(cfg['minimum_free_gpu_bytes_before_model'],gpu_budget(model,ck['n_features'])):raise RuntimeError('Insufficient conservative full-shape GPU headroom')
        torch.cuda.reset_peak_memory_stats()
    from inputs import load_test
    original=load_test(root,ch);meta=read(Path(root)/'inputs/freeze/DATA_FREEZE.json')['channels'][ch]
    z=transform(original,meta,ck['scaler']);x=torch.from_numpy(gather(z,64,320)).to(device)
    entities=torch.zeros(256,dtype=torch.long,device=device)
    if device=='cuda':torch.cuda.synchronize()
    beg=time.monotonic()
    with torch.no_grad():first=tuple(v.detach().cpu().clone() for v in model(x,entities))
    if device=='cuda':torch.cuda.synchronize()
    elapsed=time.monotonic()-beg
    if any(v.shape!=(256,) or v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in first):raise RuntimeError('Invalid full-shape FP32 outputs')
    model.load_state_dict(ck['model'],strict=True);model.eval();restore_rng(saved)
    with torch.no_grad():second=tuple(v.detach().cpu().clone() for v in model(x,entities))
    if any(not torch.equal(a,b) for a,b in zip(first,second)) or state_hash(model)!=before or before!=st['state_sha256'] or not rng_equal(saved,rng()):raise RuntimeError('Exact reference/logit/state/RNG restoration failed')
    native=arrays(Path(source)/'scores/units'/(model_job(mission,role,seed)+'__'+ch)/'scores.npz')['native_score']
    if not np.array_equal(first[1].numpy()[250-64:].astype(np.float64),native[:70]):raise RuntimeError('Real full-shape native logit replay failed; no tolerance permitted')
    mem=memory_record(device)
    if device=='cuda' and mem['peak_reserved_bytes']>cfg['maximum_gpu_reserved_fraction']*mem['total_bytes']:raise RuntimeError('Actual full-shape GPU memory budget exceeded')
    receipt={'job_id':model_job(mission,role,seed),'mission':mission,'role':role,'seed':seed,
             'shape':[256,65,ck['n_features']],'seconds_first_forward':elapsed,'cold_batch_only_not_runtime_forecast':True,
             'exact_reference_logit_restoration':'PASS','native_supported_full_batch_replay':'BITWISE_PASS',
             'precision':'FP32','measured_memory':mem,'conservative_GPU_budget_bytes':gpu_budget(model,ck['n_features']),
             'checkpoint_sha256':st['checkpoint_sha256'],'architecture_modified':False}
    event(work,'FULL_SHAPE_NATIVE_RESTORE_PASS',**receipt)
    del original,z,x,entities,first,second;gc.collect()
    return receipt
