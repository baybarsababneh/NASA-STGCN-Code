"""Unchanged GDN equations; streamed FP32 forecasting and exact native replay."""
import sys, types, importlib, time, gc
from pathlib import Path
import numpy as np
import torch
from integrity import ROOT, read, sha, checked, rng, restore_rng, rng_equal, state_hash, event
from checkpoint import load_torch

def factory(nf,device):
    p=ROOT/'vendor/gdn'
    if str(p) not in sys.path: sys.path.insert(0,str(p))
    from util.env import set_device
    set_device(device)
    for name,loc in [('official_gdn',p),('official_gdn.models',p/'models')]:
        if name not in sys.modules:
            mod=types.ModuleType(name);mod.__path__=[str(loc)];sys.modules[name]=mod
        elif list(sys.modules[name].__path__)!=[str(loc)]: raise RuntimeError('Conflicting native GDN module')
    GDN=importlib.import_module('official_gdn.models.GDN').GDN
    edge=torch.tensor([(j,i) for i in range(nf) for j in range(nf) if i!=j],dtype=torch.long).T
    return GDN([edge],nf,dim=64,out_layer_inter_dim=128,input_dim=5,out_layer_num=1,topk=5)

def load_model(source,mission,root,device):
    directory=Path(source)/'models'/f'{mission}__gdn__17'
    st=read(directory/'COMPLETE.json');ck=load_torch(directory/'selected.pt')
    checked(directory/'selected.pt',st['checkpoint_sha256'])
    if st['status']!='COMPLETE' or ck['identity']!=read(directory/'IDENTITY.json'): raise RuntimeError('Checkpoint is not the frozen selected model')
    channels=sorted(ch for ch,r in read(Path(root)/'inputs/TEST_ROSTER.json').items() if r['mission']==mission)
    nf=55 if mission=='MSL' else 25
    if ck['channels']!=channels or ck['n_features']!=nf or ck['configuration']['lookback']!=5 or ck['configuration']['batch_size']!=32: raise RuntimeError('Frozen GDN configuration/roster differs')
    if st['state_sha256']!=read(Path(source)/'calibration'/f'{mission}__gdn__17/thresholds.json')['model_state_sha256']: raise RuntimeError('Model/threshold state binding differs')
    saved=rng()
    try: model=factory(nf,device);model.load_state_dict(ck['model'],strict=True);model.to(device).eval()
    finally: restore_rng(saved)
    if state_hash(model)!=st['state_sha256'] or any(p.dtype!=torch.float32 for p in model.parameters()): raise RuntimeError('FP32 selected state differs')
    return model,ck,st

def transform(original,meta,scaler):
    # Exact EXP013 two-stage preprocessing, including the intervening FP32 cast.
    a=original.astype(np.float32,copy=True)
    a[:,0]=(original[:,0].astype(np.float64)-meta['mean'])/meta['scale']
    return np.asarray((a-np.asarray(scaler['mean']))/np.asarray(scaler['scale']),np.float32)

def inverse_target(pred,meta,scaler):
    return (np.asarray(pred,np.float64)*scaler['scale'][0]+scaler['mean'][0])*meta['scale']+meta['mean']

def gather(z,lo,hi):
    if lo<5 or not lo<=hi<=len(z): raise ValueError('Invalid forecast support')
    return np.ascontiguousarray(np.stack([z[t-5:t].T for t in range(lo,hi)]))

def native(v,stats):
    # Unchanged frozen EXP013 gdn_native, including its first three zeros.
    z=(v-np.asarray(stats['median']))/(np.abs(stats['iqr'])+.01)
    y=np.zeros_like(z)
    for i in range(3,len(z)): y[i]=z[i-3:i+1].mean(0)
    return y.max(1)

def ewma(v,span):
    v=np.asarray(v,np.float64)
    if span==1:return v.copy()
    alpha=2/(span+1);out=np.empty_like(v);out[0]=v[0]
    for i in range(1,len(v)):out[i]=alpha*v[i]+(1-alpha)*out[i-1]
    return out

def memory_record(device):
    import resource
    host={'process_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
    if device!='cuda': return {'device':'cpu','GPU_validation':False,**host}
    torch.cuda.synchronize();free,total=torch.cuda.mem_get_info()
    return {'device':'cuda','GPU_validation':True,'free_bytes':free,'total_bytes':total,'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),**host}

def host_budget(n,nf):
    # Raw FP64 data, FP32 copies, native FP64 reduction temporaries, score
    # vectors, reference arrays and window staging, with conservative slack.
    return int(n*nf*48+max(0,n-5)*128)

def gpu_budget(model,nf):
    # An engineering upper estimate for this fixed small native inference
    # shape, not a measured peak or a proof of device feasibility. Include
    # dense cached edges, generous node/edge temporary multiplicities and
    # 1 GiB of allocator/library slack; the actual full-shape gate follows.
    state=sum(t.numel()*t.element_size() for t in model.state_dict().values())
    b,d,k=32,64,5
    nodes=b*nf*d*4;edges=b*nf*(k+1)*(2*d)*4
    dense_cache=b*nf*(nf-1)*2*8
    return int(2**30+4*state+16*nodes+16*edges+4*dense_cache)

def full_shape_gate(model,ck,st,source,mission,root,device,work):
    cfg=read(Path(root)/'config.json');saved=rng();before=state_hash(model)
    if device=='cuda':
        free,total=torch.cuda.mem_get_info()
        if free<cfg['minimum_free_gpu_bytes_before_model']: raise RuntimeError('Insufficient GPU headroom before full-shape inference')
        if free<gpu_budget(model,ck['n_features']): raise RuntimeError('Conservative full-shape GPU estimate exceeds free memory')
        torch.cuda.reset_peak_memory_stats()
    # One actual full native inference batch, with real frozen TEST context.
    ch=ck['channels'][0];row=read(Path(root)/'inputs/TEST_ROSTER.json')[ch]
    estimate=host_budget(row['rows'],row['columns'])
    if estimate>cfg['maximum_single_channel_host_bytes']:raise RuntimeError('Full-shape channel host-memory estimate exceeds budget')
    a=__import__('inputs').load_test(root,ch)
    meta=read(Path(root)/'inputs/freeze/DATA_FREEZE.json')['channels'][ch]
    z=transform(a,meta,ck['scaler']);x=torch.from_numpy(gather(z,5,37)).to(device)
    if device=='cuda':torch.cuda.synchronize()
    start=time.monotonic()
    with torch.no_grad(): first=model(x,None).detach().cpu().clone()
    if device=='cuda':torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    if not torch.isfinite(first).all(): raise FloatingPointError('Nonfinite full-shape native output')
    model.load_state_dict(ck['model'],strict=True);model.eval();restore_rng(saved)
    with torch.no_grad(): second=model(x,None).detach().cpu()
    if not torch.equal(first,second) or state_hash(model)!=before or before!=st['state_sha256'] or not rng_equal(saved,rng()): raise RuntimeError('Exact output/state/RNG restoration failed')
    mem=memory_record(device)
    if device=='cuda' and mem['peak_reserved_bytes']>cfg['maximum_gpu_reserved_fraction']*mem['total_bytes']: raise RuntimeError('Full-shape inference exceeds conservative GPU memory budget')
    report={'mission':mission,'shape':[32,ck['n_features'],5],'seconds_first_forward':elapsed,'exact_restoration':'PASS','precision':'FP32','measured_memory':mem,'conservative_channel_host_bytes':estimate,'estimated_GPU_working_set_bytes':gpu_budget(model,ck['n_features']),'GPU_estimate_is_not_measured_validation':True,'source_checkpoint_sha256':st['checkpoint_sha256'],'native_architecture_modified':False}
    event(work,'FULL_SHAPE_MEMORY_RESTORE_PASS',**report)
    del x,z,a,first,second;gc.collect()
    return report

@torch.no_grad()
def predict(model,z,device,cfg,work,ch,tick=None):
    n=len(z);start=5
    if n<=250: raise RuntimeError('The frozen common support is empty')
    # Preserve EXP013's native inference batch boundaries and smoothing history
    # from t=5. Only the new Hundman detector starts afresh at t=250.
    # Bound host storage before allocating the per-channel target/error arrays.
    estimate=host_budget(n,z.shape[1])
    if estimate>cfg['maximum_single_channel_host_bytes']: raise RuntimeError('Single-channel host budget exceeded; no silent batching/scoring change')
    pred=np.empty(n-start,np.float32);errors=np.empty((n-start,z.shape[1]),np.float32)
    saved=rng();before=state_hash(model);progress=time.monotonic()
    for lo in range(start,n,32):
        hi=min(n,lo+32);x=torch.from_numpy(gather(z,lo,hi)).to(device)
        out=model(x,None).cpu().numpy()
        if out.shape!=(hi-lo,z.shape[1]) or out.dtype!=np.float32 or not np.isfinite(out).all(): raise RuntimeError('Invalid native forecast output')
        pred[lo-start:hi-start]=out[:,0]
        errors[lo-start:hi-start]=np.abs(out-z[lo:hi])
        del x,out
        if device=='cuda' and torch.cuda.memory_reserved()>cfg['maximum_gpu_reserved_fraction']*torch.cuda.get_device_properties(0).total_memory: raise RuntimeError('Inference GPU memory budget exceeded')
        if time.monotonic()-progress>20:
            if device=='cuda':torch.cuda.synchronize()
            print('GDN INFERENCE',ch,f'{hi-start}/{n-start} targets',flush=True);progress=time.monotonic()
        if tick: tick()
    if state_hash(model)!=before or not rng_equal(saved,rng()): raise RuntimeError('Inference changed frozen weights/buffers or RNG')
    return pred,errors

def verify_reference(source,mission,ch,errors,root):
    from integrity import arrays
    p=Path(source)/'scores/units'/f'{mission}__gdn__17__{ch}'
    ref=arrays(p/'scores.npz');record=read(p/'record.json')
    row=read(Path(root)/'inputs/TEST_ROSTER.json')[ch];ix=np.arange(250,row['rows'],dtype=np.int64)
    if not np.array_equal(ref['target_index'],ix): raise RuntimeError('Frozen native reference support differs')
    th=read(Path(source)/'calibration'/f'{mission}__gdn__17/thresholds.json')['channels'][ch]
    all_scores=native(errors,th['native_stats']);all_smooth=ewma(all_scores,th['span'])
    score=all_scores[250-5:];smooth=all_smooth[250-5:];flags=smooth>th['threshold']
    pairs=[('native_score',score),('detector_score',smooth),('flags',flags),('issue_index',np.where(flags,ix,-1))]
    for key,value in pairs:
        if not np.array_equal(ref[key],value):
            error=float(np.max(np.abs(np.asarray(ref[key],float)-np.asarray(value,float))))
            raise RuntimeError(f'Frozen native {key} replay failed for {ch}; maximum absolute difference {error:.12g}. No tolerance or model substitution is authorised.')
    return {'frozen_native_score_replay':'BITWISE_PASS','native_flags_issue_replay':'BITWISE_PASS','max_abs_native_difference':0.,'reference_sha256':sha(p/'scores.npz')}
