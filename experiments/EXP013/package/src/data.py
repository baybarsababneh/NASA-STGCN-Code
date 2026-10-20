"""81 independent entities, never cross-entity windows or fictitious sensor alignment."""
import io, zipfile, json, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from core import *

def stage(work):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    pins=read(ROOT/'INPUT_PINS.json')
    for name,h in pins.items(): checked(ROOT/'inputs'/name,h)
    # Reuse the completed historical reports; never replay or retrain EXP012.
    for name,mname in [('EXP012_RETROSPECTIVE_EVALUATION_REPORTS.zip','LABEL_EVALUATION_COMPLETE.json')]:
        with zipfile.ZipFile(ROOT/'inputs'/name) as z:
            m=json.loads(z.read(mname));files=m.get('files')
            if files is None: raise RuntimeError('Missing historical hash manifest')
            if set(z.namelist())!={*files,mname}: raise RuntimeError('Historical archive roster differs')
            for n,h in files.items():
                if hashlib.sha256(z.read(n)).hexdigest()!=h: raise RuntimeError('Historical payload changed: '+n)
    rows=read(ROOT/'CHANNELS.json')
    if len(rows)!=81 or len({r['channel'] for r in rows})!=81 or sum(r['mission']=='MSL' for r in rows)!=27: raise RuntimeError('NASA roster differs')
    raw={};meta={}
    with zipfile.ZipFile(ROOT/'inputs'/'NASA_arrays_only.zip') as z:
        for r in rows:
            ch=r['channel'];b=z.read('data/data/train/'+ch+'.npy')
            if hashlib.sha256(b).hexdigest()!=r['original_TRAIN_sha256']: raise RuntimeError('TRAIN hash differs: '+ch)
            a=np.load(io.BytesIO(b),allow_pickle=False)
            if a.shape!=(r['rows'],r['columns']) or not np.isfinite(a).all(): raise RuntimeError('Invalid TRAIN array: '+ch)
            f=int(.6*len(a));s=int(.8*len(a));c=int(.9*len(a))
            mean=float(np.mean(a[:f,0],dtype=np.float64));std=float(np.std(a[:f,0],dtype=np.float64))
            scale=max(std,1e-3) # declared numerical floor; no float32 cancellation on nearly constant FIT
            v=a.astype(np.float32,copy=True);v[:,0]=(a[:,0].astype(np.float64)-mean)/scale
            raw[ch]=v;meta[ch]={**r,'fit_end':f,'stop_end':s,'cal_end':c,'mean':mean,'scale':scale,'FIT_std_float64':std,'scale_floor_used':std<1e-3,'FIT_varying_command_slots':(np.flatnonzero(np.ptp(a[:f,1:],axis=0)>0)+1).tolist()}
    freeze(work/'DATA_FREEZE.json',{'version':VERSION,'inputs':pins,'channels':meta,'split':'FIT0:60 STOP60:80 CAL_SELECT80:90 CONF90:100 percent; floor then context from earlier observed points only','inherited_preprocessing':'Official NASA arrays were scaled using TEST min/max before release. Additional EXP013 transforms use FIT only; this does not certify raw-data leakage freedom.','TEST_arrays_opened':False,'NASA_labels_opened':False})
    return raw,meta

def training_data(work):
    f=read(Path(work)/'DATA_FREEZE.json');raw={}
    with zipfile.ZipFile(ROOT/'inputs'/'NASA_arrays_only.zip') as z:
        for ch,r in f['channels'].items():
            original=np.load(io.BytesIO(z.read('data/data/train/'+ch+'.npy')),allow_pickle=False)
            a=original.astype(np.float32);a[:,0]=(original[:,0].astype(np.float64)-r['mean'])/r['scale'];raw[ch]=a
    return raw,f['channels']

def external_scaler(raw,meta,channels):
    fit=np.concatenate([raw[ch][:meta[ch]['fit_end']] for ch in channels])
    mu=fit.astype(np.float64).mean(0);sd=fit.astype(np.float64).std(0);sd[sd<1e-12]=1.
    return {'mean':mu.tolist(),'scale':sd.tolist(),'fit_rows':len(fit),'target_before_scaler':'per-entity FIT telemetry standardization, all command encodings preserved'}
def transform(raw,scaler):
    if scaler is None: return raw
    return {ch:np.asarray((a-np.asarray(scaler['mean']))/np.asarray(scaler['scale']),np.float32) for ch,a in raw.items()}

class Windows(Dataset):
    def __init__(self,raw,meta,channels,kind,split='fit',cap=None):
        self.raw=raw;self.channels=channels;self.kind=kind;self.rows=[]
        self.lookback=64 if kind in ['graph','removed','fusion'] else 5 if kind=='gdn' else 191
        for entity,ch in enumerate(channels):
            m=meta[ch];lo=self.lookback if split=='fit' else max(m['fit_end'],self.lookback)
            hi=m['fit_end'] if split=='fit' else m['stop_end']
            ids=np.arange(lo,hi,dtype=int)
            if cap and len(ids)>cap: ids=ids[np.linspace(0,len(ids)-1,cap).astype(int)]
            self.rows.extend((ch,int(t),entity) for t in ids)
        if not self.rows: raise ValueError('No causal windows for '+split)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        ch,t,e=self.rows[i];a=self.raw[ch]
        if self.kind in ['graph','removed','fusion']:
            x=a[t-64:t+1].copy();x[-1,1:]=0 # no current/future command information
            return torch.from_numpy(x),torch.tensor(e),torch.tensor(a[t,0])
        if self.kind=='gdn': return torch.from_numpy(a[t-5:t].T.copy()),torch.from_numpy(a[t].copy())
        return torch.from_numpy(a[t-191:t+1].copy()),torch.from_numpy(a[t-191:t+1].copy())

def mutate(a,onset,end,family,amplitude,seed):
    b=a.copy();n=end-onset;rng=np.random.default_rng(seed);v=b[onset:end,0].copy();sign=1 if seed%2 else -1
    if family=='spike': b[onset,0]+=sign*amplitude;end=onset+1
    elif family=='offset': b[onset:end,0]+=sign*amplitude
    elif family=='ramp': b[onset:end,0]+=sign*amplitude*np.linspace(.1,1,n,dtype=np.float32)
    elif family=='stuck': b[onset:end,0]=b[max(0,onset-1),0]
    elif family=='variance': b[onset:end,0]+=amplitude*rng.standard_normal(n).astype(np.float32)
    elif family=='oscillation': b[onset:end,0]+=amplitude*np.sin(np.arange(n)*2*np.pi/7).astype(np.float32)
    else: raise ValueError(family)
    changed=np.abs(b[onset:end,0]-a[onset:end,0])>1e-7
    return b,end,bool(changed.any())

def cases(a,m,ch,phase):
    # Quarantined 90:100 time points never used for fit, checkpoint or threshold selection.
    lo=m['stop_end'] if phase=='select' else m['cal_end'];hi=m['cal_end'] if phase=='select' else m['rows']
    families=['spike','offset','ramp','stuck']+(['variance','oscillation'] if phase=='confirm' else [])
    amps=[1.5,3.] if phase=='select' else [2.,4.]
    lengths=[16,64] if phase=='select' else [48,96]
    for fi,f in enumerate(families):
        for ai,amp in enumerate(amps):
            for li,L in enumerate([1] if f=='spike' else lengths):
                seed=int(hashlib.sha256(f'{VERSION}/{phase}/{ch}/{f}/{ai}/{li}'.encode()).hexdigest()[:8],16)
                length=min(L,max(1,(hi-lo-16)//2));onset=lo+max(0,(hi-lo-length-16)//2)
                end=onset+length;start=lo;stop=min(hi,end+16)
                b,end,valid=mutate(a,onset,end,f,amp,seed)
                yield {'id':f'{ch}_{phase}_{f}_{ai}_{li}','family':f,'amplitude':amp,'requested_length':L,'effective_length':end-onset,'onset':onset,'end':end,'start':start,'stop':stop,'valid':valid,'seed':seed},b

def inject_batch(x):
    y=x.clone();B,T,_=y.shape;dev=x.device
    # Vectorized and channel-balanced; all draws belong to checkpointed Torch RNG.
    k=torch.randint(0,5,(B,),device=dev);amp=torch.empty(B,device=dev).uniform_(.5,4.)
    sign=2*torch.randint(0,2,(B,),device=dev)-1
    length=torch.where(k==0,1,torch.where(torch.randint(0,2,(B,),device=dev)==0,8,32))
    index=torch.arange(T,device=dev)[None];mask=index>=T-length[:,None]
    fraction=(index-(T-length[:,None])+1).float()/length[:,None]
    delta=(amp*sign)[:,None]*torch.where(k[:,None]==2,.1+.9*fraction,torch.ones_like(fraction))
    stuck=x[torch.arange(B,device=dev),T-length-1,0][:,None].expand(-1,T)
    donor=x.roll(1,0)[:,:,0]
    changed=torch.where(k[:,None]==4,donor,torch.where(k[:,None]==3,stuck,x[:,:,0]+delta))
    y[:,:,0]=torch.where(mask,changed,x[:,:,0])
    valid=(y[:,:,0]-x[:,:,0]).abs().amax(1)>1e-7
    return y,valid.float()
