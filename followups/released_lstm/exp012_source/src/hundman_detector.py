"""HUND_MAN_ADAPT_V1: independent implementation of dynamic thresholding/pruning.

No labels, file IO, channel-specific exceptions or parameter search across data.
Half-open ranges throughout. Batch availability is explicitly returned.
"""
import numpy as np
import pandas as pd

DEFAULTS={'batch_size':70,'window_size':30,'smoothing_perc':.05,'error_buffer':100,'p':.13,'l_s':250}
VERSION='HUNDMAN_ADAPT_V1'

def runs(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))

def dilate(mask,radius):
    result=np.zeros(len(mask),bool)
    for a,b in runs(mask): result[max(0,a-radius):min(len(mask),b+radius)]=True
    return result

def choose_threshold(signal,buffer,mean=None,sd=None):
    mean=float(np.mean(signal)) if mean is None else mean
    sd=float(np.std(signal)) if sd is None else sd
    selected=mean+12*sd;best=-np.inf;z_best=12.
    if sd==0: return selected,z_best
    for z in np.arange(2.5,12.,.5):
        cut=mean+z*sd;kept=signal[signal<cut];hits=dilate(signal>=cut,buffer-1)
        sequences=sum(b-a>1 for a,b in runs(hits));count=int(hits.sum())
        if not len(kept) or not count or sequences>5 or count>=len(signal)*.5 or mean==0: continue
        objective=((mean-float(kept.mean()))/mean+(sd-float(kept.std()))/sd)/(sequences**2+count)
        if np.isfinite(objective) and objective>=best: best=objective;selected=cut;z_best=float(z)
    return selected,z_best

def prune(signal,mask,normal_peak,p):
    groups=[(a,b) for a,b in runs(mask) if b-a>1]
    if not groups: return np.zeros(len(mask),bool)
    peaks=np.array([signal[a:b].max() for a,b in groups]);ordered=np.r_[np.sort(peaks)[::-1],normal_peak]
    pending=set()
    for i in range(len(peaks)):
        # A nonpositive peak cannot constitute positive residual-tail evidence.
        drop=(ordered[i]-ordered[i+1])/ordered[i] if ordered[i]!=0 else np.inf
        if drop<p: pending.update(np.flatnonzero(peaks==ordered[i]).tolist())
        else: pending.clear()
    out=np.zeros(len(mask),bool)
    for i,(a,b) in enumerate(groups):
        if i not in pending: out[a:b]=True
    return out

def detect(target,prediction,target_index,settings=None):
    cfg=dict(DEFAULTS if settings is None else settings)
    y=np.asarray(target,dtype=np.float64);pred=np.asarray(prediction,dtype=np.float64);ix=np.asarray(target_index,dtype=np.int64)
    if y.ndim!=1 or pred.shape!=y.shape or ix.shape!=y.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Finite equally sized telemetry/prediction/index vectors required')
    if np.any(np.diff(ix)!=1): raise ValueError('A detector stream must have contiguous original indices')
    n=len(y);batch=cfg['batch_size'];flags=np.zeros(n,bool);available=np.full(n,-1,np.int64);windows=[]
    span=int(batch*cfg['window_size']*cfg['smoothing_perc'])
    smooth=pd.Series(np.abs(y-pred)).ewm(span=span,adjust=True).mean().to_numpy()
    startup=min(n,2*cfg['l_s']);smooth[:min(n,cfg['l_s'])]=smooth[:startup].mean()
    if n<batch:
        return {'flags':flags,'issue_index':available,'smoothed':smooth,'windows':[],
                'status':'NO_OUTPUT_SHORT_STREAM','startup_averaging_lookahead':startup}
    width=min(cfg['window_size'],n//batch)*batch
    endpoints=[width+i*batch for i in range((n-width)//batch+1)]
    endpoints[-1]=n  # Last upstream-style batch includes its remainder, never drops it.
    ignore=0 if n<1800 else cfg['l_s'] if n<2500 else 2*cfg['l_s']
    previous_end=0;previous_issue=-1
    for number,end in enumerate(endpoints):
        start=number*batch
        s=smooth[start:end];v=y[start:end];mean=float(s.mean());sd=float(s.std())
        raw_cut,raw_z=choose_threshold(s,cfg['error_buffer'],mean,sd);inv=2*mean-s
        inv_cut,inv_z=choose_threshold(inv,cfg['error_buffer'],mean,sd)
        inter=float(np.percentile(v,95)-np.percentile(v,5));v_sd=float(v.std());win_flags=np.zeros(len(s),bool)
        guard=bool(sd>0 and (sd>.05*v_sd or s.max()>.05*inter) and s.max()>.05)
        for seq,cut in [(s,raw_cut),(inv,inv_cut)]:
            if not guard: continue
            candidates=dilate((seq>=cut)&(seq>.05*inter),cfg['error_buffer'])
            lower=ignore if number==0 else previous_end-start
            candidates[:max(0,lower)]=False
            existing=flags[start:end];nominal=seq[~(existing|candidates)]
            normal_peak=float(nominal.max()) if len(nominal) else 0.
            win_flags|=prune(seq,candidates,normal_peak,cfg['p'])
        issue=max(previous_issue,end-1,startup-1)
        newly=win_flags&~flags[start:end]
        flags[start:end]|=win_flags;available[start:end][newly]=ix[issue]
        windows.append({'start':int(ix[start]),'end_exclusive':int(ix[end-1]+1),'issue_index':int(ix[issue]),
                        'epsilon':float(raw_cut),'epsilon_inverse':float(inv_cut),'z':raw_z,'z_inverse':inv_z,
                        'guard_pass':guard,'new_alarm_points':int(newly.sum()),'initial_ignore':int(ignore if number==0 else 0)})
        previous_end=end;previous_issue=issue
    return {'flags':flags,'issue_index':available,'smoothed':smooth,'windows':windows,'status':'OUTPUT_COMPLETE',
            'startup_averaging_lookahead':startup}
