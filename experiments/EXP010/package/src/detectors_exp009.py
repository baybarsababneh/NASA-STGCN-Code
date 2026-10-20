"""Causal residual detectors. No model, data-loader, or NASA label dependency."""
import numpy as np

def robust_scale(x):
    x=np.asarray(x,dtype=np.float64)
    med=float(np.median(x));mad=float(np.median(np.abs(x-med)))
    return med,max(1.4826*mad,float(np.median(np.abs(x))),1e-6)

def ewma(x):
    y=np.empty(len(x),dtype=np.float64);y[0]=x[0]
    for i in range(1,len(x)):y[i]=(2/11)*x[i]+(9/11)*y[i-1]
    return y

def score(error,location,scale,family):
    z=(np.asarray(error,dtype=np.float64)-location)/scale
    if family=='absolute':return np.abs(z)
    if family=='ewma10':return ewma(np.abs(z))
    if family=='multiscale':
        result=np.abs(z);cs=np.r_[0.,np.cumsum(z,dtype=np.float64)]
        for w in (8,32):
            # Full windows only; startup uses already available shorter windows.
            v=np.zeros(len(z));v[w-1:]=np.abs((cs[w:]-cs[:-w])/np.sqrt(w))
            result=np.maximum(result,v)
        return result
    raise ValueError(family)

def calibrate(error,ref,candidate):
    if candidate['family']=='historical_raw99':
        scores=np.abs(error);return dict(location=0.,scale=1.,threshold=float(np.quantile(scores[ref],.99,method='higher')),family='historical_raw99')
    if candidate['family']=='historical_ewma995':
        scores=ewma(np.abs(error));return dict(location=0.,scale=1.,threshold=float(np.quantile(scores[ref],.995,method='higher')),family='historical_ewma995')
    location,scale=robust_scale(error[ref]);s=score(error,location,scale,candidate['family']);sr=s[ref]
    med=float(np.median(sr));iqr=float(np.quantile(sr,.75)-np.quantile(sr,.25))
    width=max(iqr/1.349,.1*med,1e-6)
    return dict(location=location,scale=scale,threshold=med+candidate['h']*width,score_median=med,score_scale=width,family=candidate['family'])

def apply(error,cal):
    family=cal['family']
    if family=='historical_raw99':s=np.abs(error)
    elif family=='historical_ewma995':s=ewma(np.abs(error))
    else:s=score(error,cal['location'],cal['scale'],family)
    return s,s>cal['threshold']

def burden(flags,mask):
    idx=np.flatnonzero(mask);assert len(idx)>0 and np.all(np.diff(idx)==1)
    f=flags[idx];previous=bool(flags[idx[0]-1]) if idx[0] else False
    starts=int(np.sum(f & ~np.r_[previous,f[:-1]]))
    edges=np.diff(np.r_[False,f,False].astype(int));lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    return dict(samples=len(f),flagged=int(f.sum()),occupancy=float(f.mean()),alarm_starts=starts,starts_per_1000=1000*starts/len(f),longest_run=int(lengths.max()) if len(lengths) else 0)

def stress(error,mask,cal,injection_scale,audit=False):
    """Residual-domain counterfactual probes, NOT perturbed-telemetry forecasts."""
    start,end=np.flatnonzero(mask)[[0,-1]];end+=1
    _,base=apply(error,cal);out=[]
    families=[('spike',1),('step8',8),('step32',32),('ramp32',32)]
    if audit:families += [('alternating32',32),('variance32',32)]
    for name,duration in families:
        requested_duration=duration;duration=min(duration,end-start)
        for amplitude in (2.,4.,8.):
            for sign in (-1,1):
                for position in (1/3,2/3):
                    onset=start+int(np.floor((end-start-duration)*position));offset=onset+duration
                    e=error.copy()
                    if name=='ramp32':delta=sign*amplitude*injection_scale*np.linspace(1/duration,1,duration)
                    elif name=='alternating32':delta=sign*amplitude*injection_scale*((-1.)**np.arange(duration))
                    elif name=='variance32':delta=sign*amplitude*(error[onset:offset]-cal['location'])
                    else:delta=sign*amplitude*injection_scale
                    e[onset:offset]+=delta;_,flag=apply(e,cal)
                    # Existing alarms do not count as sensitivity to a new disturbance.
                    base_hit=bool(base[onset:offset].any());hits=np.flatnonzero(flag[onset:offset])
                    hit=bool(len(hits)>0 and not base_hit);delay=int(hits[0]) if hit else duration
                    out.append(dict(family=name,amplitude=amplitude,sign=sign,position=position,onset=int(onset),duration=duration,requested_duration=requested_duration,baseline_hit=base_hit,incremental_hit=hit,delay_censored=delay,delay_fraction=delay/duration))
    return out
