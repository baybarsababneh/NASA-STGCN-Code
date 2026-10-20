"""Version 2 metrics and causal scorers. Historical code is never modified.
EXP008 uses nominal diagnostics only. Labelled metrics are synthetic-test utilities,
not a NASA evaluator, and are not called on project data by run_exp008.py.
"""
import numpy as np
VERSION='EXP008_METRICS_V2_1'

def finite_vector(x):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError('Expected nonempty finite vector')
    return a

def labels_vector(y):
    a=np.asarray(y)
    if a.ndim!=1 or not len(a) or not np.isin(a,[0,1]).all():raise ValueError('Expected binary labels')
    return a.astype(bool)

def average_precision(y,scores):
    y=labels_vector(y);s=finite_vector(scores)
    if y.shape!=s.shape:raise ValueError('Shape mismatch')
    if not y.sum():return float('nan')  # Explicit undefined no-positive convention.
    order=np.argsort(-s,kind='stable');ys=y[order];ss=s[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)!=0),len(ss)-1]
    tp=np.cumsum(ys)[ends];precision=tp/(ends+1);recall=tp/y.sum()
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))

def ewma(values,span):
    x=finite_vector(values)
    if int(span)!=span or span<1:raise ValueError('Positive integer span required')
    alpha=2./(span+1);out=np.empty_like(x);out[0]=x[0]
    for i in range(1,len(x)):out[i]=alpha*x[i]+(1-alpha)*out[i-1]
    return out

def higher_threshold(scores,q):
    x=finite_vector(scores)
    if not 0<q<1:raise ValueError('q must be between0 and1')
    rank=int(np.ceil((len(x)-1)*q));t=float(np.quantile(x,q,method='higher'))
    return dict(threshold=t,n_cal=len(x),rank_zero_based=rank,
                rank_positions_above=len(x)-1-rank,calibration_exceedances=int((x>t).sum()),
                at_sample_max=bool(t==x.max()),unique_scores=int(len(np.unique(x))),
                iid_max_exceedance_reference=1./(len(x)+1),
                # Only a reference scale under IID continuous scores; never a guarantee.
                q=float(q))

def segments(mask):
    b=labels_vector(mask);edge=np.diff(np.r_[False,b,False].astype(np.int8))
    return list(zip(np.flatnonzero(edge==1).tolist(),np.flatnonzero(edge==-1).tolist()))

def burden(flags,previous_flag=False):
    a=labels_vector(flags);seg=segments(a);starts=int(np.sum(a & ~np.r_[bool(previous_flag),a[:-1]]))
    return dict(samples=len(a),flagged_samples=int(a.sum()),occupancy=float(a.mean()),
                alarm_starts=starts,alarm_starts_per_1000=1000.*starts/len(a),
                intersecting_runs=len(seg),longest_run=max([b-a for a,b in seg],default=0))

def fbeta(p,r,beta=.5):
    if r==0:return 0.
    if not np.isfinite(p) or not np.isfinite(r):return float('nan')
    d=beta*beta*p+r
    return (1+beta*beta)*p*r/d if d else 0.

def event_metrics(y,flags):
    y=labels_vector(y);p=labels_vector(flags)
    if y.shape!=p.shape:raise ValueError('Shape mismatch')
    truth=segments(y);pred=segments(p)
    h=sum(bool(p[a:b].any()) for a,b in truth);miss=len(truth)-h
    u=sum(not y[a:b].any() for a,b in pred);n0=int((~y).sum());fp=int((p&~y).sum())
    pe=h/(h+u) if h+u else float('nan');rec=h/(h+miss) if h+miss else float('nan')
    pc=pe*(1-fp/n0) if n0 else float('nan')
    return dict(H=h,M=miss,U=u,FPt=fp,N0=n0,event_precision=pe,event_recall=rec,
                corrected_precision=pc,corrected_event_f0_5=fbeta(pc,rec) if n0 and truth else float('nan'),
                hundman_overlap_f0_5=fbeta(pe,rec))
