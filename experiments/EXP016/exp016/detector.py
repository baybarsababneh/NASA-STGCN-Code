"""Pinned author v2 detector with explicit compatibility/safety instrumentation.

Primary removes C-2's label-informed exception. Both arms retain retrospective
10-target scale guards, lookahead and startup averaging; issue times charge them.
Native sensitivity preserves the exception. No parameter tuning or label access.
"""
import types,warnings
import numpy as np
import pandas as pd
from .io import ROOT

_source=(ROOT/'vendor/errors_original.py').read_text()
_old='np.delete(E_seq, i_to_remove, axis=0)'
assert _source.count(_old)==1
_native=types.ModuleType('exp016_native_errors')
exec(compile(_source.replace(_old,'np.delete(E_seq, i_to_remove.astype(np.int64), axis=0)'),
             str(ROOT/'vendor/errors_original.py'),'exec'),_native.__dict__)

class SafeWindow(_native.ErrorWindow):
    def find_epsilon(self,inverse=False):
        # At zero SD upstream's NaN scores never replace its initialized threshold.
        if self.sd_e_s==0:return
        return super().find_epsilon(inverse)
    def compare_to_epsilon(self,errors_all,inverse=False):
        try:return super().compare_to_epsilon(errors_all,inverse)
        except ValueError as e:
            if 'zero-size array to reduction operation maximum' not in str(e):raise
            # An empty nominal reference makes pruning undefined. Primary explicitly
            # abstains for this branch and records it, instead of inventing a maximum.
            self.degenerate.append({'window':self.window_num,'inverse':inverse,'reason':'empty_nominal_pruning_reference_branch_abstention'})
            if inverse:self.i_anom_inv=np.array([])
            else:self.i_anom=np.array([])

def detect(test,pred,channel,mode='uniform',reference=False):
    n=len(pred)
    if n!=len(test)-260 or n<1 or not np.isfinite(pred).all():raise ValueError('Detector support/prediction mismatch')
    cfg=types.SimpleNamespace(batch_size=70,window_size=30,smoothing_perc=.05,error_buffer=100,l_s=250,p=.13)
    y=np.stack([test[250+k:250+k+n,0] for k in range(10)],axis=1)
    ch=types.SimpleNamespace(id=channel,y_test=y,y_hat=pred)
    e=_native.Errors.__new__(_native.Errors);e.config=cfg;e.window_size=30;e.n_windows=int((n-2100)/70)
    e.i_anom=np.array([]);e.E_seq=[];e.anom_scores=[]
    e.e=np.abs(np.asarray(pred)-y[:,0]);e.e_s=pd.Series(e.e).ewm(span=105,adjust=True).mean().to_numpy()
    exception=(mode=='native_C2' and channel=='C-2')
    if not exception:e.e_s[:250]=e.e_s[:500].mean()
    if reference:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');e.process_batches(ch)
        return e.E_seq
    e.adjust_window_size(ch);flags=np.zeros(n,bool);issue=np.full(n,-1,np.int64);receipts=[];degenerate=[]
    cls=SafeWindow if mode=='uniform' else _native.ErrorWindow
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        for i in range(e.n_windows+1):
            start=i*70;end=2100+i*70
            if i==e.n_windows:end=n
            w=cls(ch,cfg,start,end,e,i)
            if mode=='uniform':w.degenerate=degenerate
            for inverse in (False,True):w.find_epsilon(inverse)
            for inverse in (False,True):w.compare_to_epsilon(e,inverse)
            if len(w.i_anom) or len(w.i_anom_inv):
                w.prune_anoms();w.prune_anoms(True)
            inds=np.sort(np.unique(np.r_[w.i_anom,w.i_anom_inv])).astype(int)+start
            # y_test guard sees 9 later values than the scored first-horizon targets.
            available=250+min(end,n)-1+9
            if not exception:available=max(available,250+min(500,n)-1)
            fresh=inds[~flags[inds]];flags[inds]=True;issue[fresh]=available
            if len(inds):
                w.i_anom=inds-start;w.score_anomalies(start)
                e.anom_scores.extend(dict(start_idx=int(v['start_idx'])+250,end_idx=int(v['end_idx'])+250,score=float(v['score']),issue_index=available) for v in w.anom_scores)
            e.i_anom=np.append(e.i_anom,inds)
            receipts.append(dict(window=i,start=start,end_exclusive=min(end,n),issue_index=available,
                threshold=float(w.epsilon),inverse_threshold=float(w.epsilon_inv),new_flags=len(fresh)))
    # Match native exclusion of singleton predicted runs.
    intervals=runs(flags)
    for a,b in intervals:
        if a==b:flags[a]=False;issue[a]=-1
    seq=[(a+250,b+250) for a,b in runs(flags)]
    return dict(flags=flags,issue_index=issue,smoothed=e.e_s,raw_error=e.e,intervals=seq,windows=receipts,
                anomaly_scores=e.anom_scores,degeneracies=degenerate,warnings=sorted(set(str(x.message) for x in ws)))

def runs(mask):
    ix=np.flatnonzero(mask)
    if not len(ix):return []
    split=np.where(np.diff(ix)>1)[0]+1
    return [(int(v[0]),int(v[-1])) for v in np.split(ix,split)]
