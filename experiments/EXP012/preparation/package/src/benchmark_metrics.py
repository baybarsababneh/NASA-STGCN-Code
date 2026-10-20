"""Pure-array future benchmark utilities; no labels/data loader or evaluation runner."""
import numpy as np
from detector_metrics_v2 import event_metrics,average_precision,fbeta,segments,labels_vector

def pooled_events(counts):
    totals={k:sum(r[k] for r in counts) for k in ['H','M','U','FPt','N0']}
    h,m,u,fp,n0=[totals[k] for k in ['H','M','U','FPt','N0']]
    pe=h/(h+u) if h+u else (0. if h+m else float('nan'))
    recall=h/(h+m) if h+m else float('nan');pc=pe*(1-fp/n0) if n0 else float('nan')
    return {**totals,'event_precision':pe,'event_recall':recall,'corrected_precision':pc,
            'corrected_event_f0_5':fbeta(pc,recall),'hundman_overlap_f0_5':fbeta(pe,recall)}

def strict_points(labels,flags):
    y=labels_vector(labels);p=labels_vector(flags)
    if y.shape!=p.shape:raise ValueError('Shape mismatch')
    tp=int((y&p).sum());fp=int((~y&p).sum());fn=int((y&~p).sum());tn=int((~y&~p).sum())
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else float('nan')
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,precision=precision,recall=recall,F1=fbeta(precision,recall,1.),F0_5=fbeta(precision,recall))

def actual_delays(labels,flags,issue_index,target_index,original_events=None):
    y=labels_vector(labels);p=labels_vector(flags);ix=np.asarray(target_index,np.int64);issue=np.asarray(issue_index,np.int64)
    if y.shape!=p.shape or p.shape!=ix.shape or ix.shape!=issue.shape or np.any(np.diff(ix)!=1):raise ValueError('Invalid event timing vectors')
    if np.any(issue[p]<ix[p]):raise ValueError('Notifications cannot precede their evidence target')
    result=[]
    events=original_events if original_events is not None else [(int(ix[a]),int(ix[b-1]+1)) for a,b in segments(y)]
    for onset,end in events:
        onset=int(onset);end=int(end)
        if end<=onset:raise ValueError('Half-open event interval required')
        support=(ix>=onset)&(ix<end);use=p&support
        notification=int(issue[use].min()) if use.any() else None
        result.append(dict(onset=onset,end_exclusive=end,overlap_hit=bool(use.any()),first_notification=notification,
                           delay=None if notification is None else notification-onset,
                           within_event_notification=notification is not None and notification<end,
                           evaluation_support=bool(support.any())))
    return result

def paired_bootstrap(a,b,seed=20261005,repetitions=10000):
    """a,b: lists of channels, each containing three per-seed event-count dictionaries.
    Difference is a-b; seeds remain grouped within every resampled channel.
    """
    if len(a)!=len(b) or not a or any(len(x)!=3 for x in a+b):raise ValueError('Paired channels × three seeds required')
    def value(records,ids):return float(np.mean([pooled_events([records[i][s] for i in ids])['corrected_event_f0_5'] for s in range(3)]))
    rng=np.random.default_rng(seed);n=len(a);draw=np.empty(repetitions)
    for i in range(repetitions):
        ids=rng.integers(0,n,n);draw[i]=value(a,ids)-value(b,ids)
    estimate=value(a,range(n))-value(b,range(n));finite=draw[np.isfinite(draw)]
    return {'difference':estimate if np.isfinite(estimate) else None,
            'percentile_CI95':np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,
            'undefined_replicates':repetitions-len(finite),'CI_condition':'finite replicates; disclose any undefined draws',
            'bootstrap_seed':seed,'repetitions':repetitions,'cluster':'channel; shared contexts may violate independence'}
