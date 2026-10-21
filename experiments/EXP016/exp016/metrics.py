"""Inclusive interval overlap and strict point counts. No point adjustment."""
import ast,csv
import numpy as np
from .detector import runs

def labels(path,fixture=False):
    rows=list(csv.DictReader(open(path)));primary={};alternatives={}
    for row in rows:
        ch=row['chan_id'];ev=ast.literal_eval(row['anomaly_sequences'])
        if any(not isinstance(a,int) or not isinstance(b,int) or a>b or a<0 for a,b in ev):raise ValueError('Invalid annotation')
        if ch in primary:alternatives[ch]=ev
        else:primary[ch]={'events':ev,'mission':row['spacecraft'],'num_values':int(row['num_values'])}
    if not fixture and (len(primary)!=81 or set(alternatives)!={'P-2'} or 'T-10' in primary):raise ValueError('Unexpected label roster')
    return primary,alternatives

def average_precision(y,score):
    if y.sum()==0:return None
    order=np.argsort(-score,kind='stable');y=y[order];s=score[order]
    end=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1];tp=np.cumsum(y)[end]
    return float(np.sum(np.diff(np.r_[0,tp])/y.sum()*(tp/(end+1))))

def counts(events,flags,ix,issue=None,raw=None,smoothed=None):
    flags=np.asarray(flags,bool);ix=np.asarray(ix);truth=np.zeros(len(ix),bool)
    represented=[];outside=0
    for a,b in events:
        hit=(ix>=a)&(ix<=b)
        if hit.any():represented.append((a,b))
        else:outside+=1
        truth|=hit
    pred=[(int(ix[a]),int(ix[b])) for a,b in runs(flags)]
    tp=sum(any(a<=q and p<=b for p,q in pred) for a,b in represented)
    fp=sum(not any(a<=q and p<=b for a,b in represented) for p,q in pred)
    literal_matches=set()
    for p,q in pred:
        hits=[j for j,(a,b) in enumerate(represented) if a<=q and p<=b]
        if hits:literal_matches.add(hits[0])
    ptp=int((flags&truth).sum());pfp=int((flags&~truth).sum());pfn=int((~flags&truth).sum())
    starts=np.r_[True,~flags[:-1]]&flags
    notifications=[]
    for a,b in represented:
        idx=np.flatnonzero(flags&(ix>=a)&(ix<=b))
        first=int(issue[idx].min()) if len(idx) and issue is not None else None
        notifications.append(dict(start=a,end=b,overlap_hit=bool(len(idx)),first_issue=first,
             within_event=first is not None and first<=b,delay=None if first is None else first-a))
    result=dict(event_tp=int(tp),event_fp=int(fp),event_fn=len(represented)-int(tp),events=len(represented),
        original_events=len(events),unrepresented_events=outside,literal_event_tp=len(literal_matches),
        literal_event_fn=len(represented)-len(literal_matches),point_tp=ptp,point_fp=pfp,point_fn=pfn,
        nominal_points=int((~truth).sum()),nominal_starts=int((starts&~truth).sum()),points=len(ix),
        timely_events=sum(v['within_event'] for v in notifications))
    if raw is not None:result['raw_AP']=average_precision(truth,raw)
    if smoothed is not None:result['smoothed_AP']=average_precision(truth,smoothed)
    return result,notifications

def ratio(a,b):return float(a/b) if b else None
def fbeta(tp,fp,fn,beta=.5):return ratio((1+beta*beta)*tp,(1+beta*beta)*tp+beta*beta*fn+fp)
def aggregate(rows):
    keys=['event_tp','event_fp','event_fn','events','original_events','unrepresented_events','literal_event_tp',
          'literal_event_fn','point_tp','point_fp','point_fn','nominal_points','nominal_starts','points','timely_events']
    s={k:sum(r[k] for r in rows) for k in keys}
    tp,fp,fn=[s['event_'+k] for k in ('tp','fp','fn')]
    pp,pf,pn=[s['point_'+k] for k in ('tp','fp','fn')]
    s.update(event_precision=ratio(tp,tp+fp),event_recall=ratio(tp,tp+fn),event_F05=fbeta(tp,fp,fn),
        literal_event_F05=fbeta(s['literal_event_tp'],fp,s['literal_event_fn']),
        point_precision=ratio(pp,pp+pf),point_recall=ratio(pp,pp+pn),point_F1=fbeta(pp,pf,pn,1),point_F05=fbeta(pp,pf,pn),
        nominal_occupancy=ratio(pf,s['nominal_points']),nominal_starts_per_1000=ratio(1000*s['nominal_starts'],s['nominal_points']),
        timely_event_rate=ratio(s['timely_events'],s['events']))
    for key in ('raw_AP','smoothed_AP'):
        a=[r[key] for r in rows if r.get(key) is not None];s[key+'_macro']=float(np.mean(a)) if a else None
        s[key+'_defined_channels']=len(a)
    return s
