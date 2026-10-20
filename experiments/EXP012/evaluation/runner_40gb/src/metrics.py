"""New evaluation implementation of the unchanged contract; no preparation edits."""
import numpy as np

def fbeta(p,r,b=.5):
 if not np.isfinite(p) or not np.isfinite(r):return float('nan')
 d=b*b*p+r;return (1+b*b)*p*r/d if d else 0.
def runs(a):
 a=np.asarray(a,bool);e=np.diff(np.r_[False,a,False].astype(np.int8));return list(zip(np.flatnonzero(e==1).tolist(),np.flatnonzero(e==-1).tolist()))
def ap(y,s):
 y=np.asarray(y,bool);s=np.asarray(s,float)
 if y.shape!=s.shape or y.ndim!=1 or not np.isfinite(s).all():raise ValueError('Invalid AP inputs')
 if not y.sum():return float('nan')
 order=np.argsort(-s,kind='stable');y=y[order];s=s[order];ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1];tp=np.cumsum(y)[ends];rec=tp/y.sum();prec=tp/(ends+1)
 return float(np.sum(np.diff(np.r_[0.,rec])*prec))
def pooled(c):
 c=np.asarray(c);h,m,u,fp,n0=np.sum(c,axis=0).tolist();p=h/(h+u) if h+u else (0. if h+m else float('nan'));r=h/(h+m) if h+m else float('nan')
 pc=p*(1-fp/n0) if n0 else float('nan')
 return {'H':int(h),'M':int(m),'U':int(u),'FPt':int(fp),'N0':int(n0),'event_precision':p,'event_recall':r,'corrected_precision':pc,'corrected_event_f0_5':fbeta(pc,r),'hundman_overlap_f0_5':fbeta(p,r),'zero_nominal_support':n0==0}
def count_vector(r):return [r[k] for k in ['H','M','U','FPt','N0']]
def channel(events,ix,flags,issue,raw,ewma):
 ix=np.asarray(ix,np.int64);f=np.asarray(flags,bool);issue=np.asarray(issue,np.int64)
 if ix.ndim!=1 or f.shape!=ix.shape or issue.shape!=ix.shape or np.any(np.diff(ix)!=1):raise ValueError('Invalid support')
 if np.any(issue[f]<ix[f]) or np.any(issue[~f]!=-1):raise ValueError('Invalid issue times')
 truth=np.zeros(len(ix),bool);supported=[];outside=[]
 for a,b in events:
  use=(ix>=a)&(ix<b)
  if use.any():truth|=use;supported.append((a,b,use))
  else:outside.append((a,b))
 segments=runs(f);event_records=[];hit=0
 for a,b,use in supported:
  overlap=f&use;hit+=bool(overlap.any());notification=int(issue[overlap].min()) if overlap.any() else None
  event_records.append({'onset':a,'end_exclusive':b,'overlap_hit':bool(overlap.any()),'first_notification':notification,'delay':None if notification is None else notification-a,'within_event_notification':notification is not None and notification<b,'late_post_event_notification':notification is not None and notification>=b,'fragmentation':sum(bool(use[x:y].any()) for x,y in segments),'left_truncated':a<int(ix[0])})
 h=hit;m=len(supported)-hit;u=sum(not truth[a:b].any() for a,b in segments);fp=int((f&~truth).sum());n0=int((~truth).sum());tp=int((f&truth).sum());fn=int((~f&truth).sum());tn=int((~f&~truth).sum());precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else float('nan')
 nominal_starts=sum(not truth[a] for a,b in segments);false_runs=len(runs(f&~truth))
 r={**pooled([[h,m,u,fp,n0]]),'samples':len(ix),'anomalous_samples':int(truth.sum()),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'point_precision':precision,'point_recall':recall,'point_f1':fbeta(precision,recall,1.),'point_f0_5':fbeta(precision,recall),'AP_raw':ap(truth,raw),'AP_ewma':ap(truth,ewma),'alarm_starts':len(segments),'nominal_alarm_starts':nominal_starts,'wholly_nominal_alarm_runs':u,'false_positive_runs':false_runs,'alarm_starts_per_1000_nominal':1000*nominal_starts/n0 if n0 else float('nan'),'wholly_nominal_runs_per_1000_nominal':1000*u/n0 if n0 else float('nan'),'nominal_occupancy':fp/n0 if n0 else float('nan'),'original_events':len(events),'events_in_support':len(supported),'events_fully_outside_support':len(outside),'within_event_notifications':sum(v['within_event_notification'] for v in event_records),'late_notifications':sum(v['late_post_event_notification'] for v in event_records),'missed_notifications':sum(v['first_notification'] is None for v in event_records),'left_truncated_events':sum(v['left_truncated'] for v in event_records)}
 return r,event_records,[{'onset':a,'end_exclusive':b,'evaluation_support':False} for a,b in outside]
def aggregate(rows):
 r=pooled([count_vector(v) for v in rows]);tp=sum(v['TP'] for v in rows);fp=sum(v['FP'] for v in rows);fn=sum(v['FN'] for v in rows);tn=sum(v['TN'] for v in rows);p=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else float('nan')
 valid={k:[v[k] for v in rows if np.isfinite(v[k])] for k in ['AP_raw','AP_ewma']};n0=r['N0'];supported=sum(v['events_in_support'] for v in rows);delays=[x for v in rows for x in v.get('_delays',[])]
 return {**r,'channels':len(rows),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'point_precision':p,'point_recall':rec,'point_f1':fbeta(p,rec,1.),'point_f0_5':fbeta(p,rec),'AP_raw_macro':float(np.mean(valid['AP_raw'])) if valid['AP_raw'] else float('nan'),'AP_ewma_macro':float(np.mean(valid['AP_ewma'])) if valid['AP_ewma'] else float('nan'),'AP_raw_defined_channels':len(valid['AP_raw']),'AP_ewma_defined_channels':len(valid['AP_ewma']),'zero_N0_channels':sum(v['N0']==0 for v in rows),'nominal_occupancy':fp/n0 if n0 else float('nan'),'nominal_alarm_starts':sum(v['nominal_alarm_starts'] for v in rows),'alarm_starts_per_1000_nominal':1000*sum(v['nominal_alarm_starts'] for v in rows)/n0 if n0 else float('nan'),'within_event_notification_rate':sum(v['within_event_notifications'] for v in rows)/supported if supported else float('nan'),'late_notifications':sum(v['late_notifications'] for v in rows),'missed_notifications':sum(v['missed_notifications'] for v in rows),'events_fully_outside_support':sum(v['events_fully_outside_support'] for v in rows),'notification_delay_median':float(np.median(delays)) if delays else float('nan'),'notification_delay_defined_events':len(delays)}
def bootstrap(a,b,repetitions=10000,seed=20261005):
 """Channels x seeds x counts. Persistence retains one seed slot, never three."""
 a=np.asarray(a,float);b=np.asarray(b,float)
 if a.ndim!=3 or b.ndim!=3 or a.shape[0]!=b.shape[0] or a.shape[2]!=5 or b.shape[2]!=5 or a.shape[1] not in [1,3] or b.shape[1] not in [1,3]:raise ValueError('Invalid paired channel/seed counts')
 def score(x):
  h,m,u,fp,n0=np.moveaxis(x,-1,0)
  with np.errstate(invalid='ignore',divide='ignore'):
   p=np.where(h+u>0,h/(h+u),np.where(h+m>0,0.,np.nan));r=np.where(h+m>0,h/(h+m),np.nan);pc=np.where(n0>0,p*(1-fp/n0),np.nan);den=.25*pc+r;f=np.where(den>0,1.25*pc*r/den,0.);f=np.where(np.isfinite(pc)&np.isfinite(r),f,np.nan)
  return np.mean(f,axis=-1)
 estimate=(score(a.sum(axis=0)[None])-score(b.sum(axis=0)[None])).item();rng=np.random.default_rng(seed);draw=[]
 for start in range(0,repetitions,500):
  ids=rng.integers(0,len(a),size=(min(500,repetitions-start),len(a)));draw.extend((score(a[ids].sum(axis=1))-score(b[ids].sum(axis=1))).tolist())
 finite=np.asarray(draw)[np.isfinite(draw)]
 return {'difference':estimate,'percentile_CI95':np.quantile(finite,[.025,.975]).tolist() if len(finite) else None,'undefined_replicates':repetitions-len(finite),'repetitions':repetitions,'bootstrap_seed':seed,'pairing':'same channels; all original seeds grouped; persistence one stream','CI_condition':'finite replicates; report undefined draws and shared-context/dependence limitations'}
