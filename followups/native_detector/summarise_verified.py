"""Independent pooled arithmetic and paired detector attribution, no scoring choices."""
from pathlib import Path
import json,sys,hashlib,platform
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;O=R/'outputs';P=R.parent/'lstm_benchmark_audit'
c=pd.read_csv(O/'channel_metrics.csv');s=pd.read_csv(O/'mission_seed_metrics.csv');a=pd.read_csv(O/'mission_metrics.csv')
def f(p,r,b=.5):
 d=b*b*p+r
 return (1+b*b)*p*r/d if d else 0.
for row in s.to_dict('records'):
 q=c[(c.mission==row['mission'])&(c.role==row['role'])&(c.pipeline==row['pipeline'])&(c.seed==row['seed'])]
 h,m,u,fp,n0,tp,pfp,fn=q[['H','M','U','FPt','N0','TP','FP','FN']].sum().to_numpy()
 ep=h/(h+u) if h+u else 0.;er=h/(h+m);pc=ep*(1-fp/n0)
 expect={'corrected_event_f0_5':f(pc,er),'hundman_overlap_f0_5':f(ep,er),
 'point_f1':2*tp/(2*tp+pfp+fn),'nominal_occupancy':fp/n0,'event_recall':er,
 'within_event_notification_rate':q.within_event_notifications.sum()/q.events_in_support.sum()}
 for k,v in expect.items():assert np.isclose(row[k],v,atol=1e-14,rtol=0),(row['mission'],row['role'],k)
for row in a.to_dict('records'):
 q=s[(s.mission==row['mission'])&(s.role==row['role'])&(s.pipeline==row['pipeline'])]
 for k in ['corrected_event_f0_5','hundman_overlap_f0_5','point_f1','nominal_occupancy','event_recall','within_event_notification_rate']:
  assert np.isclose(row[k],q[k].mean(),atol=1e-14,rtol=0),(row['mission'],row['role'],k)
keys=['mission','channel','role','seed']
b=c[c.pipeline=='common'].merge(c[c.pipeline=='native_v2'],on=keys,suffixes=('_adapted','_native'))
for k in ['H','M','U','FPt','TP','within_event_notifications']:b[k+'_delta']=b[k+'_native']-b[k+'_adapted']
changed=b[(b.H_delta!=0)|(b.M_delta!=0)|(b.U_delta!=0)]
changed[keys+[k+'_delta' for k in ['H','M','U','FPt','TP','within_event_notifications']]].to_csv(O/'event_count_changes.csv',index=False)
assert set(zip(changed.mission,changed.channel))=={('MSL','C-2')}
# Confirm literal masks, not only equal aggregate counts, on every other MSL channel.
mask_checks=0
for r in b[(b.mission=='MSL')&(b.channel!='C-2')].to_dict('records'):
 def path(pipe):
  name=f"{r['role']}__{r['channel']}__{r['seed']}__{pipe}.npz";p=O/'streams'/name
  return p if p.exists() else P/'outputs/streams'/name
 with np.load(path('common')) as x,np.load(path('native_v2')) as y:
  assert np.array_equal(x['flags'],y['flags'])
 mask_checks+=1
sens=pd.read_csv(O/'sensitivity_seed_metrics.csv')
sens.groupby(['sensitivity','mission','role','pipeline'],as_index=False)[['corrected_event_f0_5','point_f1','event_recall','nominal_occupancy','within_event_notification_rate']].mean().to_csv(O/'sensitivity_mission_metrics.csv',index=False)
summary={'independent_seed_aggregates_verified':len(s),'independent_mission_aggregates_verified':len(a),
 'only_channel_changing_event_H_M_U':'MSL C-2','other_MSL_common_native_masks_exactly_equal':mask_checks,
 'literal_evaluator_discrepancy_streams':0,'Python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,
 'platform':platform.platform()}
(O/'SUMMARY_VERIFICATION.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
