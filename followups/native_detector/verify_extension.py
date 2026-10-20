"""Independently recount both-detector masks and literal evaluator diagnostics."""
from pathlib import Path
import csv,ast,json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;O=R/'outputs';P=R.parent/'lstm_benchmark_audit'
labels={}
for r in csv.DictReader((R.parent/'project_sources/02-labeled_anomalies.csv').open()):labels.setdefault(r['chan_id'],r)
rows=pd.read_csv(O/'channel_metrics.csv');checks=[];differences=[]
for r in rows.to_dict('records'):
 c=r['channel'];key=f"{r['role']}__{c}__{r['seed']}__{r['pipeline']}.npz"
 path=O/'streams'/key
 if not path.exists():path=P/'outputs/streams'/key
 q=np.load(path);ix=q['target_index'];f=q['flags'];issue=q['issue_index'];ev=ast.literal_eval(labels[c]['anomaly_sequences']);truth=np.zeros(len(ix),bool)
 h=0;total=0;within=0;late=0
 for a,b in ev:
  use=(ix>=a)&(ix<=b);truth|=use
  if use.any():
   total+=1
   if (f&use).any():h+=1;t=int(issue[f&use].min());within+=t<=b;late+=t>b
 edges=np.flatnonzero(np.diff(np.r_[False,f,False].astype(int)));segments=list(zip(edges[::2],edges[1::2]))
 u=sum(not truth[a:b].any() for a,b in segments)
 counts={'H':h,'M':total-h,'U':u,'FPt':int((f&~truth).sum()),'N0':int((~truth).sum()),
 'TP':int((f&truth).sum()),'FP':int((f&~truth).sum()),'FN':int((~f&truth).sum()),'TN':int((~f&~truth).sum()),
 'within_event_notifications':within,'late_notifications':late,'missed_notifications':total-h}
 for k,v in counts.items():assert int(r[k])==v,(key,k,v,r[k])
 # Current author evaluator credits only the first true event touched by each run.
 matched=set();literal_u=0
 for a,b in segments:
  touched=[i for i,(x,y) in enumerate(ev) if int(ix[a])<=y and int(ix[b-1])>=x]
  if touched:matched.add(touched[0])
  else:literal_u+=1
 assert len(matched)==r['literal_v2_H'] and len(ev)-len(matched)==r['literal_v2_M'] and literal_u==r['literal_v2_U'],key
 assert np.all(issue[f]>=ix[f]) and np.all(issue[~f]==-1)
 if h!=len(matched):differences.append({'role':r['role'],'channel':c,'seed':r['seed'],'pipeline':r['pipeline'],
   'mission':r['mission'],'paper_style_events_hit':h,'literal_v2_events_hit':len(matched),'difference':len(matched)-h})
 checks.append({'stream':key,'independent_counts_and_literal_evaluator_pass':True})
old=json.loads((P/'PACKAGE_MANIFEST.json').read_text())
assert all(hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in old['files'].items())
(O/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps({'streams_checked':len(checks),'all_pass':True,
 'prior_files_verified_unchanged':old['count'],'literal_evaluator_discrepancy_streams':len(differences),
 'literal_evaluator_discrepancy_channels':sorted(set(r['channel'] for r in differences))},indent=2))
(O/'independent_arithmetic_checks.json').write_text(json.dumps(checks,indent=2))
pd.DataFrame(differences,columns=['role','channel','seed','pipeline','mission','paper_style_events_hit','literal_v2_events_hit','difference']).to_csv(O/'literal_evaluator_discrepancies.csv',index=False)
print((O/'INDEPENDENT_VERIFICATION.json').read_text())
