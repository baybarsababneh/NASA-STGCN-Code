"""Independent arithmetic checks and correctly mission-scoped sensitivities.

The initial analysis loop emitted a no-op C-2 exclusion for SMAP, although
C-2 belongs to MSL. Preserve that original file and write the corrected
prespecified sensitivity to a separate file. Primary results are unchanged.
"""
from pathlib import Path
import json,hashlib,io,zipfile,ast,csv,importlib.util
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parent;O=R/'outputs'
spec=importlib.util.spec_from_file_location('m',R/'frozen_reference/vendor/frozen/metrics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=pd.read_csv(O/'channel_metrics.csv');summary=pd.read_csv(O/'mission_metrics.csv')
assert len(rows)==648
# Independent event, point, burden and timing arithmetic directly from saved streams.
labels={}
for r in csv.DictReader((R.parent/'project_sources/02-labeled_anomalies.csv').open()):labels.setdefault(r['chan_id'],r)
z=zipfile.ZipFile(R.parent/'project_sources/03-archive.zip');receipts=[]
for r in rows.to_dict('records'):
    c=r['channel'];q=np.load(O/'streams'/f"{r['role']}__{c}__{r['seed']}__{r['pipeline']}.npz")
    ix=q['target_index'];f=q['flags'];issue=q['issue_index'];truth=np.zeros(len(ix),bool);events=ast.literal_eval(labels[c]['anomaly_sequences'])
    h=0;total=0;within=0;late=0
    for a,b in events:
        use=(ix>=a)&(ix<=b);truth|=use
        if use.any():
            total+=1
            if (f&use).any():
                h+=1;t=int(issue[f&use].min());within+=t<=b;late+=t>b
    splits=np.flatnonzero(np.diff(np.r_[False,f,False].astype(int)))
    starts=splits[::2];ends=splits[1::2]
    u=sum(not truth[a:b].any() for a,b in zip(starts,ends))
    count={'H':h,'M':total-h,'U':u,'FPt':int((f&~truth).sum()),'N0':int((~truth).sum()),
           'TP':int((f&truth).sum()),'FP':int((f&~truth).sum()),'FN':int((~f&truth).sum()),'TN':int((~f&~truth).sum()),
           'within_event_notifications':within,'late_notifications':late,'missed_notifications':total-h}
    for k,v in count.items():assert int(r[k])==v,(c,r['role'],r['seed'],r['pipeline'],k)
    assert np.all(issue[f]>=ix[f]) and np.all(issue[~f]==-1)
    receipts.append({'channel':c,'role':r['role'],'seed':r['seed'],'pipeline':r['pipeline'],'pass':True})
corrected=[]
for (mission,role,pipeline,seed),g in rows.groupby(['mission','role','pipeline','seed']):
    if mission!='MSL':continue
    rr=g[g.channel!='C-2'].to_dict('records');v=m.aggregate(rr)
    corrected.append({**v,'mission':mission,'role':role,'pipeline':pipeline,'seed':int(seed),'sensitivity':'exclude_C-2'})
old=pd.read_csv(O/'sensitivity_seed_metrics.csv');p2=old[old.sensitivity=='second_P-2_label']
result=pd.concat([pd.DataFrame(corrected),p2],ignore_index=True);result.to_csv(O/'sensitivity_verified_seed_metrics.csv',index=False)
v={'independent_stream_count_checks':len(receipts),'all_pass':True,
   'original_historical_outputs_modified':False,
   'sensitivity_correction':'Use sensitivity_verified_seed_metrics.csv; initial C-2 exclusion rows in sensitivity_seed_metrics.csv apply to SMAP and are no-ops. Correct C-2 exclusion applies to MSL.',
   'all_native_intervals_match_current_official_CSV':all(v['interval_match'] for v in json.loads((O/'native_archive_interval_regression.json').read_text()))}
(O/'INDEPENDENT_ANALYSIS_VERIFICATION.json').write_text(json.dumps(v,indent=2))
(O/'independent_stream_arithmetic_checks.json').write_text(json.dumps(receipts,indent=2))
print(json.dumps(v,indent=2));print(result.groupby(['mission','sensitivity','role','pipeline'])[['corrected_event_f0_5','point_f1']].mean().to_string())
