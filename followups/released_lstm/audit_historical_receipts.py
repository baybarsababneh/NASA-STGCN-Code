"""Keep author-provided historical receipts distinct from new detector execution."""
from pathlib import Path
import csv,ast,json,zipfile,io,importlib.util
import numpy as np
R=Path(__file__).resolve().parent;OUT=R/'outputs'
s=importlib.util.spec_from_file_location('metrics',R/'frozen_reference/vendor/frozen/metrics.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rows=list(csv.DictReader((R.parent/'project_sources/02-labeled_anomalies.csv').open()));labels={}
for r in rows:labels.setdefault(r['chan_id'],r)
z=zipfile.ZipFile(R.parent/'project_sources/03-archive.zip');allrows=[];coordinate=[];literal=[]
for version,folder,offset in [('initial_2018','telemanom_v1',250),('current_v2','telemanom-master',0)]:
    saved=list(csv.DictReader(next((R/'sources'/folder/'results').glob('*.csv')).open()))
    source_labels={}
    for r in csv.DictReader((R/'sources'/folder/'labeled_anomalies.csv').open()):source_labels.setdefault(r['chan_id'],r)
    for mission in ['MSL','SMAP']:
        rr=[v for v in saved if v['spacecraft']==mission];h=sum(int(v['true_positives']) for v in rr);u=sum(int(v['false_positives']) for v in rr);miss=sum(int(v['false_negatives']) for v in rr)
        p=h/(h+u);rec=h/(h+miss)
        literal.append({'version':version,'mission':mission,'CSV_rows':len(rr),'H':h,'U':u,'M':miss,'precision':p,'recall':rec,'standard_F0_5_from_saved_counts':m.fbeta(p,rec),'duplicate_P-2_retained':True})
    seen=set()
    for r in saved:
        c=r['chan_id']
        if c in seen:continue
        seen.add(c)
        # This coordinate conversion is source-defined, not fitted to improve detection.
        supplied=ast.literal_eval(labels[c]['anomaly_sequences'])
        old=ast.literal_eval(source_labels[c]['anomaly_sequences'])
        te=np.load(io.BytesIO(z.read('data/data/test/'+c+'.npy')));ix=np.arange(250,len(te)-10)
        match=[(a+offset,b+offset) for a,b in old]==[tuple(v) for v in supplied]
        bounded=[(a+offset,min(b+offset,len(te)-1)) for a,b in old]
        bounded_match=bounded==[tuple(v) for v in supplied]
        assert bounded_match,(version,c,bounded,supplied)
        coordinate.append({'version':version,'channel':c,'offset_to_raw':offset,
                           'label_interval_coordinate_match':match,'match_after_raw_length_clipping':bounded_match,
                           'old_labels_converted_to_raw':[(a+offset,b+offset) for a,b in old],
                           'current_raw_labels':supplied})
        truth=np.zeros(len(ix),bool);flags=truth.copy();events=[(a,b+1) for a,b in supplied]
        for a,b in events:truth|=(ix>=a)&(ix<b)
        intervals=sorted(set(tuple(v) for v in ast.literal_eval(r['tp_sequences'])+ast.literal_eval(r['fp_sequences'])))
        for a,b in intervals:flags|=(ix>=a+offset)&(ix<=b+offset)
        h=sum(bool(flags[(ix>=a)&(ix<b)].any()) for a,b in events if ((ix>=a)&(ix<b)).any())
        total=sum(bool(((ix>=a)&(ix<b)).any()) for a,b in events);u=sum(not truth[a:b].any() for a,b in m.runs(flags));fp=int((flags&~truth).sum());n0=int((~truth).sum())
        tp=int((flags&truth).sum());fn=int((~flags&truth).sum());p=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else float('nan')
        allrows.append({'version':version,'channel':c,'mission':labels[c]['spacecraft'],**m.pooled([[h,total-h,u,fp,n0]]),'TP':tp,'FP':fp,'FN':fn,'point_f1':m.fbeta(p,rec,1.)})
summary=[]
for version in ['initial_2018','current_v2']:
    for mission in ['MSL','SMAP']:
        rr=[v for v in allrows if v['version']==version and v['mission']==mission];counts=[[v[k] for k in ['H','M','U','FPt','N0']] for v in rr]
        r=m.pooled(counts);tp=sum(v['TP'] for v in rr);fp=sum(v['FP'] for v in rr);fn=sum(v['FN'] for v in rr)
        r.update(version=version,mission=mission,channels=len(rr),TP=tp,FP=fp,FN=fn,strict_point_f1=2*tp/(2*tp+fp+fn),nominal_occupancy=fp/r['N0'],timeliness_available=False)
        summary.append(r)
for name,obj in [('historical_literal_count_receipts.json',literal),('historical_coordinate_audit.json',coordinate),('historical_intervals_rescored.json',summary)]:
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False))
with (OUT/'historical_intervals_channel_counts.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=allrows[0].keys());w.writeheader();w.writerows(allrows)
print(json.dumps(summary,indent=2))
