"""Frozen count evaluation and bounded, descriptive paired comparisons."""
import os, csv, ast, shutil, uuid
from pathlib import Path
import numpy as np
import pandas as pd
from integrity import ROOT, read, sha, checked, arrays, write, atomic, blob, freeze, seal, manifest
import sys
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from vendor.frozen import metrics

COUNT_KEYS=['H','M','U','FPt','N0','TP','FP','FN','TN','nominal_alarm_starts']
MEAN_KEYS=['corrected_event_f0_5','event_precision','event_recall','corrected_precision','hundman_overlap_f0_5','point_precision','point_recall','point_f1','point_f0_5','nominal_occupancy','alarm_starts_per_1000_nominal','within_event_notification_rate','AP_raw_macro','AP_ewma_macro']

def labels(root,score_seal_verified=False):
    if not score_seal_verified: raise RuntimeError('Verify all scores before parsing labels')
    root=Path(root);roster=read(root/'inputs/TEST_ROSTER.json');p=root/'inputs/labeled_anomalies.csv'
    checked(p,read(root/'INPUT_INVENTORY.json')['present']['inputs/labeled_anomalies.csv'])
    with p.open() as f: rows=list(csv.DictReader(f))
    seen={};duplicates=[];events={}
    fixture=read(root/'config.json').get('fixture',False)
    for number,r in enumerate(rows):
        ch=r['chan_id']
        if ch in seen:
            if ch!='P-2': raise RuntimeError('Unexpected duplicated label channel')
            duplicates.append({'channel':ch,'kept_csv_row':seen[ch],'ignored_csv_row':number,'ignored_row':r});continue
        seen[ch]=number
        if ch not in roster or r['spacecraft']!=roster[ch]['mission'] or int(r['num_values'])!=roster[ch]['rows']: raise RuntimeError('Label/channel support mismatch')
        out=[]
        for x in ast.literal_eval(r['anomaly_sequences']):
            if len(x)!=2 or any(type(v) is not int for v in x) or not 0<=x[0]<=x[1]<roster[ch]['rows']: raise RuntimeError('Invalid inclusive label interval')
            out.append((x[0],x[1]+1))
        events[ch]=out
    if set(events)!=set(roster): raise RuntimeError('Label roster differs')
    if not fixture and (len(rows)!=82 or len(events)!=81 or sum(map(len,events.values()))!=104 or len(duplicates)!=1): raise RuntimeError('Frozen 81-channel/104-event roster differs')
    return events,{'CSV_rows':len(rows),'unique_channels':len(events),'original_events':sum(map(len,events.values())),'duplicates':duplicates,'first_P2_policy':True,'T10_has_no_label_row_and_is_not_added':True}

def mission_tables(records):
    per_seed=[];means=[]
    for mission in ['MSL','SMAP']:
        for arm in sorted({v['arm'] for v in records if v['mission']==mission}):
            group=[v for v in records if v['mission']==mission and v['arm']==arm]
            seeds=sorted({int(v['seed']) for v in group});result=[]
            for s in seeds:
                rows=[v for v in group if int(v['seed'])==s];agg=metrics.aggregate(rows)
                item={'mission':mission,'arm':arm,'seed':s,'panel':rows[0]['panel'],'detector':rows[0]['detector'],'source_experiment':rows[0]['source_experiment'],**agg}
                result.append(item);per_seed.append(item)
            row={'mission':mission,'arm':arm,'panel':group[0]['panel'],'detector':group[0]['detector'],'source_experiment':group[0]['source_experiment'],'seed_records':len(seeds),'full_roster':len(group)//len(seeds)}
            for k in MEAN_KEYS:
                values=np.array([v[k] for v in result]);row[k]=float(values.mean()) if np.isfinite(values).all() else None
                row[k+'_defined_seed_records']=int(np.isfinite(values).sum())
            means.append(row)
    return per_seed,means

def count_array(records,mission,arm,channels):
    d={(r['channel'],int(r['seed'])):r for r in records if r['mission']==mission and r['arm']==arm}
    seeds=sorted({s for c,s in d})
    if set(c for c,s in d)!=set(channels) or len(d)!=len(channels)*len(seeds): raise RuntimeError('Paired coverage is incomplete')
    return np.array([[[d[(ch,s)][k] for k in COUNT_KEYS] for s in seeds] for ch in channels],float)

def auxiliary_bootstrap(a,b,repetitions,seed):
    """Same paired channels/grouped seeds as EXP012; additional count metrics."""
    def score(x):
        h,m,u,fp,n0,tp,fp2,fn,tn,starts=np.moveaxis(x,-1,0)
        with np.errstate(invalid='ignore',divide='ignore'):
            den=2*tp+fp2+fn;f=np.where(tp+fn>0,np.where(den>0,2*tp/den,0.),np.nan)
            vals=np.stack([f,fp/n0,1000*starts/n0],axis=-1)
        return np.mean(vals,axis=-2)
    estimate=(score(a.sum(0)[None])-score(b.sum(0)[None]))[0]
    rng=np.random.default_rng(seed);draw=[]
    for start in range(0,repetitions,500):
        ix=rng.integers(0,len(a),size=(min(500,repetitions-start),len(a)))
        draw.append(score(a[ix].sum(1))-score(b[ix].sum(1)))
    draw=np.concatenate(draw)
    out={}
    for j,k in enumerate(['point_f1','nominal_occupancy','alarm_starts_per_1000_nominal']):
        v=draw[:,j];v=v[np.isfinite(v)]
        out[k]={'difference':float(estimate[j]),'percentile_CI95':np.quantile(v,[.025,.975]).tolist() if len(v) else None,'undefined_replicates':repetitions-len(v)}
    return out

def write_csv(path,rows,columns=None):
    rows=[{k:v for k,v in r.items() if not k.startswith('_')} for r in rows]
    frame=pd.DataFrame(rows,columns=columns)
    atomic(path,frame.to_csv(index=False,float_format='%.17g').encode())

