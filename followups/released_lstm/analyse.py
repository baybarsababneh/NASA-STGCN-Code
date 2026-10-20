"""LSTM_RELEASE_BENCHMARK_AUDIT_V1. Frozen predictions only; no training."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,zipfile,io,csv,ast,types,importlib.util,warnings
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
NASA=ROOT.parent/'project_sources/03-archive.zip'
EXPORT=ROOT/'materialized/NASA Research Project/EXP012_LABEL_BLIND_SCORE_EXPORT.zip'
SOURCE=ROOT/'exp012_source';FROZEN=ROOT/'frozen_reference'
def digest(b):return hashlib.sha256(b).hexdigest()
def load_module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
metrics=load_module('frozen_metrics',FROZEN/'vendor/frozen/metrics.py')
detector=load_module('frozen_detector',SOURCE/'src/hundman_detector.py')
assert (SOURCE/'src/hundman_detector.py').read_bytes()==(FROZEN/'vendor/frozen/hundman_detector.py').read_bytes()
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def write_json(path,obj):path.write_text(json.dumps(clean(obj),indent=2,allow_nan=False))
def write_csv(path,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r if not k.startswith('_')))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
def arr(z,p):return np.load(io.BytesIO(z.read(p)),allow_pickle=False)
def values(z,job):return np.load(io.BytesIO(z.read('units/'+job+'/scores.npz')),allow_pickle=False)

native_path=ROOT/'sources/telemanom-master/telemanom/errors.py'
native_text=native_path.read_text();old='np.delete(E_seq, i_to_remove, axis=0)'
assert native_text.count(old)==1
native_compat=native_text.replace(old,'np.delete(E_seq, i_to_remove.astype(np.int64), axis=0)')
native=types.ModuleType('official_native_compat');exec(compile(native_compat,str(native_path),'exec'),native.__dict__)
(ROOT/'sources/native_errors_integer_compat.py').write_text(native_compat)

def native_detect(c,te,pred):
 """Execute official methods, recording issue times without changing decisions."""
 cfg=types.SimpleNamespace(batch_size=70,window_size=30,smoothing_perc=.05,error_buffer=100,p=.13,l_s=250)
 n=len(pred);ix=np.arange(250,len(te)-10)
 y=np.column_stack([te[250+j:250+j+n,0] for j in range(10)])
 e=np.abs(y[:,0]-pred);smooth=pd.Series(e).ewm(span=105,adjust=True).mean().to_numpy()
 if c!='C-2':smooth[:250]=smooth[:500].mean()
 channel=types.SimpleNamespace(y_test=y,y_hat=pred,id=c)
 er=native.Errors.__new__(native.Errors);er.config=cfg;er.window_size=30
 er.n_windows=int((n-70*30)/70);er.i_anom=np.array([]);er.E_seq=[];er.anom_scores=[];er.e=e;er.e_s=smooth
 er.adjust_window_size(channel);issue=np.full(n,-1,np.int64);win=[]
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  for number in range(er.n_windows+1):
   start=number*70;end=30*70+number*70
   if number==er.n_windows:end=n
   w=native.ErrorWindow(channel,cfg,start,end,er,number)
   w.find_epsilon();w.find_epsilon(inverse=True)
   w.compare_to_epsilon(er);w.compare_to_epsilon(er,inverse=True)
   if len(w.i_anom) or len(w.i_anom_inv):
    w.prune_anoms();w.prune_anoms(inverse=True)
    if len(w.i_anom) or len(w.i_anom_inv):
     w.i_anom=np.sort(np.unique(np.append(w.i_anom,w.i_anom_inv))).astype(int)
     w.score_anomalies(start);loc=w.i_anom+start
     issued=250+max(end-1+9,(499 if c!='C-2' else 0))
     assert issued<len(te)
     fresh=issue[loc]<0;issue[loc[fresh]]=issued
     er.i_anom=np.append(er.i_anom,loc);er.anom_scores+=w.anom_scores
   win.append({'start':250+start,'end_exclusive':250+end,
      'issue_index':250+max(end-1+9,(499 if c!='C-2' else 0)),
      'epsilon':float(w.epsilon),'epsilon_inverse':float(w.epsilon_inv),
      'z':float(w.sd_threshold),'z_inverse':float(w.sd_threshold_inv)})
 flags=np.zeros(n,bool);flags[np.unique(er.i_anom).astype(int)]=True
 # Match official final sequence export, which discards isolated singleton flags.
 for a,b in metrics.runs(flags):
  if b-a==1:flags[a:b]=False;issue[a:b]=-1
 return {'flags':flags,'issue_index':issue,'smoothed':smooth,'windows':win,
         'warnings':sorted(set(str(w.message) for w in caught))}

def bootstrap_multi(a,b,metric,rep=10000):
 """Same-channel sampling; grouped project seeds vs a single released fit."""
 a=np.asarray(a,float);b=np.asarray(b,float);rng=np.random.default_rng(20261016)
 def score(v):
  if metric=='corrected_event_f0_5':
   h,m,u,fp,n0=np.moveaxis(v,-1,0);p=h/(h+u);r=h/(h+m);pc=p*(1-fp/n0)
   f=np.where(.25*pc+r>0,1.25*pc*r/(.25*pc+r),0.)
  else:
   tp,fp,fn=np.moveaxis(v,-1,0);den=2*tp+fp+fn;f=np.where(den>0,2*tp/den,0.)
  return np.mean(f,axis=-1)
 with np.errstate(divide='ignore',invalid='ignore'):
  estimate=float((score(a.sum(0)[None])-score(b.sum(0)[None]))[0]);draw=[]
  for start in range(0,rep,500):
   idx=rng.integers(0,len(a),size=(min(500,rep-start),len(a)))
   draw.extend((score(a[idx].sum(1))-score(b[idx].sum(1))).tolist())
 q=np.asarray(draw);finite=q[np.isfinite(q)]
 return {'metric':metric,'difference':estimate,'CI95':np.quantile(finite,[.025,.975]).tolist(),
  'CI98_75':np.quantile(finite,[.00625,.99375]).tolist(),'undefined_replicates':int((~np.isfinite(q)).sum()),
  'repetitions':rep,'seed':20261016,'condition':'saved fits and channels; shared-incident dependence unresolved'}

def main():
 z=zipfile.ZipFile(NASA);exp=zipfile.ZipFile(EXPORT)
 verify=json.loads((OUT/'SOURCE_VERIFICATION.json').read_text())
 assert verify['NASA_archive_sha256']==digest(NASA.read_bytes())
 assert verify['score_export_sha256']==digest(EXPORT.read_bytes())
 manifest=json.loads((SOURCE/'data_manifest.json').read_text())
 labels=list(csv.DictReader((ROOT.parent/'project_sources/02-labeled_anomalies.csv').open()))
 labelmap={};second={}
 for r in labels:
  c=r['chan_id'];ev=[(a,b+1) for a,b in ast.literal_eval(r['anomaly_sequences'])]
  if c in labelmap:second[c]=ev
  else:labelmap[c]=ev
 inputs={str(p.relative_to(ROOT.parent)):digest(p.read_bytes()) for p in [NASA,EXPORT,
  ROOT.parent/'project_sources/02-labeled_anomalies.csv',ROOT.parent/'upload/NASA_Project_SMAP_MSL_STGCN_Research_Report.pdf',
  SOURCE/'config.json',SOURCE/'src/lstm_baseline.py',SOURCE/'src/models_io.py',
  SOURCE/'src/hundman_detector.py',FROZEN/'vendor/frozen/metrics.py',native_path,ROOT/'PROTOCOL.json',Path(__file__)]}
 seal={'analysis_id':'LSTM_RELEASE_BENCHMARK_AUDIT_V1','input_sha256':inputs,
       'native_compat_sha256':digest(native_compat.encode()),'protocol_sha256':inputs['lstm_benchmark_audit/PROTOCOL.json']}
 sealpath=OUT/'ANALYSIS_FREEZE.json'
 if sealpath.exists():assert json.loads(sealpath.read_text())==seal,'Analysis inputs/source changed after scoring freeze'
 else:write_json(sealpath,seal)
 # Regression of existing exported decisions on original support, all 567 relevant streams.
 oldrows=list(csv.DictReader((FROZEN/'inputs/exp012_reports/channel_metrics.csv').open()))
 oldmap={(r['arm'],r['channel'],r['seed']):r for r in oldrows if r['detector']=='hundman_adapt_v1'}
 native_csv=list(csv.DictReader((ROOT/'sources/telemanom-master/results/2018-05-19_15.00.10.csv').open()))
 archive_map={}
 for r in native_csv:archive_map.setdefault(r['chan_id'],r)
 rows=[];notes=[];regression=[];native_matches=[];sensitivity=[]
 (OUT/'streams').mkdir(exist_ok=True);(OUT/'windows').mkdir(exist_ok=True)
 for i,row in enumerate(manifest):
  c=row['channel'];mission=row['mission'];te=arr(z,'data/data/test/'+c+'.npy');n=len(te)
  ix=np.arange(250,n-10);y=te[ix,0];events=labelmap[c]
  released=arr(z,'data/data/2018-05-19_15.00.10/y_hat/'+c+'.npy')
  roles=[('released_lstm',0,released)]
  for role in ['gcn','lstm']:
   for seed in [17,29,43]:
    job=f'{role}__{c}__{seed}';q=values(exp,job)
    assert np.array_equal(q['target_index'],np.arange(250,n))
    assert np.array_equal(q['target'],te[250:,0])
    rec,ev,_=metrics.channel(events,q['target_index'],q['adaptive_flags'],q['adaptive_issue_index'],q['scores_raw'],q['scores_ewma'])
    original=oldmap[(role,c,str(seed))]
    for k in ['H','M','U','FPt','N0','TP','FP','FN','TN','within_event_notifications','late_notifications','missed_notifications']:
     assert rec[k]==int(original[k]),(job,k,rec[k],original[k])
    regression.append({'job_id':job,'saved_count_and_notification_regression':True})
    roles.append((role,seed,q['prediction'][:-10]))
  for role,seed,pred in roles:
   d=detector.detect(y,pred,ix);key=f'{role}__{c}__{seed}__common'
   r,ev,out=metrics.channel(events,ix,d['flags'],d['issue_index'],np.abs(y-pred),d['smoothed'])
   r.update({'role':role,'channel':c,'mission':mission,'seed':seed,'pipeline':'common','MAE':float(np.abs(y-pred).mean()),'MSE':float(np.mean((y-pred)**2))})
   r['_delays']=[v['delay'] for v in ev if v['first_notification'] is not None];rows.append(r)
   notes.extend({**v,'role':role,'channel':c,'mission':mission,'seed':seed,'pipeline':'common'} for v in ev)
   np.savez_compressed(OUT/'streams'/f'{key}.npz',target_index=ix,flags=d['flags'],issue_index=d['issue_index'],smoothed=d['smoothed'])
   write_json(OUT/'windows'/f'{key}.json',d['windows'])
   if c in second:
    rr,_,_=metrics.channel(second[c],ix,d['flags'],d['issue_index'],np.abs(y-pred),d['smoothed'])
    sensitivity.append({**rr,'role':role,'channel':c,'mission':mission,'seed':seed,'pipeline':'common'})
  # Complementary original-author detector pipeline on released predictions.
  d=native_detect(c,te,released);key=f'released_lstm__{c}__0__native_v2'
  r,ev,out=metrics.channel(events,ix,d['flags'],d['issue_index'],np.abs(y-released),d['smoothed'])
  r.update({'role':'released_lstm','channel':c,'mission':mission,'seed':0,'pipeline':'native_v2','MAE':float(np.abs(y-released).mean()),'MSE':float(np.mean((y-released)**2))})
  r['_delays']=[v['delay'] for v in ev if v['first_notification'] is not None];rows.append(r)
  notes.extend({**v,'role':'released_lstm','channel':c,'mission':mission,'seed':0,'pipeline':'native_v2'} for v in ev)
  np.savez_compressed(OUT/'streams'/f'{key}.npz',target_index=ix,flags=d['flags'],issue_index=d['issue_index'],smoothed=d['smoothed'])
  write_json(OUT/'windows'/f'{key}.json',d['windows'])
  saved=archive_map[c];savedsegments=sorted(set(tuple(v) for v in ast.literal_eval(saved['tp_sequences'])+ast.literal_eval(saved['fp_sequences'])))
  gotsegments=[(250+a,250+b-1) for a,b in metrics.runs(d['flags'])]
  native_matches.append({'channel':c,'interval_match':savedsegments==gotsegments,'saved_intervals':savedsegments,'new_intervals':gotsegments,'warnings':d['warnings']})
  if c in second:
   rr,_,_=metrics.channel(second[c],ix,d['flags'],d['issue_index'],np.abs(y-released),d['smoothed'])
   sensitivity.append({**rr,'role':'released_lstm','channel':c,'mission':mission,'seed':0,'pipeline':'native_v2'})
  print(f'[{i+1}/81] completed frozen-artifact comparison: {c}',flush=True)
 write_csv(OUT/'channel_metrics.csv',rows);write_csv(OUT/'event_notifications.csv',notes)
 write_json(OUT/'EXP012_full_support_regression.json',regression);write_json(OUT/'native_archive_interval_regression.json',native_matches)
 # Summaries: ratio of pooled counts within seed; average ratios across project seeds.
 groups={}
 for r in rows:groups.setdefault((r['mission'],r['role'],r['pipeline']),[]).append(r)
 seedrows=[];summary=[]
 for (mission,role,pipeline),rr in groups.items():
  seeds=sorted(set(v['seed'] for v in rr));summ=[]
  for seed in seeds:
   sub=[v for v in rr if v['seed']==seed];v=metrics.aggregate(sub)
   v.update(mission=mission,role=role,pipeline=pipeline,seed=seed);seedrows.append(v);summ.append(v)
  v={k:float(np.mean([r[k] for r in summ])) for k in summ[0] if isinstance(summ[0][k],(int,float,np.generic)) and k!='seed'}
  v.update(mission=mission,role=role,pipeline=pipeline,seeds=len(seeds));summary.append(v)
 write_csv(OUT/'mission_seed_metrics.csv',seedrows);write_csv(OUT/'mission_metrics.csv',summary)
 contrasts=[]
 for mission in ['MSL','SMAP']:
  channels=sorted(r['channel'] for r in rows if r['mission']==mission and r['role']=='released_lstm' and r['pipeline']=='common')
  for role in ['gcn','lstm']:
   for metric,keys in [('corrected_event_f0_5',['H','M','U','FPt','N0']),('point_f1',['TP','FP','FN'])]:
    a=[[[next(r for r in groups[(mission,role,'common')] if r['channel']==c and r['seed']==s)[k] for k in keys] for s in [17,29,43]] for c in channels]
    b=[[[next(r for r in groups[(mission,'released_lstm','common')] if r['channel']==c)[k] for k in keys]] for c in channels]
    contrasts.append({**bootstrap_multi(a,b,metric), 'mission':mission,'contrast':role+'_minus_released_lstm','pipeline':'common'})
 write_json(OUT/'paired_channel_bootstrap.json',contrasts)
 sens=[]
 for (mission,role,pipeline),rr in groups.items():
  for which in ['exclude_C-2','second_P-2_label']:
   if mission!='SMAP':continue
   for seed in sorted(set(v['seed'] for v in rr)):
    sub=[v for v in rr if v['seed']==seed and (v['channel']!='C-2' or which!='exclude_C-2')]
    if which=='second_P-2_label':
     replacement=next(v for v in sensitivity if v['role']==role and v['seed']==seed and v['pipeline']==pipeline)
     sub=[v for v in sub if v['channel']!='P-2']+[replacement]
    r=metrics.aggregate(sub);r.update(mission=mission,role=role,pipeline=pipeline,seed=seed,sensitivity=which);sens.append(r)
 write_csv(OUT/'sensitivity_seed_metrics.csv',sens)
 write_json(OUT/'ANALYSIS_COMPLETE.json',{'analysis_id':seal['analysis_id'],'protocol_sha256':seal['protocol_sha256'],
  'source_verification_sha256':digest((OUT/'SOURCE_VERIFICATION.json').read_bytes()),'new_training_fits':0,
  'common_detector_streams':567,'native_detector_streams':81,'original_support_count_regressions':len(regression),
  'native_saved_intervals_exact':sum(r['interval_match'] for r in native_matches),'native_interval_mismatches':[r['channel'] for r in native_matches if not r['interval_match']],
  'label_tuning':False,'historical_results_modified':False})
 print(pd.DataFrame(summary)[['mission','role','pipeline','corrected_event_f0_5','hundman_overlap_f0_5','point_f1','nominal_occupancy','within_event_notification_rate']].to_string(index=False))
if __name__=='__main__':main()
