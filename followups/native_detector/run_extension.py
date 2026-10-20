"""Complete native/adapted detector comparison using frozen forecasts only."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import ast,types,sys,json,hashlib,zipfile,io,csv,logging,warnings,importlib.util
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;PRIOR=ROOT.parent/'lstm_benchmark_audit';OUT=ROOT/'outputs'
NASA=ROOT.parent/'project_sources/03-archive.zip'
EXPORT=PRIOR/'materialized/NASA Research Project/EXP012_LABEL_BLIND_SCORE_EXPORT.zip'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def save(name,x):(OUT/name).write_text(json.dumps(clean(x),indent=2,allow_nan=False))
def write_csv(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r if not k.startswith('_')))
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def load_mod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def array(z,n):return np.load(io.BytesIO(z.read(n)),allow_pickle=False)

metrics=load_mod('metrics',PRIOR/'frozen_reference/vendor/frozen/metrics.py')
native_original=(PRIOR/'sources/telemanom-master/telemanom/errors.py').read_text()
old='np.delete(E_seq, i_to_remove, axis=0)';assert native_original.count(old)==1
native_text=native_original.replace(old,'np.delete(E_seq, i_to_remove.astype(np.int64), axis=0)')
native=types.ModuleType('native');exec(compile(native_text,'pinned_native_errors','exec'),native.__dict__)
# Reuse the prior adapter function exactly, without importing its file-writing module.
prior_tree=ast.parse((PRIOR/'analyse.py').read_text())
selected=[n for n in prior_tree.body if isinstance(n,ast.FunctionDef) and n.name in ['native_detect','bootstrap_multi']]
assert len(selected)==2
globals_for_adapter={'np':np,'pd':pd,'warnings':warnings,'types':types,'native':native,'metrics':metrics}
exec(compile(ast.Module(body=selected,type_ignores=[]),'prior_adapter_functions','exec'),globals_for_adapter)
native_detect=globals_for_adapter['native_detect'];bootstrap=globals_for_adapter['bootstrap_multi']
# Execute the author evaluator method verbatim as a separate diagnostic.
eval_tree=ast.parse((PRIOR/'sources/telemanom-master/telemanom/detector.py').read_text())
method=next(m for n in eval_tree.body if isinstance(n,ast.ClassDef) and n.name=='Detector' for m in n.body if isinstance(m,ast.FunctionDef) and m.name=='evaluate_sequences')
eval_class=ast.ClassDef(name='LiteralEvaluator',bases=[],keywords=[],body=[method],decorator_list=[])
ns={'np':np,'logger':logging.getLogger('literal_event_evaluator')}
exec(compile(ast.fix_missing_locations(ast.Module(body=[eval_class],type_ignores=[])),'literal_evaluator','exec'),ns)

def literal_counts(events,ix,flags):
 e=types.SimpleNamespace(E_seq=[(int(ix[a]),int(ix[b-1])) for a,b in metrics.runs(flags)],anom_scores=[])
 q=ns['LiteralEvaluator']();q.result_tracker={}
 r=q.evaluate_sequences(e,{'anomaly_sequences':repr([[a,b-1] for a,b in events])})
 h=r['true_positives'];m=r['false_negatives'];u=r['false_positives']
 p=h/(h+u) if h+u else 0.;rec=h/(h+m) if h+m else float('nan')
 return {'literal_v2_H':h,'literal_v2_M':m,'literal_v2_U':u,'literal_v2_event_f0_5':metrics.fbeta(p,rec)}

def verify_original_loop(c,te,pred,computed):
 """Independent original process_batches control flow, no save/load/training calls."""
 cfg=types.SimpleNamespace(batch_size=70,window_size=30,smoothing_perc=.05,error_buffer=100,p=.13,l_s=250)
 n=len(pred);y=np.column_stack([te[250+j:250+j+n,0] for j in range(10)])
 ch=types.SimpleNamespace(y_test=y,y_hat=pred,id=c)
 e=native.Errors.__new__(native.Errors);e.config=cfg;e.window_size=30;e.n_windows=int((n-2100)/70)
 e.i_anom=np.array([]);e.E_seq=[];e.anom_scores=[];e.e=np.abs(pred-y[:,0]);e.e_s=pd.Series(e.e).ewm(span=105).mean().to_numpy()
 if c!='C-2':e.e_s[:250]=e.e_s[:500].mean()
 with warnings.catch_warnings():warnings.simplefilter('ignore');e.process_batches(ch)
 expected=np.zeros(n,bool)
 for a,b in e.E_seq:expected[a-250:b-250+1]=True
 assert np.array_equal(expected,computed['flags']),c
 assert np.array_equal(e.e_s,computed['smoothed']),c
 return len(e.E_seq)

def main():
 OUT.mkdir(exist_ok=True);(OUT/'streams').mkdir(exist_ok=True);(OUT/'windows').mkdir(exist_ok=True)
 old_manifest=json.loads((PRIOR/'PACKAGE_MANIFEST.json').read_text())
 for p,h in old_manifest['files'].items():assert sha(PRIOR/p)==h,p
 assert sha(NASA)==json.loads((PRIOR/'outputs/SOURCE_VERIFICATION.json').read_text())['NASA_archive_sha256']
 assert sha(EXPORT)==json.loads((PRIOR/'outputs/SOURCE_VERIFICATION.json').read_text())['score_export_sha256']
 inputs={str(p.relative_to(ROOT.parent)):sha(p) for p in [Path(__file__),ROOT/'PROTOCOL.json',
  PRIOR/'PACKAGE_MANIFEST.json',PRIOR/'analyse.py',PRIOR/'outputs/channel_metrics.csv',
  PRIOR/'outputs/ANALYSIS_FREEZE.json',PRIOR/'sources/telemanom-master/telemanom/errors.py',
  PRIOR/'sources/telemanom-master/telemanom/detector.py',NASA,EXPORT,
  ROOT.parent/'project_sources/02-labeled_anomalies.csv']}
 freeze={'analysis_id':'HUNDMAN_NATIVE_DETECTOR_CROSSOVER_V1','inputs':inputs,
  'native_compat_sha256':hashlib.sha256(native_text.encode()).hexdigest(),
  'prior_package_files_verified_unchanged':len(old_manifest['files'])}
 if (OUT/'ANALYSIS_FREEZE.json').exists():assert json.loads((OUT/'ANALYSIS_FREEZE.json').read_text())==freeze
 else:save('ANALYSIS_FREEZE.json',freeze)
 labels={};alternate={}
 for r in csv.DictReader((ROOT.parent/'project_sources/02-labeled_anomalies.csv').open()):
  ev=[(a,b+1) for a,b in ast.literal_eval(r['anomaly_sequences'])];c=r['chan_id']
  if c in labels:alternate[c]=ev
  else:labels[c]=ev
 z=zipfile.ZipFile(NASA);exports=zipfile.ZipFile(EXPORT)
 manifest=json.loads((PRIOR/'exp012_source/data_manifest.json').read_text())
 old_rows=pd.read_csv(PRIOR/'outputs/channel_metrics.csv').to_dict('records');rows=[];notes=[];parity=[];comparison=[];sens=[];failures=[];literal_checks=[]
 for r in old_rows:
  ix=np.load(PRIOR/'outputs/streams'/f"{r['role']}__{r['channel']}__{r['seed']}__{r['pipeline']}.npz")
  ev=labels[r['channel']];r.update(literal_counts(ev,ix['target_index'],ix['flags']));rows.append(r)
 for i,row in enumerate(manifest):
  c=row['channel'];mission=row['mission'];te=array(z,'data/data/test/'+c+'.npy');n=len(te);ix=np.arange(250,n-10);y=te[ix,0]
  for role in ['gcn','lstm']:
   for seed in [17,29,43]:
    job=f'{role}__{c}__{seed}';pred=array(exports,'units/'+job+'/scores.npz')['prediction'][:-10]
    try:
     d=native_detect(c,te,pred);nseg=verify_original_loop(c,te,pred,d)
    except Exception as exc:
     failures.append({'job_id':job,'type':type(exc).__name__,'error':str(exc)});save('NATIVE_FAILURES.json',failures);raise
    parity.append({'job_id':job,'original_process_batches_exact':True,'segments':nseg})
    r,ev,out=metrics.channel(labels[c],ix,d['flags'],d['issue_index'],np.abs(y-pred),d['smoothed'])
    r.update(role=role,channel=c,mission=mission,seed=seed,pipeline='native_v2',MAE=float(np.abs(y-pred).mean()),MSE=float(np.mean((y-pred)**2)))
    r.update(literal_counts(labels[c],ix,d['flags']));rows.append(r)
    notes.extend({**v,'role':role,'channel':c,'mission':mission,'seed':seed,'pipeline':'native_v2'} for v in ev)
    key=job+'__native_v2';np.savez_compressed(OUT/'streams'/f'{key}.npz',target_index=ix,flags=d['flags'],issue_index=d['issue_index'],smoothed=d['smoothed'])
    save('windows/'+key+'.json',d['windows'])
    a=np.load(PRIOR/'outputs/streams'/f'{job}__common.npz');assert np.array_equal(a['target_index'],ix)
    comparison.append({'job_id':job,'mission':mission,'channel':c,'role':role,'seed':seed,
     'different_flag_points':int(np.sum(a['flags']!=d['flags'])),'different_issue_points':int(np.sum(a['issue_index']!=d['issue_index'])),
     'different_EWMA_points':int(np.sum(a['smoothed']!=d['smoothed'])),
     'native_warnings':' | '.join(d['warnings'])})
    if c in alternate:
     rr,_,_=metrics.channel(alternate[c],ix,d['flags'],d['issue_index'],np.abs(y-pred),d['smoothed'])
     rr.update(role=role,channel=c,mission=mission,seed=seed,pipeline='native_v2');sens.append(rr)
  print(f'[{i+1}/81] verified literal native detector on frozen project forecasts: {c}',flush=True)
 write_csv('channel_metrics.csv',rows);write_csv('new_event_notifications.csv',notes);write_csv('flag_and_issue_differences.csv',comparison)
 save('NATIVE_FAILURES.json',failures);save('original_process_batches_parity.json',parity)
 groups={}
 for r in rows:groups.setdefault((r['mission'],r['role'],r['pipeline']),[]).append(r)
 seed_rows=[];summary=[];sensitivity=[]
 for (mission,role,pipeline),rr in groups.items():
  ss=[]
  for seed in sorted(set(v['seed'] for v in rr)):
   sub=[v for v in rr if v['seed']==seed];v=metrics.aggregate(sub)
   h=sum(x['literal_v2_H'] for x in sub);m=sum(x['literal_v2_M'] for x in sub);u=sum(x['literal_v2_U'] for x in sub)
   p=h/(h+u) if h+u else 0.;rec=h/(h+m) if h+m else float('nan')
   v.update(mission=mission,role=role,pipeline=pipeline,seed=seed,
    literal_v2_H=h,literal_v2_M=m,literal_v2_U=u,literal_v2_event_f0_5=metrics.fbeta(p,rec))
   seed_rows.append(v);ss.append(v)
   if mission=='MSL':
    v=metrics.aggregate([r for r in sub if r['channel']!='C-2']);v.update(mission=mission,role=role,pipeline=pipeline,seed=seed,sensitivity='exclude_C-2');sensitivity.append(v)
   if mission=='SMAP':
    replacement=next((v for v in sens if v['role']==role and v['seed']==seed and v['pipeline']==pipeline),None)
    if replacement is None:
     prev=pd.read_csv(PRIOR/'outputs/sensitivity_verified_seed_metrics.csv');replacement=prev[(prev.role==role)&(prev.seed==seed)&(prev.pipeline==pipeline)&(prev.sensitivity=='second_P-2_label')].iloc[0].to_dict()
     # The prior sensitivity CSV contains already aggregated mission rows; reuse it directly.
     replacement.update(mission=mission,role=role,pipeline=pipeline,seed=seed,sensitivity='second_P-2_label');sensitivity.append(replacement)
    else:
     v=metrics.aggregate([v for v in sub if v['channel']!='P-2']+[replacement]);v.update(mission=mission,role=role,pipeline=pipeline,seed=seed,sensitivity='second_P-2_label');sensitivity.append(v)
  v={k:float(np.mean([r[k] for r in ss])) for k in ss[0] if isinstance(ss[0][k],(int,float,np.generic)) and k!='seed'}
  v.update(mission=mission,role=role,pipeline=pipeline,seeds=len(ss));summary.append(v)
 write_csv('mission_seed_metrics.csv',seed_rows);write_csv('mission_metrics.csv',summary);write_csv('sensitivity_seed_metrics.csv',sensitivity)
 contrasts=[]
 for mission in ['MSL','SMAP']:
  channels=sorted(v['channel'] for v in groups[(mission,'released_lstm','common')])
  def tensor(role,pipeline,keys):
   ss=[0] if role=='released_lstm' else [17,29,43]
   return [[[next(v for v in groups[(mission,role,pipeline)] if v['channel']==c and v['seed']==s)[k] for k in keys] for s in ss] for c in channels]
  for role in ['gcn','lstm']:
   for metric,keys in [('corrected_event_f0_5',['H','M','U','FPt','N0']),('point_f1',['TP','FP','FN'])]:
    contrasts.append({**bootstrap(tensor(role,'native_v2',keys),tensor('released_lstm','native_v2',keys),metric),
      'mission':mission,'contrast':role+'_minus_released_lstm','pipeline':'native_v2'})
  for role in ['gcn','lstm','released_lstm']:
   metric='corrected_event_f0_5';keys=['H','M','U','FPt','N0']
   contrasts.append({**bootstrap(tensor(role,'native_v2',keys),tensor(role,'common',keys),metric),
     'mission':mission,'contrast':'native_minus_adapted','role':role})
 save('paired_channel_bootstrap.json',contrasts)
 save('ANALYSIS_COMPLETE.json',{'analysis_id':freeze['analysis_id'],'new_detector_streams':len(parity),'new_neural_fits':0,'new_neural_inference':0,
  'reused_common_streams':567,'reused_release_native_streams':81,'total_table_streams':len(rows),
  'native_original_control_flow_exact_passes':len(parity),'native_failures':failures,
  'prior_package_unchanged':all(sha(PRIOR/p)==h for p,h in old_manifest['files'].items())})
 print(pd.DataFrame(summary)[['mission','role','pipeline','corrected_event_f0_5','hundman_overlap_f0_5','literal_v2_event_f0_5','point_f1','nominal_occupancy','within_event_notification_rate']].to_string(index=False))
if __name__=='__main__':main()
