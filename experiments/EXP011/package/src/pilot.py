"""Bounded synthetic-supervision pilot. Reads EXP010 export only; never loads a forecaster."""
import argparse, hashlib, io, itertools, json, os, platform, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(17,29,43)
ARMS=('residual','fusion','removed','persistence')
THRESHOLDS=(0.,2.,4.,6.,8.)
MECH={'MSL':('C-1','P-10','M-7','P-14','T-13','D-16'),'SMAP':('P-3','A-8','G-6','E-1','G-4','A-5')}
FAMILIES=('spike','offset32','ramp32')

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(b):return hashlib.sha256(b).hexdigest()
def filehash(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def atomic(p,b):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_bytes(b);os.replace(t,p)
def save(p,x):atomic(p,canonical(x))
def rolling(x,w,fn):return np.array([fn(x[max(0,i-w+1):i+1]) for i in range(len(x))])

def features(a,arm):
 y=a['target'].astype(float);p=a['persistence'].astype(float) if arm=='persistence' else a['prediction'].astype(float)
 e=y-p
 # Causal, fixed fit-standardised input units. No estimated tiny residual scale.
 cols=[abs(e),abs(rolling(e,8,np.mean)),rolling(abs(e),8,np.max)]
 if arm!='residual':
  dy=y-a['persistence'];dp=np.r_[0.,np.diff(p)];lag=np.r_[np.repeat(y[0],8),y[:-8]] if len(y)>=8 else np.repeat(y[0],len(y))
  cols.extend([abs(dy),abs(dp),abs(y-lag),rolling(y,8,np.ptp)])
 x=np.arcsinh(np.column_stack(cols))
 if arm!='residual':x=np.column_stack([x,rolling((dy==0).astype(float),8,np.mean)])
 assert np.isfinite(x).all()
 return np.column_stack([np.ones(len(y)),x])

def load_export(path):
 expected=json.loads((ROOT/'input.json').read_text())
 assert filehash(path)==expected['exp010_sha256'],'Wrong EXP010 input bytes'
 plan=json.loads((ROOT/'parent_plan.json').read_text());pm={p['job_id']:p for p in plan}
 data={};count=0
 with zipfile.ZipFile(path) as z:
  assert len(z.namelist())==len(set(z.namelist())),'Duplicate archive entries'
  manifest=json.loads(z.read('OUTPUT_MANIFEST.json'))
  for name,h in manifest.items():assert digest(z.read(name))==h,name
  freeze=json.loads(z.read('RUN_FREEZE.json'));rh=digest(canonical(freeze))
  # Parent used the same sorted compact canonical JSON hash.
  for name in sorted(n for n in z.namelist() if n.startswith('units/') and n.endswith('.done.json')):
   m=json.loads(z.read(name));ident=m['identity'];assert ident['runhash']==rh
   uid=digest(canonical(ident))[:24];assert name==f'units/{uid}.done.json'
   b=z.read(f'units/{uid}.zip');assert digest(b)==m['sha256'];p=pm[ident['parent_job']];j=p['job'];key=(j['mission'],j['channel'],j['seed'],j['role'])
   with zipfile.ZipFile(io.BytesIO(b)) as u:
    r=json.loads(u.read('record.json'))
    with np.load(io.BytesIO(u.read('arrays.npz')),allow_pickle=False) as a:arr={k:a[k].copy() for k in a.files}
   assert all(np.isfinite(v).all() for v in arr.values())
   slot=data.setdefault(key,{'n':p['reference_result']['num_values'],'cases':[]})
   if r['kind']=='nominal':
    assert 'nominal' not in slot
    assert all(v['exact_match'] and v['repeat_prediction_exact'] for v in r['parent_replay']['metrics'].values())
    slot['nominal']=arr
   else:
    assert r['parent_archive_sha256']==p['archive_sha256'];assert r['case']['id']==ident['case'];slot['cases'].append((r['case'],arr))
   count+=1
  close=json.loads(z.read('STUDY_CLOSE.json'));assert count==3078 and close['units']==3078 and not close['detector_selected']
 assert len(data)==114
 for d in data.values():assert 'nominal' in d and len(d['cases'])==26 and len({c['id'] for c,a in d['cases']})==26
 return data

def prepare(data):
 out={}
 for (mission,ch,seed,role),d in data.items():
  arms=('removed',) if role=='message_removed' else ('residual','fusion','persistence')
  for arm in arms:
   a=d['nominal'];idx=a['target_index'];warm=idx>=int(.6*d['n'])+32;assessment=idx>=int(.7*d['n'])
   x=features(a,arm);cases=[]
   for c,b in d['cases']:
    assert np.array_equal(idx,b['target_index'])
    mask=(idx>=c['onset'])&(idx<c['end']);nonnull=bool(np.any(b['target'][mask]!=a['target'][mask]))
    cases.append(dict(case=c,x=features(b,arm),mask=mask,nonnull=nonnull))
   out[(mission,ch,seed,arm)]=dict(x=x,warm=warm,assessment=assessment,cases=cases)
 return out

def dataset(prepared,mission,channels,seed,arm):
 xs=[];ys=[];ws=[]
 for ch in channels:
  d=prepared[(mission,ch,seed,arm)];n=d['warm'].sum();xs.append(d['x'][d['warm']]);ys.append(np.zeros(n));ws.append(np.full(n,.5/len(channels)/n))
  # Train only amplitude-controlled spike/offset/ramp, both amplitudes. Stuck is family-OOD audit.
  cases=[c for c in d['cases'] if c['case']['family'] in FAMILIES and c['nonnull']]
  assert len(cases)==24
  for c in cases:
   n=c['mask'].sum();xs.append(c['x'][c['mask']]);ys.append(np.ones(n));ws.append(np.full(n,.5/len(channels)/len(cases)/n))
 return np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)

def fit(x,y,w):
 def objective(b):
  s=x@b;pen=b.copy();pen[0]=0
  return float(np.sum(w*(np.logaddexp(0,s)-y*s))+.005*np.dot(pen,pen)),x.T@(w*(expit(s)-y))+.01*pen
 r=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8,'maxls':40})
 assert r.success and np.isfinite(r.x).all(),str(r.message)
 return r.x,dict(iterations=int(r.nit),objective=float(r.fun),gradient_max=float(np.max(np.abs(r.jac))))

def evaluate(d,b,h):
 nom=d['x']@b;flag=nom>h;mask=d['assessment'];v=flag[mask];starts=v & ~np.r_[False,v[:-1]]
 out=dict(occupancy=float(v.mean()),starts=float(starts.sum()*1000/len(v)),samples=int(len(v)),alarm_samples=int(v.sum()),alarm_starts=int(starts.sum()))
 events=[]
 for c in d['cases']:
  if not c['nonnull']:continue
  f=c['x']@b>h;m=c['mask'];hit=bool(f[m].any());bg=bool(flag[m].any());duration=int(m.sum());inds=np.flatnonzero(f[m])
  events.append(dict(family=c['case']['family'],amplitude=c['case']['amplitude'],case_id=c['case']['id'],hit=hit,background_hit=bg,incremental=bool(hit and not bg),delay=int(inds[0]) if len(inds) else duration,duration=duration))
 out['null_cases']=sum(not c['nonnull'] for c in d['cases']);out['events']=events
 return out

def summarise(rows):
 # Equal channel weights after averaging seeds; seeds are not independent telemetry.
 d=pd.DataFrame([{k:r[k] for k in ('channel','seed','occupancy','starts')} for r in rows]);ch=d.groupby('channel')[['occupancy','starts']].mean();sd=d.groupby('seed')[['occupancy','starts']].mean()
 fam={}
 for f in FAMILIES:
  a=[]
  for r in rows:
   ev=[e['incremental'] for e in r['events'] if e['family']==f and e['amplitude']==2.]
   assert len(ev)==4
   a.append(dict(channel=r['channel'],value=float(np.mean(ev))))
  fam[f]=float(pd.DataFrame(a).groupby('channel').value.mean().mean())
 nominal=bool(ch.occupancy.mean()<=.01 and ch.starts.mean()<=2 and ch.occupancy.quantile(.9)<=.05 and ch.starts.quantile(.9)<=10 and (sd.occupancy<=.01).all() and (sd.starts<=2).all())
 return dict(mean_occupancy=float(ch.occupancy.mean()),mean_starts=float(ch.starts.mean()),p90_occupancy=float(ch.occupancy.quantile(.9)),p90_starts=float(ch.starts.quantile(.9)),family_incremental=fam,worst_family=min(fam.values()),nominal_pass=nominal,feasible=bool(nominal and min(fam.values())>=.5),channels=len(ch))

def choose(allrows):
 eligible=[]
 for h,rows in allrows.items():
  s=summarise(rows)
  if s['feasible']:eligible.append((s['worst_family'],np.mean(list(s['family_incremental'].values())),-s['mean_starts'],-s['mean_occupancy'],h))
 return max(eligible)[-1] if eligible else None

def breakdown(rows):
 result=[]
 for r in rows:
  for family in (*FAMILIES,'stuck32'):
   for amplitude in ((None,) if family=='stuck32' else (.5,2.)):
    ev=[e for e in r['events'] if e['family']==family and e['amplitude']==amplitude]
    clean=[e for e in ev if not e['background_hit']]
    result.append(dict(**{k:r[k] for k in ('mission','channel','seed','arm','cohort','abstain')},family=family,amplitude=amplitude,nonnull_events=len(ev),unconditional_hit=float(np.mean([e['hit'] for e in ev])) if ev else None,background_hit=float(np.mean([e['background_hit'] for e in ev])) if ev else None,incremental_hit=float(np.mean([e['incremental'] for e in ev])) if ev else None,clean_events=len(clean),conditional_clean_hit=float(np.mean([e['hit'] for e in clean])) if clean else None))
 return result

def bootstrap(rows):
 rng=np.random.default_rng(11011);results=[]
 for mission,roster in MECH.items():
  for control in ('residual','removed','persistence'):
   values={}
   for arm in ('fusion',control):
    rr=[r for r in rows if r['mission']==mission and r['arm']==arm and r['cohort']=='mechanical']
    a=[]
    for ch in roster:
     cr=[r for r in rr if r['channel']==ch]
     hit=np.mean([[np.mean([e['incremental'] for e in r['events'] if e['family']==f and e['amplitude']==2.]) for f in FAMILIES] for r in cr],axis=0)
     a.append([np.mean([r['occupancy'] for r in cr]),np.mean([r['starts'] for r in cr]),*hit])
    values[arm]=np.array(a)
   ix=rng.integers(0,len(roster),(10000,len(roster)))
   f=values['fusion'][ix].mean(axis=1);c=values[control][ix].mean(axis=1)
   stats={'occupancy':f[:,0]-c[:,0],'starts_per_1000':f[:,1]-c[:,1],'worst_family_incremental':f[:,2:].min(axis=1)-c[:,2:].min(axis=1)}
   results.append(dict(mission=mission,contrast='fusion-minus-'+control,bootstrap_repetitions=10000,channels=6,intervals={k:np.quantile(v,[.025,.5,.975]).tolist() for k,v in stats.items()},warning='Descriptive percentile intervals, only six mechanically chosen channels; abstentions retained. Not population inference.'))
 return results

def run(args):
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 # Exclusive ownership: no unsafe auto-stealing on a slow/shared filesystem.
 lock=out/'ACTIVE_RUN.json'
 if lock.exists():raise RuntimeError('ACTIVE_RUN exists. Verify prior worker stopped; use --release-lock ONLY after runtime termination.')
 fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,canonical({'pid':os.getpid(),'host':platform.node(),'started':time.time()}));os.close(fd)
 try:
  manifest=json.loads((ROOT/'MANIFEST.json').read_text())
  for p,h in manifest.items():assert filehash(ROOT/p)==h,p
  env=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,platform=platform.platform(),threads={k:os.getenv(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
  freeze=dict(experiment='EXP011_FUSION_PILOT_V1',manifest_sha256=filehash(ROOT/'MANIFEST.json'),input_sha256=filehash(args.input),environment=env)
  frozen=out/'RUN_FREEZE.json'
  if frozen.exists():assert json.loads(frozen.read_text())==freeze,'Run environment/source/input changed'
  else:save(frozen,freeze)
  rh=digest(canonical(freeze));data=prepare(load_export(args.input));models={};done=0
  for mission,roster in MECH.items():
   sets=sorted(set(tuple(sorted(x)) for size in (4,5,6) for x in itertools.combinations(roster,size)))
   for channels in sets:
    for seed in SEEDS:
     for arm in ARMS:
      key=(mission,channels,seed,arm)
      if arm=='persistence' and seed!=17:
       models[key]=models[(mission,channels,17,arm)];continue
      identity=dict(runhash=rh,mission=mission,channels=channels,seed=seed,arm=arm);jid=digest(canonical(identity))[:24];p=out/'fits'/f'{jid}.json'
      if p.exists():
       r=json.loads(p.read_text());assert r['identity']==json.loads(canonical(identity));assert digest(canonical(r['payload']))==r['payload_sha256'];b=np.array(r['payload']['coefficients']);assert np.isfinite(b).all()
      else:
       start=time.time();x,y,w=dataset(data,mission,channels,seed,arm);b,diag=fit(x,y,w);payload=dict(coefficients=b.tolist(),diagnostics=diag,elapsed_seconds=time.time()-start);save(p,dict(identity=identity,payload=payload,payload_sha256=digest(canonical(payload))))
      models[key]=b;done+=1
      if done%12==0:save(out/'STATUS.json',dict(status='RUNNING',detector_fits=done,max_detector_fits=440));print(f'{done}/440 detector fits',flush=True)
  assert done==440 and len(list((out/'fits').glob('*.json')))==440
  rows=[];thresholds=[];decisions={}
  for mission,roster in MECH.items():
   for arm in ARMS:
    armrows=[];missing=0
    for held in roster:
     train=tuple(sorted(set(roster)-{held}));inner={h:[] for h in THRESHOLDS}
     for innerheld in train:
      innertrain=tuple(sorted(set(train)-{innerheld}))
      for seed in SEEDS:
       b=models[(mission,innertrain,seed,arm)]
       for h in THRESHOLDS:inner[h].append(dict(channel=innerheld,seed=seed,**evaluate(data[(mission,innerheld,seed,arm)],b,h)))
     h=choose(inner);missing+=h is None
     thresholds.append(dict(mission=mission,arm=arm,held_channel=held,threshold=h,status='SELECTED' if h is not None else 'NO_FEASIBLE_INNER_THRESHOLD',inner_summaries={str(k):summarise(v) for k,v in inner.items()}))
     # Abstention is explicit. Do not replace missing calibration with a least-bad threshold.
     for seed in SEEDS:
      r=dict(mission=mission,channel=held,seed=seed,arm=arm,cohort='mechanical',threshold=h,abstain=h is None,**evaluate(data[(mission,held,seed,arm)],models[(mission,train,seed,arm)],float('inf') if h is None else h));armrows.append(r);rows.append(r)
    s=summarise(armrows);s['infeasible_inner_folds']=missing;s['pilot_pass']=bool(s['feasible'] and missing==0);decisions[f'{mission}/{arm}']=s
    # Posthoc enriched channels are stress audits only. Threshold uses mechanical crossfit predictions.
    full=tuple(sorted(roster));cal={h:[] for h in THRESHOLDS}
    for held in roster:
     train=tuple(sorted(set(roster)-{held}))
     for seed in SEEDS:
      for h in THRESHOLDS:cal[h].append(dict(channel=held,seed=seed,**evaluate(data[(mission,held,seed,arm)],models[(mission,train,seed,arm)],h)))
    h=choose(cal);thresholds.append(dict(mission=mission,arm=arm,held_channel='POSTHOC_COHORT',threshold=h,status='SELECTED' if h is not None else 'NO_FEASIBLE_INNER_THRESHOLD'))
    extra=sorted({k[1] for k in data if k[0]==mission}-set(roster))
    for held in extra:
     for seed in SEEDS:rows.append(dict(mission=mission,channel=held,seed=seed,arm=arm,cohort='posthoc_mechanism',threshold=h,abstain=h is None,**evaluate(data[(mission,held,seed,arm)],models[(mission,full,seed,arm)],float('inf') if h is None else h)))
  save(out/'sensitivity_breakdown.json',breakdown(rows));save(out/'paired_channel_bootstrap.json',bootstrap(rows));save(out/'threshold_selection.json',thresholds);save(out/'channel_seed_results.json',rows);save(out/'arm_decisions.json',decisions)
  pd.DataFrame([{k:v for k,v in r.items() if k!='events'} for r in rows]).to_csv(out/'nominal_results.csv',index=False)
  pd.DataFrame([dict(**{k:r[k] for k in ('mission','channel','seed','arm','cohort','abstain')},**e) for r in rows for e in r['events']]).to_csv(out/'synthetic_events.csv',index=False)
  close=dict(status='COMPLETE_PILOT_NO_DEPLOYMENT_DETECTOR',detector_fits=440,forecasting_fits=0,NASA_TEST_accessed=False,final_calibration_accessed=False,missions={m:'ELIGIBLE_FOR_TRAIN_ONLY_CONFIRMATION' if decisions[f'{m}/fusion']['pilot_pass'] else 'STOP_REQUIRE_NEW_INFORMATION' for m in MECH},warning='Synthetic, posthoc-designed pilot on reused development channels; no real-anomaly performance or operational false-alarm guarantee.')
  save(out/'STUDY_CLOSE.json',close);save(out/'STATUS.json',close)
  files={str(p.relative_to(out)):filehash(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('ACTIVE_RUN.json','OUTPUT_MANIFEST.json')};save(out/'OUTPUT_MANIFEST.json',files)
  print(json.dumps(close,indent=2),flush=True)
 finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--release-lock',action='store_true');a=p.parse_args()
 if a.release_lock:
  q=Path(a.output)/'ACTIVE_RUN.json';assert q.exists();atomic(q.with_name('released_lock_'+str(time.time_ns())+'.json'),q.read_bytes());q.unlink()
 run(a)
