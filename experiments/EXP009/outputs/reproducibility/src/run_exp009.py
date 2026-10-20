"""Frozen residual-only study. The only data input is the allowlisted EXP008 export."""
import argparse,hashlib,io,json,os,platform,sys,time,zipfile,fcntl,shutil,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from detectors import calibrate,apply,burden,stress,robust_scale

ROOT=Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def dumps(x):return json.dumps(x,sort_keys=True,indent=2,allow_nan=False).encode()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(dumps(value));os.replace(tmp,p)
def freeze(p,value):
    if p.exists():assert json.loads(p.read_text())==value,f'Frozen record changed: {p}'
    else:write(p,value)

def load_parent(path,cfg):
    b=path.read_bytes();assert sha(b)==cfg['parent_sha256'],'Wrong EXP008 export'
    z=zipfile.ZipFile(io.BytesIO(b));assert len(z.namelist())==len(set(z.namelist()))
    status=json.loads(z.read('STATUS.json'));assert status['run_hash']==cfg['parent_run_hash'] and status['committed']==534
    close=json.loads(z.read('STUDY_CLOSE.json'));assert not close['NASA_TEST_accessed'] and not close['anomaly_labels_accessed'] and not close['final_calibration_accessed']
    plan=json.loads(z.read('PLAN.json'));assert len(plan)==534 and len({p['job_id']for p in plan})==534
    streams=[]
    for p in plan:
        jid=p['job_id'];b=z.read('replays/'+jid+'.zip');m=json.loads(z.read('replays/'+jid+'.done.json'))
        assert m==dict(job_id=jid,run_hash=cfg['parent_run_hash'],parent_archive_sha256=p['archive_sha256'],archive_sha256=sha(b))
        zz=zipfile.ZipFile(io.BytesIO(b));r=json.loads(zz.read('record.json'));assert r['job']==p['job'] and r['parent_archive_sha256']==p['archive_sha256']
        assert all(v['exact_match'] and v['repeat_prediction_exact'] and v['mse']==v['reference_mse'] for v in r['metrics'].values())
        with np.load(io.BytesIO(zz.read('predictions.npz')),allow_pickle=False) as a:a={k:a[k].copy()for k in a.files}
        assert set(a)=={'target_index','target','prediction','persistence'}
        assert all(np.isfinite(v).all()for v in a.values());assert np.array_equal(a['target_index'],np.arange(r['target_start'],r['target_end_exclusive']))
        assert a['target_index'][-1]<int(np.floor(.8*p['reference_result']['num_values']))
        streams.append((p,a))
    # Common additive disturbances require identical target units and target sequences.
    grouped={}
    for p,a in streams:
        key=(p['job']['channel'],p['job']['fold']);r=p['reference_result']
        signature=(r['target_mean'],r['target_scale'])
        if key in grouped:
            sig,base=grouped[key];assert sig==signature
            for k in ('target_index','target','persistence'):assert np.array_equal(a[k],base[k]),'Control target alignment mismatch'
        else:grouped[key]=(signature,a)
    return streams

def masks(p,a,stage):
    t=a['target_index'];r=p['reference_result'];n=r['num_values']
    if p['job']['fold']=='canonical':
        b=int(np.floor(.7*n));c=int(np.floor(.75*n));ref=(t>=r['fit_end'])&(t<b)
        assess=(t>=b)&(t<c) if stage=='selection' else (t>=c)&(t<r['stop_end'])
    else:
        b=int(np.floor((.55 if p['job']['fold']=='R1' else .65)*n));ref=(t>=r['stop_end'])&(t<b);assess=(t>=b)&(t<r['assess_end'])
    assert ref.sum()>=16 and assess.sum()>=8
    return ref,assess

def candidates(cfg):return [dict(id=f'{f}_h{h}',family=f,h=h)for f in cfg['families']for h in cfg['multipliers']]
BASELINES=[dict(id='raw_q99',family='historical_raw99'),dict(id='ewma10_q995',family='historical_ewma995')]

def evaluate(p,a,model,candidate,stage,injection_scale):
    ref,assess=masks(p,a,stage);error=a['target'].astype(float)-a['persistence' if model=='persistence' else 'prediction'].astype(float)
    end=int(np.flatnonzero(assess)[-1])+1
    error=error[:end];ref=ref[:end];assess=assess[:end]
    cal=calibrate(error,ref,candidate);s,f=apply(error,cal)
    probes=stress(error,assess,cal,injection_scale,audit=stage!='selection')
    for probe in probes:probe['onset_target_index']=int(a['target_index'][probe['onset']])
    meta=dict(mission=p['job']['mission'],channel=p['job']['channel'],seed=0 if model=='persistence' else p['job']['seed'],fold=p['job']['fold'],model=model,candidate=candidate['id'],stage=stage)
    nom=dict(**meta,**burden(f,assess),reference_samples=int(ref.sum()),reference_start=int(a['target_index'][:end][ref][0]),reference_end=int(a['target_index'][:end][ref][-1]+1),assessment_start=int(a['target_index'][:end][assess][0]),assessment_end=int(a['target_index'][:end][assess][-1]+1),**cal,injection_scale=float(injection_scale),scale_at_floor=bool(cal['scale']<=1e-6))
    return dict(nominal=nom,probes=[dict(**meta,**v)for v in probes])

def table_summary(nom,probes,cfg):
    n=pd.DataFrame(nom);s=pd.DataFrame(probes);rows=[]
    for (mission,candidate),g in n.groupby(['mission','candidate']):
        c=g.groupby('channel')[['occupancy','starts_per_1000']].mean()
        ss=s[(s.mission==mission)&(s.candidate==candidate)]
        f=ss.groupby(['channel','family']).incremental_hit.mean().groupby('family').mean()
        high=ss[ss.amplitude==8].groupby(['channel','family']).incremental_hit.mean().groupby('family').mean()
        seed=g.groupby('seed')[['occupancy','starts_per_1000']].mean()
        nominal_ok=bool(c.occupancy.mean()<=cfg['mean_occupancy_max'] and c.starts_per_1000.mean()<=cfg['mean_starts_per_1000_max'] and c.occupancy.quantile(.9)<=cfg['channel_p90_occupancy_max'] and c.starts_per_1000.quantile(.9)<=cfg['channel_p90_starts_per_1000_max'] and (seed.occupancy<=cfg['mean_occupancy_max']).all() and (seed.starts_per_1000<=cfg['mean_starts_per_1000_max']).all())
        rows.append(dict(mission=mission,candidate=candidate,channels=len(c),mean_occupancy=float(c.occupancy.mean()),mean_starts_per_1000=float(c.starts_per_1000.mean()),p90_occupancy=float(c.occupancy.quantile(.9)),p90_starts_per_1000=float(c.starts_per_1000.quantile(.9)),worst_family_hit=float(f.min()),mean_family_hit=float(f.mean()),high_amplitude_worst_family_hit=float(high.min()),nominal_ok=nominal_ok,sensitivity_ok=bool(high.min()>=cfg['minimum_worst_family_incremental_recall'])))
    return rows

def choose(rows,cs):
    selected={};order={c['id']:i for i,c in enumerate(cs)}
    for mission in ('MSL','SMAP'):
        eligible=[r for r in rows if r['mission']==mission and r['nominal_ok'] and r['sensitivity_ok']]
        eligible.sort(key=lambda r:(-r['worst_family_hit'],-r['mean_family_hit'],r['mean_starts_per_1000'],r['mean_occupancy'],order[r['candidate']]))
        selected[mission]=next(c for c in cs if c['id']==eligible[0]['candidate']) if eligible else None
    return selected

def run(args):
    assert np.__version__=='2.3.5' and pd.__version__=='2.2.3','Use the frozen CPU requirements'
    cfg=json.loads((ROOT/'config.json').read_text());manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for f,h in manifest.items():assert sha((ROOT/f).read_bytes())==h,f'Package changed: {f}'
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    lease=open(out/'RUN.lock','a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,platform=platform.platform())
    identity=dict(config=cfg,manifest_sha256=sha(dumps(manifest)),environment=env)
    freeze(out/'RUN_FREEZE.json',identity);runhash=sha(dumps(identity));start=time.monotonic()
    for f in list(manifest)+['MANIFEST.json']:
        if f.startswith('data/'):continue
        dest=out/'reproducibility'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    (out/'requirements-resolved.txt').write_bytes(subprocess.check_output([sys.executable,'-m','pip','freeze']))
    streams=load_parent(Path(args.input),cfg);write(out/'INPUT_AUDIT.json',dict(passed=True,replays=534,parent_sha256=cfg['parent_sha256']))
    # Common per-channel/origin stress amplitudes: median fixed-GCN reference scale over available seeds.
    scales={}
    for p,a in streams:
        if p['job']['role']!='fixed':continue
        ref,_=masks(p,a,'selection');e=a['target'].astype(float)-a['prediction'].astype(float)
        key=(p['job']['channel'],p['job']['fold']);scales.setdefault(key,[]).append(robust_scale(e[ref])[1])
    scales={k:float(np.median(v))for k,v in scales.items()}
    completed=0
    def unit(p,a,model,cand,stage):
        nonlocal completed
        key=dict(job_id=p['job_id'],model=model,candidate=cand,stage=stage,runhash=runhash)
        uid=sha(dumps(key))[:24];path=out/'units'/f'{uid}.json';done=path.with_suffix('.done.json')
        if done.exists():
            mark=json.loads(done.read_text());assert mark==dict(key=key,sha256=sha(path.read_bytes())),'Resume integrity failure';value=json.loads(path.read_text())
        else:
            value=evaluate(p,a,model,cand,stage,scales[(p['job']['channel'],p['job']['fold'])]);write(path,value);write(done,dict(key=key,sha256=sha(path.read_bytes())))
        completed+=1
        if completed%100==0:print(f'{completed} detector units committed/verified',flush=True)
        return value
    cs=candidates(cfg);sn=[];sp=[]
    for p,a in streams:
        if p['job']['fold']=='canonical' and p['job']['role']=='fixed':
            for c in cs:
                v=unit(p,a,'gcn',c,'selection');sn.append(v['nominal']);sp.extend(v['probes'])
    assert len(sn)==2916
    rank=table_summary(sn,sp,cfg);sel=choose(rank,cs)
    freeze(out/'DETECTOR_CANDIDATE_FREEZE.json',dict(runhash=runhash,selected=sel,decision_basis='canonical TRAIN development only; audit blocks not scored yet'))
    pd.DataFrame(rank).to_csv(out/'selection_ranking.csv',index=False);pd.DataFrame(sn).to_csv(out/'selection_nominal.csv',index=False)
    pd.DataFrame(sp).to_csv(out/'selection_stress.csv',index=False)
    an=[];ap=[]
    for p,a in streams:
        mission=p['job']['mission'];stage='canonical_audit' if p['job']['fold']=='canonical' else 'earlier_origin_audit'
        models=['gcn' if p['job']['role']=='fixed' else 'message_removed']
        if p['job']['role']=='fixed' and p['job']['seed']==17:models.append('persistence')
        for model in models:
            for c in BASELINES+([sel[mission]] if sel[mission] else []):
                v=unit(p,a,model,c,stage);an.append(v['nominal']);ap.extend(v['probes'])
    pd.DataFrame(an).to_csv(out/'audit_nominal.csv',index=False);pd.DataFrame(ap).to_csv(out/'audit_stress.csv',index=False)
    verdict={};gates=[]
    for mission in ('MSL','SMAP'):
        if sel[mission] is None:verdict[mission]='NO_FEASIBLE_CANDIDATE';continue
        good=True
        for stage in ('canonical_audit','earlier_origin_audit'):
            nn=[r for r in an if r['mission']==mission and r['stage']==stage and r['model']=='gcn' and r['candidate']==sel[mission]['id']]
            pp=[r for r in ap if r['mission']==mission and r['stage']==stage and r['model']=='gcn' and r['candidate']==sel[mission]['id'] and r['family']in ('spike','step8','step32','ramp32')]
            r=table_summary(nn,pp,cfg)[0];r['stage']=stage;gates.append(r);good &= r['nominal_ok'] and r['sensitivity_ok']
            if stage=='earlier_origin_audit':
                # Each origin separately, not a favourable average hiding failure.
                for fold in ('R1','R2'):
                    rr=table_summary([x for x in nn if x['fold']==fold],[x for x in pp if x['fold']==fold],cfg)[0];rr['stage']=stage+'_'+fold;gates.append(rr);good &= rr['nominal_ok'] and rr['sensitivity_ok']
        verdict[mission]='PROVISIONAL_FREEZE_REQUIRES_REVIEW' if good else 'AUDIT_FAILED_NO_FALLBACK'
    pd.DataFrame(gates).to_csv(out/'audit_gates.csv',index=False)
    # Channel-level paired descriptive intervals; repeated seeds are averaged first.
    df=pd.DataFrame(an);rng=np.random.default_rng(cfg['bootstrap_seed']);cis=[]
    for mission in ('MSL','SMAP'):
        if not sel[mission]:continue
        for comparator in ('raw_q99','ewma10_q995'):
            d=df[(df.mission==mission)&(df.stage=='canonical_audit')&(df.model=='gcn')]
            for metric in ('occupancy','starts_per_1000'):
                tab=d.groupby(['channel','candidate'])[metric].mean().unstack();delta=(tab[sel[mission]['id']]-tab[comparator]).to_numpy()
                boot=delta[rng.integers(0,len(delta),size=(cfg['bootstrap_replicates'],len(delta)))].mean(axis=1)
                cis.append(dict(mission=mission,comparator=comparator,metric=metric,mean_difference=float(delta.mean()),lower95=float(np.quantile(boot,.025)),upper95=float(np.quantile(boot,.975)),min_leave_one_out=float(min(np.delete(delta,i).mean()for i in range(len(delta)))),max_leave_one_out=float(max(np.delete(delta,i).mean()for i in range(len(delta)))),interpretation='descriptive conditional on development; channels not guaranteed independent'))
    pd.DataFrame(cis).to_csv(out/'paired_descriptive_intervals.csv',index=False)
    close=dict(status='COMPLETE_PROVISIONAL_ONLY',mission_decisions=verdict,units=completed,new_model_fits=0,NASA_TEST_accessed=False,anomaly_labels_accessed=False,final_calibration_accessed=False,elapsed_seconds=time.monotonic()-start,selection_stream_rows=len(sn),audit_stream_rows=len(an))
    write(out/'STUDY_CLOSE.json',close)
    evidence={str(p.relative_to(out)):sha(p.read_bytes())for p in out.rglob('*')if p.is_file()and p.name not in ('OUTPUT_MANIFEST.json','RUN.lock')};write(out/'OUTPUT_MANIFEST.json',evidence)
    print(json.dumps(close,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);run(parser.parse_args())
