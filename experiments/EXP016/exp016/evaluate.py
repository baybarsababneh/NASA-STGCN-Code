import csv,io
from pathlib import Path
import numpy as np
from .io import ROOT,save_json,read_json,atomic,sha,committed
from .metrics import labels,counts,aggregate

def table(path,rows):
    if not rows:return
    f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows);atomic(path,f.getvalue().encode())

def evaluate(out,c,roster,label_path=None,fixture=False):
    # This is the only production phase permitted to parse annotation intervals.
    seal=read_json(out/'SCORE_SEAL.json')
    for name,h in seal['files'].items():
        if sha(out/name)!=h:raise RuntimeError('Score seal invalid '+name)
    primary,alternative=labels(label_path or ROOT/'inputs/labeled_anomalies.csv',fixture=fixture)
    rows=[];events=[];sens=[];failures=[];degenerate=[]
    for meta in roster:
        ch=meta['channel']
        if ch not in primary:continue
        if primary[ch]['num_values']!=meta['test_rows']:raise ValueError('Label length mismatch '+ch)
        for role,seed in [('released_lstm',0)]+[('stgcn',s) for s in c['seeds']]:
            for mode in ('uniform','native_C2'):
                folder=out/'scores'/f'{role}_{seed}_{ch}_{mode}'
                if not committed(folder):raise RuntimeError('Missing score commit '+str(folder))
                receipt=read_json(folder/'receipt.json')
                if receipt.get('failure'):
                    failures.append(dict(channel=ch,role=role,seed=seed,mode=mode,**receipt));continue
                degenerate.extend(dict(channel=ch,role=role,seed=seed,mode=mode,**d) for d in receipt.get('degeneracies',[]))
                z=np.load(folder/'stream.npz');ev=primary[ch]['events']
                count,notes=counts(ev,z['flags'],z['target_index'],z['issue_index'],z['raw_error'],z['smoothed'])
                meta2=dict(channel=ch,mission=meta['mission'],role=role,seed=seed,detector=mode)
                rows.append(dict(**meta2,**count,MAE=float(np.mean(z['raw_error'])),MSE=float(np.mean(z['raw_error']**2))))
                events.extend(dict(**meta2,**n) for n in notes)
                if ch=='P-2':
                    other,_=counts(alternative[ch],z['flags'],z['target_index'],z['issue_index'],z['raw_error'],z['smoothed'])
                    sens.append(dict(**meta2,**other))
    metrics=[]
    for mission in ('MSL','SMAP'):
        expected=sum(r['mission']==mission and r['channel'] in primary for r in roster)
        for role,seeds in [('released_lstm',[0]),('stgcn',c['seeds'])]:
            for seed in seeds:
                for mode in ('uniform','native_C2'):
                    rr=[r for r in rows if (r['mission'],r['role'],r['seed'],r['detector'])==(mission,role,seed,mode)]
                    if len(rr)!=expected:continue # no silently reduced-roster score
                    for sensitivity in ('primary_labels','exclude_C2','alternative_P2'):
                        data=[r for r in rr if not(sensitivity=='exclude_C2' and r['channel']=='C-2')]
                        if sensitivity=='alternative_P2':
                            data=[next((a for a in sens if (a['channel'],a['role'],a['seed'],a['detector'])==(r['channel'],role,seed,mode)),r) if r['channel']=='P-2' else r for r in data]
                        metrics.append(dict(mission=mission,role=role,seed=seed,detector=mode,sensitivity=sensitivity,channels=len(data),**aggregate(data)))
    table(out/'channel_metrics.csv',rows);table(out/'mission_seed_metrics.csv',metrics);table(out/'event_notifications.csv',events)
    save_json(out/'DETECTOR_FAILURES.json',failures);save_json(out/'DEGENERATE_BRANCHES.json',degenerate)
    # Average complete seed metrics, never pool identical author forecasts three times.
    means=[]
    for mission in ('MSL','SMAP'):
        for mode in ('uniform','native_C2'):
            for role,num in [('stgcn',len(c['seeds'])),('released_lstm',1)]:
                rr=[r for r in metrics if r['mission']==mission and r['detector']==mode and r['role']==role and r['sensitivity']=='primary_labels']
                if len(rr)!=num:continue
                m=dict(mission=mission,detector=mode,role=role,fits_per_channel=num)
                for k in ('event_precision','event_recall','event_F05','point_F1','point_F05','nominal_occupancy','timely_event_rate','raw_AP_macro','smoothed_AP_macro'):
                    vals=[r[k] for r in rr if r[k] is not None];m[k]=float(np.mean(vals)) if len(vals)==num else None
                means.append(m)
    table(out/'mission_mean_metrics.csv',means)
    boot=bootstrap(rows,c);save_json(out/'paired_bootstrap.json',boot)
    lines=['# EXP016 completed evaluation','',
      'Retrospective comparison with one author-released LSTM fit per channel. No new LSTM was trained.',
      'Primary: uniform author-derived detector; native_C2 retains the historical label-informed exception.',
      'Event F0.5 uses the standard formula and inclusive any-overlap events. No point adjustment.',
      'Test-extrema preprocessing, historical TEST exposure, different validation split and unknown original fit seeds prevent a pure architecture attribution.','',
      '| Mission | Detector | Role | Event F0.5 | Strict point F1 | Nominal occupancy |',
      '|---|---|---|---:|---:|---:|']
    for r in means:
        fmt=lambda x:'undefined' if x is None else f'{x:.6f}'
        lines.append('| '+' | '.join([r['mission'],r['detector'],r['role'],fmt(r['event_F05']),fmt(r['point_F1']),fmt(r['nominal_occupancy'])])+' |')
    lines+=['','Read mission_seed_metrics.csv for every seed and P-2/C-2 sensitivities; event_notifications.csv records actual availability, including detector lookahead.',
       'Bootstrap intervals condition on available fits and channel resampling; shared-incident dependence and LSTM fit variance are unresolved.',
       'DETECTOR_FAILURES.json must be empty for a complete native comparison. Inspect DEGENERATE_BRANCHES.json for primary branch abstentions. Runtime and validation gates do not establish scientific superiority.']
    atomic(out/'RESULTS.md','\n'.join(lines).encode())
    return dict(labelled_channels=len(primary),training_channels=len(roster),score_rows=len(rows),detector_failures=len(failures),primary_degenerate_branches=len(degenerate),mission_rows=len(means))

def bootstrap(rows,c):
    result=[];rng=np.random.default_rng(c['bootstrap_seed']);keys=('event_tp','event_fp','event_fn','point_tp','point_fp','point_fn','nominal_points')
    def score(x,endpoint):
        if endpoint=='event_F05':a,b,d=x[...,0],x[...,1],x[...,2];den=1.25*a+b+.25*d;num=1.25*a
        elif endpoint=='point_F1':a,b,d=x[...,3],x[...,4],x[...,5];den=2*a+b+d;num=2*a
        else:num=x[...,4];den=x[...,6]
        return np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den!=0)
    for mission in ('MSL','SMAP'):
        for mode in ('uniform','native_C2'):
            lookup={(r['channel'],r['role'],r['seed']):r for r in rows if r['mission']==mission and r['detector']==mode}
            ch=sorted(k[0] for k in lookup if k[1:] == ('released_lstm',0))
            expected_ch=sorted({r['channel'] for r in rows if r['mission']==mission and r['detector']=='uniform' and r['role']=='released_lstm'})
            if ch!=expected_ch or not ch or any((x,'stgcn',s) not in lookup for x in ch for s in c['seeds']):continue
            g=np.array([[[lookup[x,'stgcn',s][k] for k in keys] for x in ch] for s in c['seeds']],float)
            l=np.array([[lookup[x,'released_lstm',0][k] for k in keys] for x in ch],float)
            # Chunk to limit peak memory; same channel draw for both roles and all seeds.
            draws={k:[] for k in ('event_F05','point_F1','nominal_occupancy')}
            for start in range(0,c['bootstrap_repetitions'],250):
                ix=rng.integers(0,len(ch),size=(min(250,c['bootstrap_repetitions']-start),len(ch)))
                gg=g[:,ix,:].sum(2);ll=l[ix].sum(1)
                for metric in draws:draws[metric].extend((score(gg,metric).mean(0)-score(ll,metric)).tolist())
            for metric,values in draws.items():
                v=np.array(values);ok=v[np.isfinite(v)]
                result.append(dict(mission=mission,detector=mode,metric=metric,contrast='STGCN mean seeds minus fixed released LSTM',
                    difference=(lambda x:float(x) if np.isfinite(x) else None)(score(g.sum(1),metric).mean()-score(l.sum(0),metric)),
                    ci95=np.quantile(ok,[.025,.975]).tolist() if len(ok) else None,
                    undefined_replicates=int(len(v)-len(ok)),replicates=len(v),channels=len(ch),conditional=True))
    return result
