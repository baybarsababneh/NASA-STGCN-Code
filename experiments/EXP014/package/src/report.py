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

def check_historical_rows(root,events):
    """Recompute counts/timing from byte-preserved flags, without fitting/detection."""
    root=Path(root)
    frozen=pd.read_csv(root/'inputs/exp012_reports/channel_metrics.csv')
    frozen=frozen[(frozen['detector']=='hundman_adapt_v1')&frozen['arm'].isin(['gcn','lstm'])]
    lookup={v['job_id']:v for v in frozen.to_dict('records')}
    records=[];event_rows=[];outside=[]
    for j in read(root/'inputs/exp012_scores/SUBSET.json')['roster']:
        base=root/'inputs/exp012_scores/units'/j;m=read(base/'record.json');a=arrays(base/'scores.npz')
        expected=np.arange(250,m['original_TEST_rows'],dtype=np.int64)
        if not np.array_equal(expected,a['target_index']): raise RuntimeError('EXP012 score support differs')
        r,er,outs=metrics.channel(events[m['channel']],a['target_index'],a['adaptive_flags'],a['adaptive_issue_index'],a['scores_raw'],a['scores_ewma'])
        original=lookup[j]
        for k,v in r.items():
            old=original[k]
            if isinstance(v,(int,bool,np.integer)):
                if old!=v: raise RuntimeError('Frozen EXP012 count changed: '+j+' '+k)
            elif np.isfinite(v) and (not np.isfinite(old) or not np.isclose(v,old,atol=2e-12,rtol=0)):
                raise RuntimeError('Frozen EXP012 arithmetic differs: '+j+' '+k)
        # Keep the published row values, rather than overwrite them with a replay.
        row={**original,'panel':'common_detector','arm':m['arm'],'source_experiment':'EXP012','source_kind':'reused_frozen_result','detector':'hundman_adapt_v1','_delays':[e['delay'] for e in er if e['delay'] is not None]}
        records.append(row)
        common={'arm':m['arm'],'channel':m['channel'],'mission':m['mission'],'seed':m['seed'],'source_experiment':'EXP012','detector':'hundman_adapt_v1'}
        event_rows.extend({**common,'event_number':i,**e} for i,e in enumerate(er))
        outside.extend({**common,**e} for e in outs)
    if len(records)!=len(events)*6: raise RuntimeError('Historical model roster incomplete')
    return records,event_rows,outside

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

def contrasts(root,panels):
    cfg=read(Path(root)/'config.json');roster=read(Path(root)/'inputs/TEST_ROSTER.json');out=[]
    for panel,records in panels.items():
        for mission in ['MSL','SMAP']:
            channels=sorted(ch for ch,r in roster.items() if r['mission']==mission)
            controls=['lstm','gdn'] if panel=='common_detector' else ['gdn','catch']
            for control in controls:
                a=count_array(records,mission,'gcn',channels);b=count_array(records,mission,control,channels)
                event=metrics.bootstrap(a[:,:,:5],b[:,:,:5],cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])
                aux=auxiliary_bootstrap(a,b,cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])
                out.append({'mission':mission,'panel':panel,'contrast':'gcn_minus_'+control,'interpretation':'descriptive retrospective, conditional on saved fits; shared-model/channel dependence and unequal initialisations; no TEST selection, equivalence or independent superiority claim','multiplicity':'exploratory 95%; no multiplicity adjustment','architecture_only':False,'seed_counts':[a.shape[1],b.shape[1]],**event,'auxiliary_count_metrics':aux})
    return out

def write_csv(path,rows,columns=None):
    rows=[{k:v for k,v in r.items() if not k.startswith('_')} for r in rows]
    frame=pd.DataFrame(rows,columns=columns)
    atomic(path,frame.to_csv(index=False,float_format='%.17g').encode())

def generate(root,work,run_identity,events,label_audit,fixture=False):
    root=Path(root);work=Path(work);final=work/'reports'
    if final.exists():
        m=manifest(final,'COMPLETE.json')
        if m['score_seal_sha256']!=sha(work/'SCORE_SEAL.json') or m['run_identity']!=run_identity: raise RuntimeError('Report identity differs')
        return read(final/'SUMMARY.json')
    old,ers,outs=check_historical_rows(root,events);new=[]
    for ch,r in read(root/'inputs/TEST_ROSTER.json').items():
        p=work/'units'/f"{r['mission']}__gdn_hundman__17__{ch}"
        a=arrays(p/'scores.npz');record=read(p/'record.json')
        result,er,outside=metrics.channel(events[ch],a['target_index'],a['adaptive_flags'],a['adaptive_issue_index'],a['scores_raw'],a['scores_ewma'])
        base={'job_id':p.name,'arm':'gdn','channel':ch,'mission':r['mission'],'seed':17,'panel':'common_detector','detector':'hundman_adapt_v1','source_experiment':'EXP013_checkpoint_EXP014_detector','source_kind':'new_frozen_checkpoint_inference'}
        new.append({**base,**result,'_delays':[v['delay'] for v in er if v['delay'] is not None]})
        ers.extend({**base,'event_number':i,**v} for i,v in enumerate(er));outs.extend({**base,**v} for v in outside)
    common=old+new
    native=pd.read_csv(root/'inputs/exp013_reports/channel_metrics.csv')
    native=native[native['role'].isin(['gdn','catch'])].to_dict('records')
    native_events=pd.read_csv(root/'inputs/exp013_reports/event_notifications.csv')
    delays={}
    for r in native_events.to_dict('records'):
        if r['role'] not in ['gdn','catch']: continue
        key=(r['mission'],r['role'],r['channel'],int(r['seed']))
        if pd.notna(r['delay']): delays.setdefault(key,[]).append(float(r['delay']))
    pipeline=[{**v,'panel':'complete_pipeline'} for v in old]
    for r in native:
        r.update(arm=r['role'],panel='complete_pipeline',detector='EXP013_TRAIN_selected_EWMA_quantile',source_experiment='EXP013',source_kind='reused_frozen_result')
        r['_delays']=delays.get((r['mission'],r['role'],r['channel'],int(r['seed'])),[])
        pipeline.append(r)
    per_seed=[];means=[]
    for group in [common,pipeline]:
        a,b=mission_tables(group);per_seed+=a;means+=b
    # Frozen EXP012 mission numbers are a regression, never an optimisation target.
    historical=pd.read_csv(root/'inputs/exp012_reports/mission_metrics.csv')
    for v in means:
        if v['panel']=='common_detector' and v['arm'] in ['gcn','lstm']:
            row=historical[(historical['mission']==v['mission'])&(historical['arm']==v['arm'])&(historical['detector']=='hundman_adapt_v1')].iloc[0]
            for key in ['corrected_event_f0_5','point_f1','nominal_occupancy','alarm_starts_per_1000_nominal']:
                if not np.isclose(v[key],row[key],atol=2e-12,rtol=0): raise RuntimeError('Frozen historical mission result differs: '+key)
    stats=contrasts(root,{'common_detector':common,'complete_pipeline':pipeline})
    tmp=work/('.tmp-reports-'+uuid.uuid4().hex);tmp.mkdir()
    try:
        write_csv(tmp/'common_detector_channels.csv',common);write_csv(tmp/'complete_pipeline_channels.csv',pipeline)
        write_csv(tmp/'mission_seed_metrics.csv',per_seed);write_csv(tmp/'mission_metrics.csv',means)
        write_csv(tmp/'common_detector_event_notifications.csv',ers)
        write_csv(tmp/'events_outside_support.csv',outs,['arm','channel','mission','seed','source_experiment','detector','onset','end_exclusive','evaluation_support'])
        write_csv(tmp/'native_pipeline_event_notifications_frozen.csv',native_events[native_events['role'].isin(['gdn','catch'])].to_dict('records'))
        write(tmp/'paired_channel_bootstrap.json',stats);write(tmp/'LABEL_ROSTER_AUDIT.json',label_audit)
        summary={'status':'CPU_SYNTHETIC_FIXTURE_COMPLETE' if fixture else 'COMPLETE_RETROSPECTIVE_COMPARISON','real_NASA_comparison':not fixture,'new_scientific_fits':0,'new_GDN_residual_streams':len(new),'historical_EXP012_streams_reused':len(old),'native_EXP013_result_rows_reused':len(native),'original_results_overwritten':False,'model_architecture_or_precision_changed':False,'TEST_label_selection':False,'point_adjustment':False,'historical_TEST_exposure':True,'inherited_TEST_extrema_preprocessing':True,'architecture_only_comparison':False,'new_comparisons':means,'score_seal_sha256':sha(work/'SCORE_SEAL.json'),'limitations':['GDN lookback5/all within-file dimensions/shared mission fit versus selected context/lookback64/per-channel EXP012 models','Three GCN/LSTM initialisations versus one GDN/CATCH initialisation','Paired-channel bootstrap ignores refitting/seed variation and assumes channels as resampling units despite shared models/commands','Hundman adapts to unlabelled TEST residuals and has noncausal startup averaging and batch issuance; exact historical protocol retained','Complete-pipeline native score adaptations retain the different EXP013 TRAIN-selected detector','STGLR fidelity exclusion and OmniAnomaly environment exclusion are not efficacy findings','EXP013 auxiliary synthetic seed-reporting defect is preserved and is not imported into these NASA comparisons']}
        write(tmp/'SUMMARY.json',summary)
        table=['# '+('CPU synthetic fixture — no NASA results' if fixture else 'EXP014 final bounded literature comparison'),'','No new training, model selection, detector tuning or replacement of EXP012/EXP013 results occurred.','', '| Panel | Mission | Model | Fc0.5 | Strict point F1 | Nominal occupancy | Nominal starts/1000 | Seeds |','|---|---|---|---:|---:|---:|---:|---:|']
        for v in means:
            table.append(f"|{v['panel']}|{v['mission']}|{v['arm']}|{v['corrected_event_f0_5']:.6f}|{v['point_f1']:.6f}|{v['nominal_occupancy']:.6f}|{v['alarm_starts_per_1000_nominal']:.6f}|{v['seed_records']}|")
        table+=['','Common detector uses the exact frozen EXP012 HUNDMAN_ADAPT_V1 on original-unit telemetry forecast residuals. GDN is an adapted forecaster, not a perfectly controlled architecture ablation. Complete pipelines retain their frozen native-score detector adaptations. Paper-specific numbers remain in the original report and literature decision document; they are not ranked in this table.','','All intervals are descriptive 95% paired-channel intervals conditional on the saved fits. No independent generalisation, equivalence or deployment claim follows. Notification timing credits the earliest actual issued event-overlapping flag, and may be after the event. Misses and late notifications are retained.','','See mission_seed_metrics.csv for event/point precision and recall, exact count denominators and per-seed notification medians. See paired_channel_bootstrap.json for event, point and alarm contrasts. Historical values and frozen native results are reused unchanged.']
        atomic(tmp/'REPORT_READY_SUMMARY.md','\n'.join(table).encode())
        latex=[]
        for v in means:
            label=v['panel'].replace('_',r'\_')
            latex.append(f"{label} & {v['mission']} & {v['arm']} & {v['corrected_event_f0_5']:.6f} & {v['point_f1']:.6f} & {v['nominal_occupancy']:.6f} & {v['seed_records']} "+r'\\')
        atomic(tmp/'table_rows.tex',('\n'.join(latex)+'\n').encode())
        build_figure(tmp,means,fixture)
        seal(tmp,'COMPLETE.json',{'run_identity':run_identity,'score_seal_sha256':sha(work/'SCORE_SEAL.json'),'scientific_status':summary['status']})
        manifest(tmp,'COMPLETE.json');os.replace(tmp,final)
        return summary
    finally:
        if tmp.exists(): shutil.rmtree(tmp)

def build_figure(out,means,fixture=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(9,6.5),constrained_layout=True)
    colours={'gcn':'#2c6fbb','lstm':'#659447','gdn':'#c67b37','catch':'#8661ac'}
    for i,panel in enumerate(['common_detector','complete_pipeline']):
        for j,mission in enumerate(['MSL','SMAP']):
            ax=axes[i,j];rows=[r for r in means if r['panel']==panel and r['mission']==mission]
            ax.bar([r['arm'].upper() for r in rows],[r['corrected_event_f0_5'] for r in rows],color=[colours[r['arm']] for r in rows])
            ax.set_ylim(0,1);ax.set_ylabel('Corrected event F0.5');ax.set_title(mission+' · '+panel.replace('_',' '),pad=12);ax.spines[['top','right']].set_visible(False)
            for k,r in enumerate(rows):
                height=r['corrected_event_f0_5'];inside=height>.90
                ax.text(k,height-.025 if inside else height+.018,f"{height:.3f}",ha='center',va='top' if inside else 'bottom',color='white' if inside else '#222222',fontsize=9)
    fig.suptitle('CPU SYNTHETIC FIXTURE — NO NASA RESULTS' if fixture else 'Frozen NASA adaptations · retrospective comparisons')
    fig.savefig(out/'NASA_comparisons.png',dpi=180);fig.savefig(out/'NASA_comparisons.pdf');plt.close(fig)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description='Verify completed exports and regenerate display assets only')
    ap.add_argument('--work',type=Path,required=True);a=ap.parse_args()
    manifest(a.work/'reports','COMPLETE.json')
    print('Report exports and their manifest verified. Frozen outputs are never overwritten.')
