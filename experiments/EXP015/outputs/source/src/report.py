"""Exact frozen NASA evaluator; unchanged historical rows; descriptive contrasts."""
import os,shutil,uuid
from pathlib import Path
import numpy as np
import pandas as pd
from integrity import ROOT,read,sha,write,atomic,seal,manifest,blob
from report_common import mission_tables,count_array,auxiliary_bootstrap,write_csv
from vendor.frozen import metrics

LABELS={'graph_residual':'EXP013 GCN + residual/Hundman','removed_residual':'EXP013 removal + residual/Hundman',
        'fusion_residual':'EXP013 fusion + residual/Hundman','exp012_gcn':'EXP012 GCN + Hundman',
        'exp012_lstm':'EXP012 controlled LSTM + Hundman','exp014_gdn':'EXP014 GDN + Hundman',
        'graph_native':'EXP013 GCN + original pipeline','removed_native':'EXP013 removal + original pipeline',
        'fusion_native':'EXP013 fusion + original pipeline'}

def delayed_rows(rows,events):
    delays={}
    for r in events:
        if pd.notna(r.get('delay')):delays.setdefault((r['mission'],r['arm'],r['channel'],int(r['seed'])),[]).append(float(r['delay']))
    for r in rows:r['_delays']=delays.get((r['mission'],r['arm'],r['channel'],int(r['seed'])),[])
    return rows

def historical(root):
    root=Path(root);common=pd.read_csv(root/'inputs/exp014_reports/common_detector_channels.csv').to_dict('records')
    er=pd.read_csv(root/'inputs/exp014_reports/common_detector_event_notifications.csv').to_dict('records')
    mapping={'gcn':'exp012_gcn','lstm':'exp012_lstm','gdn':'exp014_gdn'}
    for r in common:
        r['arm']=mapping[r['arm']];r.update(panel='common_detector',source_kind='reused_frozen_result')
    for r in er:r['arm']=mapping[r['arm']]
    native=pd.read_csv(root/'inputs/exp013_reports/channel_metrics.csv')
    native=native[native['role'].isin(['graph','removed','fusion'])].to_dict('records')
    ner=pd.read_csv(root/'inputs/exp013_reports/event_notifications.csv')
    ner=ner[ner['role'].isin(['graph','removed','fusion'])].to_dict('records')
    for r in native:r.update(arm=r['role']+'_native',panel='original_pipeline',detector='EXP013_frozen_EWMA_quantile',source_experiment='EXP013',source_kind='reused_frozen_result')
    for r in ner:r.update(arm=r['role']+'_native',source_experiment='EXP013',detector='EXP013_frozen_EWMA_quantile')
    return delayed_rows(common,er),delayed_rows(native,ner),er+ner

def contrasts(root,records):
    cfg=read(Path(root)/'config.json');roster=read(Path(root)/'inputs/TEST_ROSTER.json');out=[]
    pairs=[('graph_residual','removed_residual','graph_pathway'),('graph_residual','fusion_residual','graph_pathway'),
           ('graph_residual','graph_native','pipeline_change'),('removed_residual','removed_native','pipeline_change'),
           ('fusion_residual','fusion_native','pipeline_change'),('graph_residual','exp012_gcn','historical_context'),
           ('graph_residual','exp012_lstm','historical_context'),('graph_residual','exp014_gdn','historical_context')]
    for mission in ['MSL','SMAP']:
        channels=sorted(ch for ch,r in roster.items() if r['mission']==mission)
        for arm,control,scope in pairs:
            a=count_array(records,mission,arm,channels);b=count_array(records,mission,control,channels)
            event=metrics.bootstrap(a[:,:,:5],b[:,:,:5],cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])
            aux=auxiliary_bootstrap(a,b,cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])
            out.append({'mission':mission,'contrast':arm+'_minus_'+control,'arm':arm,'control':control,'scope':scope,
                        'seed_counts':[a.shape[1],b.shape[1]],'architecture_only':False,
                        'multiplicity':'exploratory descriptive 95%; no multiplicity adjustment',
                        'interpretation':'conditional on frozen fits; grouped seeds, paired channels; shared-model dependence and refitting uncertainty not resolved',
                        **event,'auxiliary_count_metrics':aux})
    return out

def regression_historical(root,means):
    root=Path(root);exp014=pd.read_csv(root/'inputs/exp014_reports/mission_metrics.csv')
    exp013=pd.read_csv(root/'inputs/exp013_reports/mission_metrics.csv');checks=[]
    for r in means:
        if r['arm'].endswith('_residual'):continue
        if r['arm'].endswith('_native'):
            role=r['arm'].replace('_native','');old=exp013[(exp013['mission']==r['mission'])&(exp013['role']==role)].iloc[0]
        else:
            arm={'exp012_gcn':'gcn','exp012_lstm':'lstm','exp014_gdn':'gdn'}[r['arm']]
            old=exp014[(exp014['mission']==r['mission'])&(exp014['arm']==arm)&(exp014['panel']=='common_detector')].iloc[0]
        for key in ['corrected_event_f0_5','point_f1','nominal_occupancy','alarm_starts_per_1000_nominal']:
            if not np.isclose(r[key],old[key],atol=2e-12,rtol=0):raise RuntimeError('Historical count aggregation changed: '+r['mission']+'/'+r['arm']+'/'+key)
            checks.append({'mission':r['mission'],'arm':r['arm'],'metric':key,'absolute_difference':float(abs(r[key]-old[key]))})
    return checks

def generate(root,work,identity,events,label_audit,fixture=False):
    root=Path(root);work=Path(work);final=work/'reports'
    if final.exists():
        m=manifest(final,'COMPLETE.json')
        if m['score_seal_sha256']!=sha(work/'SCORE_SEAL.json') or m['run_identity']!=identity:raise RuntimeError('Report identity differs')
        return read(final/'SUMMARY.json')
    cfg=read(root/'config.json');new=[];ers=[];outside=[]
    for ch,row in read(root/'inputs/TEST_ROSTER.json').items():
        for role in cfg['roles']:
            for seed in cfg['seeds']:
                jid=f"{row['mission']}__{role}_residual_hundman__{seed}__{ch}";base=work/'units'/jid
                a=__import__('integrity').arrays(base/'scores.npz')
                result,er,outs=metrics.channel(events[ch],a['target_index'],a['adaptive_flags'],a['adaptive_issue_index'],a['scores_raw'],a['scores_ewma'])
                meta={'job_id':jid,'arm':role+'_residual','role':role,'channel':ch,'mission':row['mission'],'seed':seed,
                      'panel':'common_detector','detector':'HUNDMAN_ADAPT_V1','source_experiment':'EXP015_frozen_EXP013',
                      'source_kind':'new_frozen_checkpoint_reference_inference'}
                new.append({**meta,**result,'_delays':[v['delay'] for v in er if v['delay'] is not None]})
                ers.extend({**meta,'event_number':i,**v} for i,v in enumerate(er));outside.extend({**meta,**v} for v in outs)
    common,native,old_events=historical(root);all_rows=new+common+native;ers+=old_events
    per_seed,means=mission_tables(all_rows);checks=regression_historical(root,means)
    stats=contrasts(root,all_rows)
    tmp=work/('.tmp-reports-'+uuid.uuid4().hex);tmp.mkdir()
    try:
        write_csv(tmp/'new_channel_metrics.csv',new);write_csv(tmp/'all_comparison_channels.csv',all_rows)
        write_csv(tmp/'mission_seed_metrics.csv',per_seed);write_csv(tmp/'mission_metrics.csv',means)
        write_csv(tmp/'event_notifications.csv',ers);write_csv(tmp/'missed_events.csv',[r for r in ers if not bool(r['overlap_hit'])],list(pd.DataFrame(ers).columns))
        write_csv(tmp/'events_outside_support.csv',outside,['arm','channel','mission','seed','source_experiment','detector','onset','end_exclusive','evaluation_support'])
        write(tmp/'paired_channel_bootstrap.json',stats);write(tmp/'LABEL_ROSTER_AUDIT.json',label_audit)
        write_csv(tmp/'historical_aggregation_regression.csv',checks)
        timing=[{k:r[k] for k in ['mission','arm','seed','H','M','events_fully_outside_support','within_event_notification_rate','late_notifications','missed_notifications','notification_delay_median','notification_delay_defined_events']} for r in per_seed]
        write_csv(tmp/'timing_summary.csv',timing)
        flat=[]
        for r in stats:
            flat.append({'mission':r['mission'],'contrast':r['contrast'],'metric':'corrected_event_f0_5',
                         'difference':r['difference'],'CI_low':r['percentile_CI95'][0] if r['percentile_CI95'] else None,
                         'CI_high':r['percentile_CI95'][1] if r['percentile_CI95'] else None,'undefined_replicates':r['undefined_replicates']})
            for k,v in r['auxiliary_count_metrics'].items():flat.append({'mission':r['mission'],'contrast':r['contrast'],'metric':k,'difference':v['difference'],'CI_low':v['percentile_CI95'][0] if v['percentile_CI95'] else None,'CI_high':v['percentile_CI95'][1] if v['percentile_CI95'] else None,'undefined_replicates':v['undefined_replicates']})
        write_csv(tmp/'paired_contrasts.csv',flat)
        by={(r['mission'],r['arm']):r for r in means};findings=[]
        for m in ['MSL','SMAP']:
            g=by[(m,'graph_residual')];rem=by[(m,'removed_residual')];fusion=by[(m,'fusion_residual')]
            change={role:by[(m,role+'_residual')]['corrected_event_f0_5']-by[(m,role+'_native')]['corrected_event_f0_5'] for role in cfg['roles']}
            findings.append({'mission':m,'pipeline_corrected_event_changes':change,
                'graph_minus_removed_corrected_event':g['corrected_event_f0_5']-rem['corrected_event_f0_5'],
                'graph_minus_fusion_corrected_event':g['corrected_event_f0_5']-fusion['corrected_event_f0_5'],
                'removal_pipeline_improvement_at_least_graph':change['removed']>=change['graph'],
                'removal_residual_result_at_least_graph':rem['corrected_event_f0_5']>=g['corrected_event_f0_5'],
                'interpretation':'Descriptive changes in score representation and detector jointly; do not isolate thresholding from representation or establish causal primacy.'})
        summary={'status':'CPU_SYNTHETIC_FIXTURE_COMPLETE' if fixture else 'COMPLETE_RETROSPECTIVE_EXP015_SUPPLEMENT',
                 'real_NASA_comparison':not fixture,'new_scientific_fits':0,'new_residual_streams':len(new),
                 'original_results_overwritten':False,'TEST_label_selection':False,'point_adjustment':False,
                 'historical_TEST_exposure':True,'inherited_TEST_extrema_scaling':True,'discriminator_hybrid_executed':False,
                 'findings':findings,'mission_metrics':means,'score_seal_sha256':sha(work/'SCORE_SEAL.json'),
                 'limitations':['command-only reference differs from telemetry-history forecasting; removal is entity-only',
                     'frozen weights were selected for synthetic classification plus clean-reference objectives, not this detector',
                     'pipeline change replaces score representation and detector together; causal primacy cannot be isolated',
                     'adaptive unlabelled TEST thresholds, startup lookahead and batched issuance are retained',
                     'three graph/control seeds versus one GDN seed; channel bootstrap is conditional and ignores shared-model dependence and refitting',
                     'EXP013 auxiliary synthetic seed-export defect remains in original evidence and is not used in this NASA analysis']}
        write(tmp/'SUMMARY.json',summary)
        lines=['# '+('CPU synthetic fixture — no NASA findings' if fixture else 'EXP015 frozen EXP013 residual/Hundman supplement'),'',
               'All neural weights and previous results remain frozen. This analysis changes the anomaly evidence to the command-reference residual and applies the exact EXP012 detector. It is retrospective and exploratory.','',
               '| Mission | Method | Corrected event F0.5 | Event P | Event R | Point P | Point R | Point F1 | Nominal occupancy | Starts/1000 nominal | Seeds |',
               '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for m in ['MSL','SMAP']:
            for arm in LABELS:
                r=by[(m,arm)];nums=[r[k] for k in ['corrected_event_f0_5','event_precision','event_recall','point_precision','point_recall','point_f1','nominal_occupancy','alarm_starts_per_1000_nominal']]
                lines.append('|'+m+'|'+LABELS[arm]+'|'+'|'.join('undefined' if v is None or not np.isfinite(v) else f'{v:.6f}' for v in nums)+f"|{r['seed_records']}|")
        lines+=['','## What the graph comparison shows','']
        for f in findings:
            m=f['mission'];lines.append(f"{m}: graph-minus-removal corrected event F0.5 = {f['graph_minus_removed_corrected_event']:+.6f}; graph-minus-fusion = {f['graph_minus_fusion_corrected_event']:+.6f}.")
            if f['removal_residual_result_at_least_graph']:lines.append('The graph-removal control matches or exceeds the graph on corrected event F0.5; this outcome does not establish graph-specific benefit.')
            if f['removal_pipeline_improvement_at_least_graph']:lines.append("The removal control's change relative to its own original pipeline is at least as favourable as the graph's. Any observed pipeline improvement therefore cannot be attributed to graph messages alone.")
            changes=', '.join(f"{r} {d:+.6f}" for r,d in f['pipeline_corrected_event_changes'].items());lines.append('Changes versus each frozen original pipeline: '+changes+'.')
        lines+=['','Positive deltas establish sensitivity to this joint score/detector substitution within the frozen retrospective study. Poor or unchanged performance means this specific substitution did not rescue the model; it does not identify a failure mechanism. Substantial graph value requires examining both controls, point discrimination, alarm burden and the descriptive paired intervals.','',
                'See `paired_contrasts.csv` and `paired_channel_bootstrap.json` for 95% descriptive intervals, including all seeds grouped and undefined resamples. See `mission_seed_metrics.csv`, `timing_summary.csv`, `event_notifications.csv` and `missed_events.csv` for actual issuance, delays and misses. A label-overlap hit may be issued after event end.','',
                'The GCN reference uses past commands only, while GDN and EXP012 forecast from telemetry history. These are complete pipeline comparisons, not a perfectly controlled architecture ablation. Original-unit means released NASA telemetry units, whose TEST-extrema preprocessing is inherited. No physical leakage isolation, independent generalisation, equivalence, SOTA or operational performance claim follows.','',
                'The discriminator hybrid was excluded before scoring: signed logits lack the telemetry scale and observed-target variation required by the exact Hundman guards. No score transformation or TEST-selected calibration was added.']
        atomic(tmp/'REPORT_READY_SUMMARY.md',('\n'.join(lines)+'\n').encode())
        latex=[]
        for r in means:latex.append(f"{r['mission']} & {LABELS[r['arm']]} & {r['corrected_event_f0_5']:.6f} & {r['point_f1']:.6f} & {r['nominal_occupancy']:.6f} & {r['alarm_starts_per_1000_nominal']:.6f} "+r'\\')
        atomic(tmp/'table_rows.tex',('\n'.join(latex)+'\n').encode())
        build_figures(tmp,by,fixture)
        seal(tmp,'COMPLETE.json',{'run_identity':identity,'score_seal_sha256':sha(work/'SCORE_SEAL.json'),'scientific_status':summary['status']})
        manifest(tmp,'COMPLETE.json');os.replace(tmp,final);return summary
    finally:
        if tmp.exists():shutil.rmtree(tmp)

def build_figures(out,by,fixture=False):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    arms=['graph_residual','removed_residual','fusion_residual','exp012_gcn','exp012_lstm','exp014_gdn']
    short=['EXP013 GCN','EXP013 removal','EXP013 fusion','EXP012 GCN','EXP012 LSTM','EXP014 GDN']
    colours=['#176b9b','#8a9ba5','#e58c35','#6489bc','#459b73','#a06bac']
    for name,keys,labels in [('detector_comparison',['corrected_event_f0_5','point_f1'],['Corrected event F0.5','Strict point F1']),
                             ('alarm_burden',['nominal_occupancy','alarm_starts_per_1000_nominal'],['Nominal alarm occupancy','Alarm starts / 1000 nominal samples'])]:
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for j,m in enumerate(['MSL','SMAP']):
            for i,(key,label) in enumerate(zip(keys,labels)):
                ax=axes[i,j];values=[by[(m,a)][key] for a in arms]
                ax.bar(short,values,color=colours);ax.set_title(m);ax.set_ylabel(label)
                ax.tick_params(axis='x',labelrotation=28,labelsize=8);ax.spines[['top','right']].set_visible(False)
                if key in ['corrected_event_f0_5','point_f1','nominal_occupancy']:ax.set_ylim(0,1)
                for k,v in enumerate(values):ax.annotate(f'{v:.3f}',(k,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
        fig.suptitle('CPU SYNTHETIC FIXTURE — NO NASA RESULTS' if fixture else 'Frozen models with HUNDMAN_ADAPT_V1 · retrospective supplement')
        fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    x=np.arange(3)
    for ax,m in zip(axes,['MSL','SMAP']):
        ax.bar(x-.18,[by[(m,r+'_native')]['corrected_event_f0_5'] for r in ['graph','removed','fusion']],width=.36,label='Frozen original pipeline',color='#aeb8bd')
        ax.bar(x+.18,[by[(m,r+'_residual')]['corrected_event_f0_5'] for r in ['graph','removed','fusion']],width=.36,label='Reference residual + Hundman',color='#176b9b')
        ax.set_xticks(x,['GCN','Removal','Fusion']);ax.set_title(m);ax.set_ylim(0,1);ax.set_ylabel('Corrected event F0.5');ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8,loc='upper left');fig.suptitle('CPU SYNTHETIC FIXTURE — NO NASA RESULTS' if fixture else 'EXP013 pipeline substitution · all three frozen arms')
    fig.savefig(out/'pipeline_substitution.png',dpi=180);fig.savefig(out/'pipeline_substitution.pdf');plt.close(fig)
