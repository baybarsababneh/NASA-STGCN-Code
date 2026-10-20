"""One notebook, explicit freeze boundaries, reusable local stages."""
import argparse,zipfile,io,csv,ast,collections,time,sys
from pathlib import Path
from core import *
from data import stage,training_data,transform
from upstream import prepare,PINS
from train import jobs,train_all,get_model,job_name,NEW_ROLES
from scoring import select_bank,threshold_all,confirm,scores,native,causal_mean,stable_seed
from historical_metrics import channel,aggregate,count_vector,fbeta

def initial(work,device):
    pkg=manifest(ROOT,'PACKAGE_MANIFEST.json');env=configure(device);raw,meta=stage(work)
    identity={'version':VERSION,'package_manifest_sha256':sha(ROOT/'PACKAGE_MANIFEST.json'),'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'data_freeze_sha256':sha(work/'DATA_FREEZE.json'),'environment':env,'historical_label_derived_reports_present_in_inputs':True,'physical_label_isolation_claimed':False}
    freeze(work/'RUN_FREEZE.json',identity);event(work,'STAGE_TRAIN_ONLY',NASA_TEST_float_arrays_opened=False,NASA_labels_opened=False)
    return raw,meta,identity
def verify_run(work,device=None):
    manifest(ROOT,'PACKAGE_MANIFEST.json');r=read(work/'RUN_FREEZE.json')
    checked(ROOT/'PACKAGE_MANIFEST.json',r['package_manifest_sha256']);checked(ROOT/'PROTOCOL.json',r['protocol_sha256']);checked(work/'DATA_FREEZE.json',r['data_freeze_sha256'])
    if device is not None and configure(device)!=r['environment']:raise RuntimeError('Training/scoring environment differs from frozen run. Restore the pinned environment; do not silently refit.')
    return r
def study_files(work):
    names=['RUN_FREEZE.json','DATA_FREEZE.json','ENGINEERING_GATE.json']
    for d in ['models','development','calibration','nominal','official']:
        names.extend(f.relative_to(work).as_posix() for f in (work/d).rglob('*') if f.is_file() and '__pycache__' not in f.parts and '.tmp-' not in f.name)
    return {n:sha(work/n) for n in sorted(names)}
def verify_study(work):
    verify_run(work);f=read(work/'STUDY_FREEZE.json')
    if f['files']!=study_files(work):raise RuntimeError('Sealed study changed; fitting or retuning is forbidden')
    return f
def fit_and_select(work,device,checkpoint_archive=None):
    if (work/'STUDY_FREEZE.json').exists():verify_study(work);print('REUSE SEALED DEVELOPMENT; zero additional fits');return
    if device!='cuda' and not read(ROOT/'PROTOCOL.json').get('fixture_only'):
        raise RuntimeError('Scientific development requires the actual full-batch CUDA engineering gate. CPU is for staging/evaluation and isolated fictional fixtures only.')
    r=verify_run(work,device)
    g=read(work/'ENGINEERING_GATE.json')
    if g['status']!='PASS' or g['environment']!=r['environment'] or g['package_manifest_sha256']!=r['package_manifest_sha256']:raise RuntimeError('Run engineering/runtime validation before training')
    if device=='cuda' and not g.get('full_GPU_validation'):raise RuntimeError('Actual full-batch GPU validation is required before GPU fits')
    raw,meta=training_data(work);paths=prepare(work)
    train_all(work,raw,meta,paths,device,r,on_complete=(lambda:snapshot(work,checkpoint_archive)) if checkpoint_archive else None)
    choices={}
    for mission in ['MSL','SMAP']:
        for role in NEW_ROLES+['gdn','catch']:
            choices[(mission,role)]=select_bank(work,raw,meta,paths,device,mission,role)
    eligible=[];exclusions=[]
    for m,role,s in jobs():
        c=choices[(m,role)]
        if c['status'] not in ['SELECTED_TRAIN_SYNTHETIC','NO_FEASIBLE_CANDIDATE']:
            if role in NEW_ROLES:raise RuntimeError('Primary model cannot be excluded')
            exclusions.append({'mission':m,'role':role,'seed':s,'reason':c});continue
        threshold_all(work,raw,meta,paths,device,m,role,s,c);eligible.append([m,role,s])
    freeze(work/'STUDY_FREEZE.json',{'version':VERSION,'files':study_files(work),'eligible_model_jobs':eligible,'explicit_published_exclusions':exclusions,'NASA_TEST_float_arrays_opened':False,'NASA_labels_opened':False,'threshold_selection_complete':True,'configuration_changes_after_freeze_forbidden':True})
    event(work,'STUDY_SEALED',NASA_TEST_float_arrays_opened=False,NASA_labels_opened=False)
    if checkpoint_archive:snapshot(work,checkpoint_archive)
    print('STUDY SEALED:',len(eligible),'model jobs. No subsequent training or retuning is permitted.')
def confirm_all(work,device):
    f=verify_study(work);verify_run(work,device);raw,meta=training_data(work);paths=prepare(work)
    if (work/'CONFIRMATION_COMPLETE.json').exists():
        for n,h in read(work/'CONFIRMATION_COMPLETE.json')['files'].items():checked(work/n,h)
        print('REUSE CONFIRMATION');return
    for m,r,s in f['eligible_model_jobs']:
        th=read(work/'calibration'/job_name(m,r,s)/'thresholds.json')
        try:confirm(work,raw,meta,paths,device,m,r,s,th)
        except FloatingPointError as ex:
            if r in NEW_ROLES:raise
            out=work/'confirmation'/job_name(m,r,s);write(out/'results.json',{'mission':m,'role':r,'seed':s,'status':'FAILED_PUBLISHED_NATIVE_SCORE','reason':str(ex),'cases':[],'nominal':[],'never_use_to_retune':True});seal(out,'COMPLETE.json')
    files={p.relative_to(work).as_posix():sha(p) for p in (work/'confirmation').rglob('*') if p.is_file()}
    freeze(work/'CONFIRMATION_COMPLETE.json',{'files':files,'study_freeze_sha256':sha(work/'STUDY_FREEZE.json'),'confirmation_is_synthetic_only':True,'no_retuning':True})
    event(work,'SYNTHETIC_CONFIRMATION_COMPLETE',NASA_TEST_float_arrays_opened=False,NASA_labels_opened=False)

def unit_reuse(d,identity):
    if not (d/'COMMIT.json').exists():return False
    c=manifest(d,'COMMIT.json')
    if c['identity']!=identity:raise RuntimeError('Score unit identity changed')
    return True
def verify_export(work):
    verify_study(work);manifest(work/'scores','SCORE_SEAL.json');seal=read(work/'scores'/'SCORE_SEAL.json')
    if seal['study_freeze_sha256']!=sha(work/'STUDY_FREEZE.json') or seal['NASA_labels_opened']:raise RuntimeError('Wrong score-seal identity')
    rows=read(ROOT/'TEST_ROSTER.json');expected={job_name(m,r,s)+'__'+ch for m,r,s in read(work/'STUDY_FREEZE.json')['eligible_model_jobs'] for ch,v in rows.items() if v['mission']==m}
    actual={p.name for p in (work/'scores'/'units').iterdir() if p.is_dir()}
    if expected!=actual or seal['expected_streams']!=len(expected):raise RuntimeError('Incomplete score unit roster')
    for jid in sorted(actual):
        d=work/'scores'/'units'/jid;u=manifest(d,'COMMIT.json');v=read(d/'record.json')
        if v['job_id']!=jid or v['identity']!=u['identity']:raise RuntimeError('Invalid score metadata')
        if jid!=job_name(v['mission'],v['role'],v['seed'])+'__'+v['channel']:raise RuntimeError('Score job metadata differs from roster')
        th=read(work/'calibration'/job_name(v['mission'],v['role'],v['seed'])/'thresholds.json');t=th['channels'][v['channel']]
        identity={'study':sha(work/'STUDY_FREEZE.json'),'model_state_sha256':th['model_state_sha256'],'threshold_record_sha256':digest(t),'TEST_array_sha256':rows[v['channel']]['sha256'],'channel':v['channel'],'mission':v['mission'],'role':v['role'],'seed':v['seed']}
        if v['identity']!=identity or v['threshold']!=t['threshold'] or v['detector_span']!=t['span'] or v['selection_status']!=th['selection_status']:raise RuntimeError('Score threshold/provenance differs from pre-TEST freeze')
        if v['status']=='FINITE_SCORES':
            a=arrays(d/'scores.npz');ix=a['target_index']
            if set(a)!=set(['target_index','native_score','detector_score','flags','issue_index']) or v['samples']!=len(ix):raise RuntimeError('Unexpected score payload schema')
            if not np.array_equal(ix,np.arange(250,rows[v['channel']]['rows'])) or any(a[k].shape!=ix.shape for k in a) or not np.isfinite(a['native_score']).all() or not np.isfinite(a['detector_score']).all():raise RuntimeError('Invalid score support')
            if not np.array_equal(a['flags'],a['detector_score']>v['threshold']) or not np.array_equal(a['issue_index'],np.where(a['flags'],ix,-1)):raise RuntimeError('Changed threshold/decision/issue timing')
            if t['span']==1 and not np.array_equal(a['native_score'],a['detector_score']):raise RuntimeError('Detector span1 score changed')
            if len(ix)>1 and t['span']>1:
                alpha=2/(t['span']+1);expected_smooth=alpha*a['native_score'][1:]+(1-alpha)*a['detector_score'][:-1]
                if not np.array_equal(expected_smooth,a['detector_score'][1:]):raise RuntimeError('Causal detector smoothing changed')
        elif v['role'] in NEW_ROLES or v['status']!='NATIVE_SCORE_FIDELITY_FAILURE' or not v.get('reason'):raise RuntimeError('Invalid failed score stream')
    return seal

def export_test(work,device):
    f=verify_study(work);verify_run(work,device)
    c=read(work/'CONFIRMATION_COMPLETE.json')
    for n,h in c['files'].items():checked(work/n,h)
    if c['study_freeze_sha256']!=sha(work/'STUDY_FREEZE.json'):raise RuntimeError('Confirmation belongs to another study')
    if (work/'scores'/'SCORE_SEAL.json').exists():verify_export(work);print('REUSE SEALED NASA SCORES; no replay');return
    meta=read(work/'DATA_FREEZE.json')['channels'];testmeta=read(ROOT/'TEST_ROSTER.json');paths=prepare(work)
    copyaudit=read(ROOT/'NASA_ARRAY_COPY_AUDIT.json');checked(ROOT/'inputs'/'NASA_arrays_only.zip',copyaudit['new_array_archive_sha256'])
    for m,r,s in f['eligible_model_jobs']:
        model,ck=get_model(work,m,r,s,paths,device);th=read(work/'calibration'/job_name(m,r,s)/'thresholds.json')
        for e,ch in enumerate(ck['channels']):
            t=th['channels'][ch];jid=job_name(m,r,s)+'__'+ch;d=work/'scores'/'units'/jid
            identity={'study':sha(work/'STUDY_FREEZE.json'),'model_state_sha256':th['model_state_sha256'],'threshold_record_sha256':digest(t),'TEST_array_sha256':testmeta[ch]['sha256'],'channel':ch,'mission':m,'role':r,'seed':s}
            if unit_reuse(d,identity):continue
            record={'job_id':jid,'identity':identity,'channel':ch,'mission':m,'role':r,'seed':s,'threshold':t['threshold'],'detector_span':t['span'],'selection_status':th['selection_status'],'historical_TEST_exposure':True,'NASA_labels_opened':False}
            with zipfile.ZipFile(ROOT/'inputs'/'NASA_arrays_only.zip') as z:b=z.read('data/data/test/'+ch+'.npy')
            if hashlib.sha256(b).hexdigest()!=testmeta[ch]['sha256']:raise RuntimeError('TEST payload hash changed')
            a=np.load(io.BytesIO(b),allow_pickle=False)
            if a.shape!=(testmeta[ch]['rows'],meta[ch]['columns']) or not np.isfinite(a).all():raise RuntimeError('Invalid TEST array shape/values')
            event(work,'NASA_TEST_ARRAY_READ',channel=ch,phase='post_freeze_score_export',NASA_labels_opened=False)
            v=a.astype(np.float32);v[:,0]=(a[:,0].astype(np.float64)-meta[ch]['mean'])/meta[ch]['scale'];v=transform({ch:v},ck['scaler'])[ch]
            ix=np.arange(250,len(a),dtype=np.int64)
            try:
                look=64 if r in NEW_ROLES else 5 if r=='gdn' else 191
                if len(ix):
                    n=native(scores(model,r,v,e,look,len(a),device,paths,ck['configuration'],stable_seed((m,r,s,ch,'NASA_TEST'))),r,t['native_stats']);smooth=causal_mean(n,t['span'])
                    n=n[250-look:];smooth=smooth[250-look:];flags=smooth>t['threshold']
                else:n=np.array([],float);smooth=n.copy();flags=np.array([],bool)
                npz(d/'scores.npz',target_index=ix,native_score=n,detector_score=smooth,flags=flags,issue_index=np.where(flags,ix,-1))
                record.update(status='FINITE_SCORES',samples=len(ix))
            except FloatingPointError as ex:
                if r in NEW_ROLES:raise
                record.update(status='NATIVE_SCORE_FIDELITY_FAILURE',reason=str(ex),samples=0)
            write(d/'record.json',record);seal(d,'COMMIT.json',{'identity':identity})
        del model
        if device=='cuda':torch.cuda.empty_cache()
        print('TEST scores committed:',job_name(m,r,s),flush=True)
    out=work/'scores';seal(out,'SCORE_SEAL.json',{'version':VERSION,'study_freeze_sha256':sha(work/'STUDY_FREEZE.json'),'NASA_labels_opened':False,'expected_streams':sum(v['mission']==m for m,r,s in f['eligible_model_jobs'] for v in testmeta.values()),'original_TEST_arrays_byte_identical':True,'historically_exposed_TEST':True})
    verify_export(work);event(work,'NASA_SCORE_SEAL_VERIFIED',NASA_labels_opened=False);print('SCORE SEAL VERIFIED. The exact NASA label CSV may now be uploaded.')

def read_labels(path,rows):
    checked(path,'057ce2d6c8875982bf4e5404aefea14efdcbce413d80826d2b737c95b59b7539')
    with Path(path).open() as f:csvrows=list(csv.DictReader(f))
    seen={};duplicates=[];events={}
    for i,v in enumerate(csvrows):
        ch=v['chan_id']
        if ch in seen:
            if ch!='P-2':raise RuntimeError('Unexpected duplicate label entity')
            duplicates.append({'channel':ch,'kept_row':seen[ch][0],'ignored_row':i});continue
        seen[ch]=(i,v)
    if set(seen)!=set(rows):raise RuntimeError('Label entity roster differs')
    for ch,(i,v) in seen.items():
        if v['spacecraft']!=rows[ch]['mission'] or int(v['num_values'])!=rows[ch]['rows']:raise RuntimeError('Label mission/TEST support mismatch')
        ev=ast.literal_eval(v['anomaly_sequences']);events[ch]=[]
        for a,b in ev:
            if type(a)is not int or type(b)is not int or not 0<=a<=b<rows[ch]['rows']:raise RuntimeError('Invalid label interval')
            events[ch].append((a,b+1))
    if sum(len(v) for v in events.values())!=104:raise RuntimeError('Expected104 original events')
    return events,{'rows':len(csvrows),'entities':len(seen),'original_events':104,'duplicates':duplicates,'inclusive_to_half_open':True}

def table(p,rows):
    from historical_common import table as t
    t(p,rows)
def metrics_mean(records):
    fields=['corrected_precision','event_recall','corrected_event_f0_5','point_precision','point_recall','point_f0_5','point_f1','AP_raw_macro','nominal_occupancy','alarm_starts_per_1000_nominal','within_event_notification_rate']
    result={}
    for k in fields:
        v=[r[k] for r in records];result[k]=float(np.mean(v)) if all(np.isfinite(x) for x in v) else None
        result[k+'_defined_seeds']=sum(np.isfinite(x) for x in v)
    return result
def paired_bootstrap(a,b,confidence=.95):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if a.shape[0]!=b.shape[0] or a.ndim!=3 or b.ndim!=3:raise ValueError('Invalid paired arrays')
    def calc(v):
        h,m,u,fp,n0=np.moveaxis(v[...,:5],-1,0)
        with np.errstate(divide='ignore',invalid='ignore'):
            p=np.where(h+u>0,h/(h+u),np.where(h+m>0,0.,np.nan));r=h/(h+m);pc=p*(1-fp/n0);d=.25*pc+r
            f=np.where(d>0,1.25*pc*r/d,0.);f=np.where(np.isfinite(pc)&np.isfinite(r),f,np.nan)
        return f.mean(-1)
    def point(v):
        fpnom,n0,tp,fp,fn=np.moveaxis(v[...,[3,4,5,6,7]],-1,0)
        with np.errstate(divide='ignore',invalid='ignore'):
            p=np.where(tp+fp>0,tp/(tp+fp),0.);r=tp/(tp+fn)
            f05=np.where(.25*p+r>0,1.25*p*r/(.25*p+r),0.);f1=np.where(p+r>0,2*p*r/(p+r),0.)
            f05=np.where(np.isfinite(r),f05,np.nan);f1=np.where(np.isfinite(r),f1,np.nan)
        return {k:x.mean(-1) for k,x in [('point_precision',p),('point_recall',r),('point_f0_5',f05),('point_f1',f1),('nominal_occupancy',fpnom/n0)]}
    estimate=(calc(a.sum(0)[None])-calc(b.sum(0)[None])).item();gen=np.random.default_rng(20261006);draw=[];extra={}
    if a.shape[-1]>=8 and b.shape[-1]>=8:
        av=point(a.sum(0)[None]);bv=point(b.sum(0)[None]);extra={k:{'difference':float(av[k][0]-bv[k][0]),'draws':[]} for k in av}
    for _ in range(20):
        ids=gen.integers(0,len(a),(500,len(a)));asum=a[ids].sum(1);bsum=b[ids].sum(1);draw.extend((calc(asum)-calc(bsum)).tolist())
        if extra:
            av=point(asum);bv=point(bsum)
            for k in extra:extra[k]['draws'].extend((av[k]-bv[k]).tolist())
    finite=np.array(draw)[np.isfinite(draw)];alpha=1-confidence
    for k,v in extra.items():
        f=np.asarray(v.pop('draws'));f=f[np.isfinite(f)];v.update(confidence=.95,percentile_interval=np.quantile(f,[.025,.975]).tolist() if len(f) else None,undefined_draws=10000-len(f),exploratory_not_multiplicity_adjusted=True)
    return {'difference':estimate,'confidence':confidence,'percentile_interval':np.quantile(finite,[alpha/2,1-alpha/2]).tolist() if len(finite) else None,'draws':10000,'undefined_draws':10000-len(finite),'seed':20261006,'point_and_alarm_contrasts':extra,'dependence_warning':'entities share command context and model training; intervals are descriptive, not independent NASA confirmation'}

def evaluate(work,label_path):
    seal0=verify_export(work);rows=read(ROOT/'TEST_ROSTER.json');out=work/'reports';out.mkdir(exist_ok=True)
    if (out/'COMPLETE.json').exists():manifest(out,'COMPLETE.json');print('REUSE COMPLETED REPORTS');return
    identity={'study':sha(work/'STUDY_FREEZE.json'),'score_seal':sha(work/'scores'/'SCORE_SEAL.json'),'label_sha256':'057ce2d6c8875982bf4e5404aefea14efdcbce413d80826d2b737c95b59b7539','evaluation_environment':{'numpy':np.__version__,'python':platform.python_version()}}
    freeze(out/'EVALUATION_FREEZE.json',identity)
    # verify_export must succeed BEFORE this first semantic label read.
    events,audit=read_labels(label_path,rows);event(work,'NASA_LABELS_READ',after_verified_score_seal=True)
    cr=[];er=[];outside_rows=[];fail=[];allrows={}
    for d in sorted((work/'scores'/'units').iterdir()):
        rec=read(d/'record.json');base={k:rec[k] for k in ['job_id','channel','mission','role','seed']};base['selection_status']=rec['selection_status']
        if rec['status']!='FINITE_SCORES':fail.append(rec);continue
        a=arrays(d/'scores.npz');v,ev,outside=channel(events[rec['channel']],a['target_index'],a['flags'],a['issue_index'],a['native_score'],a['detector_score']);v['_delays']=[x['delay'] for x in ev if x['delay'] is not None]
        cr.append({**base,**{k:x for k,x in v.items() if not k.startswith('_')}});er.extend({**base,'event_number':i,**x} for i,x in enumerate(ev));allrows[(rec['mission'],rec['role'],rec['seed'],rec['channel'])]=v
        outside_rows.extend({**base,'event_number':i,**x} for i,x in enumerate(outside))
    missionseed=[];missions=[]
    for m in ['MSL','SMAP']:
        chans=sorted(ch for ch,v in rows.items() if v['mission']==m)
        for role in NEW_ROLES+['gdn','catch']:
            seeds0=[17,29,43] if role in NEW_ROLES else [17];agg=[]
            for s in seeds0:
                values=[allrows[(m,role,s,ch)] for ch in chans if (m,role,s,ch) in allrows]
                if not values:continue
                v=aggregate(values);missionseed.append({'mission':m,'role':role,'seed':s,'coverage_channels':len(values),'full_roster':len(chans),**v});agg.append(v)
            if agg:missions.append({'mission':m,'role':role,'seed_records':len(agg),'minimum_channel_coverage':min(v['channels'] for v in agg),'full_roster':len(chans),'complete_coverage':all(v['channels']==len(chans) for v in agg),'evidence':'historically_exposed_NASA_RETROSPECTIVE',**metrics_mean(agg)})
    # Historical LSTM rows are reused from the audited final CSV, never recomputed.
    with zipfile.ZipFile(ROOT/'inputs'/'EXP012_RETROSPECTIVE_EVALUATION_REPORTS.zip') as z:
        historical=list(csv.DictReader(io.StringIO(z.read('channel_metrics.csv').decode())))
        original_mission=z.read('mission_metrics.csv');atomic(out/'EXP012_original_mission_metrics.csv',original_mission)
    comparisons=[]
    for m in ['MSL','SMAP']:
        chans=sorted(ch for ch,v in rows.items() if v['mission']==m)
        for control in ['removed','fusion','gdn','catch','EXP012_lstm']:
            aa=[];bb=[];used=[]
            for ch in chans:
                if not all((m,'graph',s,ch) in allrows for s in [17,29,43]):continue
                if control=='EXP012_lstm':
                    hs=[v for v in historical if v['mission']==m and v['arm']=='lstm' and v['channel']==ch and v['detector']=='hundman_adapt_v1']
                    hs=sorted(hs,key=lambda v:int(v['seed']))
                    if len(hs)!=3:continue
                    bv=[[int(v[k]) for k in ['H','M','U','FPt','N0','TP','FP','FN']] for v in hs]
                else:
                    ss=[17,29,43] if control in NEW_ROLES else [17]
                    if not all((m,control,s,ch) in allrows for s in ss):continue
                    bv=[count_vector(allrows[(m,control,s,ch)])+[allrows[(m,control,s,ch)][k] for k in ['TP','FP','FN']] for s in ss]
                aa.append([count_vector(allrows[(m,'graph',s,ch)])+[allrows[(m,'graph',s,ch)][k] for k in ['TP','FP','FN']] for s in [17,29,43]]);bb.append(bv);used.append(ch)
            if aa:comparisons.append({'mission':m,'contrast':'graph_minus_'+control,'paired_channels':used,'primary':control in ['removed','fusion'],'historical_LSTM_pipeline_unchanged':control=='EXP012_lstm',**paired_bootstrap(aa,bb,.9875 if control in ['removed','fusion'] else .95)})
    table(out/'channel_metrics.csv',cr);table(out/'event_notifications.csv',er);table(out/'events_outside_support.csv',outside_rows);table(out/'mission_seed_metrics.csv',missionseed);table(out/'mission_metrics.csv',missions);write(out/'paired_channel_bootstrap.json',comparisons);write(out/'PUBLISHED_FIDELITY_FAILURES.json',fail);write(out/'LABEL_ROSTER_AUDIT.json',audit)
    write(out/'PUBLISHED_FIDELITY_STATUS.json',{'controlled_adaptations_not_leaderboard_reproductions':True,'source_pins':PINS,'pre_freeze_exclusions':read(work/'STUDY_FREEZE.json')['explicit_published_exclusions'],'confirmation_failures':[read(p) for p in (work/'confirmation').glob('*/results.json') if read(p).get('status')=='FAILED_PUBLISHED_NATIVE_SCORE'],'NASA_native_score_failures':fail,'STGLR':'No source implementation located; faithful reproduction unsupported','OmniAnomaly':'Legacy official environment unsupported in this modern single-notebook panel; CATCH selected instead'})
    confirmation=[]
    for p in (work/'confirmation').glob('*/results.json'):
        v=read(p)
        for item in v['cases']:confirmation.append({'mission':v['mission'],'role':v['role'],'seed':v['seed'],**item})
    table(out/'synthetic_confirmation_cases.csv',confirmation)
    summarize_confirmation(work,out,confirmation)
    decision_report(work,out,confirmation,missions,comparisons)
    write_report(work,out,missions,comparisons,fail)
    seal(out,'COMPLETE.json',{'status':'COMPLETE_RETROSPECTIVE_RESEARCH_REPORT','study_sha256':identity['study'],'score_seal_sha256':identity['score_seal'],'no_independent_real_anomaly_confirmation':True,'EXP012_retrained_or_replayed':False})
    print('REPORTS COMPLETE. NASA results are retrospective; do not retune this release.')

def summarize_confirmation(work,out,cases0):
    rows=[]
    for m in ['MSL','SMAP']:
        for role in NEW_ROLES+['gdn','catch']:
            items=[x for x in cases0 if x['mission']==m and x['role']==role and x['valid']]
            for fam in sorted({x['family'] for x in items}):
                a=[x for x in items if x['family']==fam];clean=[x for x in a if not x['background_overlap']]
                invalid=[x for x in cases0 if x['mission']==m and x['role']==role and x['family']==fam and not x['valid']]
                rows.append({'mission':m,'role':role,'family':fam,'valid_cases':len(a),'invalid_unchanged_cases':len(invalid),'clean_window_cases':len(clean),'background_overlap':np.mean([x['background_overlap'] for x in a]),'unconditional_recall':np.mean([x['unconditional_hit'] for x in a]),'incremental_recall':np.mean([x['incremental_hit'] for x in a]),'conditional_clean_window_recall':np.mean([x['unconditional_hit'] for x in clean]) if clean else None,'point_f0_5':np.mean([x['point_f0_5'] for x in a]),'point_f1':np.mean([x['point_f1'] for x in a]),'held_out_family':fam in ['variance','oscillation']})
    table(out/'synthetic_confirmation_summary.csv',rows)
    nominal=[]
    for p in (work/'confirmation').glob('*/results.json'):
        v=read(p);nominal.extend({'mission':v['mission'],'role':v['role'],'seed':v['seed'],**x} for x in v['nominal'])
    table(out/'confirmation_nominal_alarms.csv',nominal)
    write(out/'SYNTHETIC_CONFIRMATION_SCOPE.json',{'temporal_segment':'TRAIN90:100% quarantined within EXP013 only; historically exposed in earlier research','held_out_families':['variance','oscillation'],'synthetic_evidence_only':True,'retuning_forbidden':True})

def synthetic_objective(items):
    # Equal-family average, then equal-seed average; invalid injections never count.
    seeds0=sorted({x['seed'] for x in items});values=[]
    for s in seeds0:
        a=[x for x in items if x['seed']==s and x['valid']]
        families=sorted({x['family'] for x in a})
        if families:values.append(np.mean([np.mean([.5*float(x['incremental_hit'])+.5*x['point_f0_5'] for x in a if x['family']==f]) for f in families]))
    return float(np.mean(values)) if values else None

def decision_report(work,out,cases0,missions,contrasts):
    synthetic=[];gates=[]
    for m in ['MSL','SMAP']:
        channels=sorted({x['channel'] for x in cases0 if x['mission']==m})
        for role in NEW_ROLES:
            cal=read(work/'development'/f'{m}__{role}'/'selection.json')
            nominal=[x for p in (work/'confirmation').glob('*/results.json') for v in [read(p)] if v['mission']==m and v['role']==role for x in v['nominal']]
            occ=float(np.mean([x['occupancy'] for x in nominal]));starts=float(np.mean([x['starts_per_1000'] for x in nominal]))
            gates.append({'mission':m,'role':role,'CAL_selection_status':cal['status'],'CONF_occupancy_macro':occ,'CONF_starts_per_1000_macro':starts,'gate_pass':cal['status']=='SELECTED_TRAIN_SYNTHETIC' and occ<=.01 and starts<=1.})
        for control in ['removed','fusion']:
            for scope in ['all_families','held_out_families']:
                pairs=[];used=[]
                for ch in channels:
                    def pick(role):return [x for x in cases0 if x['mission']==m and x['channel']==ch and x['role']==role and (scope=='all_families' or x['family'] in ['variance','oscillation'])]
                    a=synthetic_objective(pick('graph'));b=synthetic_objective(pick(control))
                    if a is not None and b is not None:pairs.append(a-b);used.append(ch)
                if not pairs:continue
                d=np.asarray(pairs);gen=np.random.default_rng(20261006);draw=np.concatenate([d[gen.integers(0,len(d),(500,len(d)))].mean(1) for _ in range(20)])
                synthetic.append({'mission':m,'contrast':'graph_minus_'+control,'scope':scope,'paired_channels':used,'difference':float(d.mean()),'confidence':.9875,'percentile_interval':np.quantile(draw,[.00625,.99375]).tolist(),'draws':10000,'evidence':'synthetic mechanism only; shared training limits inferential independence'})
    held=[x for x in synthetic if x['scope']=='held_out_families']
    supported=len(held)==4 and all(x['percentile_interval'][0]>0 for x in held) and all(x['gate_pass'] for x in gates)
    # A joint outcome is descriptive. No architecture, detector or seed is changed here.
    joint=[]
    with (out/'EXP012_original_mission_metrics.csv').open() as f:historical=list(csv.DictReader(f))
    for m in ['MSL','SMAP']:
        g=next((x for x in missions if x['mission']==m and x['role']=='graph'),None)
        for role in ['removed','fusion','gdn','catch','EXP012_lstm']:
            c=next((x for x in missions if x['mission']==m and x['role']==role),None)
            fields=['point_f0_5','point_f1','corrected_event_f0_5','nominal_occupancy','alarm_starts_per_1000_nominal']
            if role=='EXP012_lstm':
                old=next((x for x in historical if x.get('mission')==m and x.get('arm')=='lstm' and x.get('detector')=='hundman_adapt_v1'),None)
                if old:c={k:float(old[k]) if old[k] else None for k in fields};c['complete_coverage']=True
            if not g or not c or not g['complete_coverage'] or not c['complete_coverage'] or any(g[k] is None or c[k] is None for k in fields):
                joint.append({'mission':m,'control':role,'status':'INCOMPLETE_OR_UNDEFINED_NO_SUPERIORITY_CLAIM'});continue
            differences={k:g[k]-c[k] for k in fields}
            ok=all(differences[k]>0 for k in fields[:3]) and all(differences[k]<=0 for k in fields[3:])
            joint.append({'mission':m,'control':role,'graph_minus_control':differences,'joint_descriptive_improvement':ok})
    write(out/'synthetic_paired_channel_bootstrap.json',synthetic)
    write(out/'DECISION.json',{'synthetic_mechanism_support':'SUPPORTED_ON_BOTH_MISSIONS' if supported else 'NOT_ESTABLISHED_ON_BOTH_MISSIONS','synthetic_nominal_gates':gates,'NASA_joint_outcomes':joint,'NASA_corrected_event_contrasts':contrasts,'independent_real_anomaly_improvement_demonstrated':False,'NASA_TEST_previously_exposed':True,'inherited_TEST_minmax_preprocessing':True,'no_TEST_winner_selection_or_retuning':True,'claim':'Even a positive retrospective outcome does not establish independent generalization or SOTA.'})

def write_report(work,out,missions,comparisons,fail):
    lines=['# EXP013 completed research report','','NASA TEST and final TRAIN were previously exposed. Every NASA comparison below is exploratory retrospective evidence. Synthetic confirmation does not establish real spacecraft generalization. EXP012 is unchanged.','','|Mission|Model|Corrected event F0.5|Point F1|Point precision|Point recall|Nominal alarm occupancy|Coverage|','|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return 'undefined' if x is None else f'{x:.4f}'
    for v in missions:lines.append('|'+ '|'.join([v['mission'],v['role']]+[fmt(v[k]) for k in ['corrected_event_f0_5','point_f1','point_precision','point_recall','nominal_occupancy']]+[f"{v['minimum_channel_coverage']}/{v['full_roster']}"])+'|')
    lines+=['','GDN/CATCH use pinned official model code and losses under this controlled NASA protocol; the common splits, pooled entity-window training, causal scoring and TRAIN-only thresholds differ from published leaderboard protocols. These are not reproductions of their reported SOTA scores. Published baselines have one initialization; new models have three. Missing/failed coverage is explicit. STGLR and OmniAnomaly are excluded for the documented fidelity/environment gaps.','','The primary architecture contrasts are graph minus message removal and direct fusion on both missions. Their bootstrap intervals use98.75% coverage to describe four prespecified contrasts. Shared model fitting, shared commands and earlier TEST exposure prevent an independent superiority claim. Historical LSTM complete-pipeline comparisons retain its original adaptive detector and timing; they cannot isolate architecture alone.','','See event notification timing, strict point metrics, alarm burden, calibration/select statuses and separate synthetic confirmation, including background-overlap and invalid-injection counts. A NO_FEASIBLE_CANDIDATE selection remains a negative result regardless of NASA scores.','','The official NASA arrays were scaled using TEST extrema before release. Our additional transformations use FIT only, but inherited preprocessing prevents a raw-data leakage-free claim.','','No new TEST-dependent threshold, seed, smoothing, postprocessing or model selection occurred. No point adjustment is reported. Independent real-anomaly confirmation is still required.','','Synthetic decision: '+read(out/'DECISION.json')['synthetic_mechanism_support']+'. See DECISION.json for all prespecified gates and retrospective joint outcomes.']
    atomic(out/'RESEARCH_REPORT.md','\n'.join(lines).encode())
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,2,figsize=(12,4))
        for a,m in zip(ax,['MSL','SMAP']):
            v=[r for r in missions if r['mission']==m];a.bar([x['role'] for x in v],[x['point_f1'] if x['point_f1'] is not None else np.nan for x in v]);a.set_title(m+' strict point F1 (retrospective)');a.set_ylim(0,1);a.tick_params(axis='x',rotation=25)
            for i,x in enumerate(v):
                if x['point_f1'] is None:a.text(i,.02,'undefined',rotation=90,ha='center')
        fig.tight_layout();fig.savefig(out/'strict_point_F1.png',dpi=180);plt.close(fig)
    except ImportError:pass

def audit(work):
    verify_study(work);s=verify_export(work);manifest(work/'reports','COMPLETE.json')
    access=[json.loads(x) for x in (work/'ACCESS.jsonl').read_text().splitlines()];actions=[v['action'] for v in access]
    if 'NASA_TEST_ARRAY_READ' in actions and actions.index('NASA_TEST_ARRAY_READ')<actions.index('STUDY_SEALED'):raise RuntimeError('NASA TEST accessed before study freeze')
    if 'NASA_LABELS_READ' in actions and actions.index('NASA_LABELS_READ')<actions.index('NASA_SCORE_SEAL_VERIFIED'):raise RuntimeError('Labels accessed before score seal')
    result={'status':'PASS','version':VERSION,'study_freeze_sha256':sha(work/'STUDY_FREEZE.json'),'score_seal_sha256':sha(work/'scores'/'SCORE_SEAL.json'),'report_seal_sha256':sha(work/'reports'/'COMPLETE.json'),'expected_score_units':s['expected_streams'],'actual_score_units':len(list((work/'scores'/'units').iterdir())),'EXP012_fits_or_replay':0,'automated_hyperparameter_selection_from_new_NASA_TEST':False,'architecture_design_informed_by_prior_NASA_TEST_results':True,'NASA_annotation_CSV_parsed_only_after_score_seal':True,'historical_label_derived_reports_present':True,'physical_label_isolation_claimed':False,'scientific_claim':'retrospective exploratory NASA; synthetic mechanism confirmation only; no independent SOTA or deployment confirmation'}
    write(work/'FINAL_AUDIT.json',result);print(json.dumps(result,indent=2));return result
def snapshot(work,dst):
    work=Path(work);dst=Path(dst)
    # .lock files are harmless kernel locks, not stage data.
    records={p.relative_to(work).as_posix():sha(p) for p in work.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.tmp-' not in p.name and not p.name.endswith('.lock')}
    tmp=dst.with_name(dst.name+'.tmp');dst.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for n in sorted(records):z.write(work/n,n)
        z.writestr('SNAPSHOT_MANIFEST.json',blob({'version':VERSION,'files':records}))
    os.replace(tmp,dst);write(str(dst)+'.sha.json',{'sha256':sha(dst),'bytes':dst.stat().st_size});print('Snapshot:',str(dst),dst.stat().st_size,'bytes')
def restore_snapshot(src,work):
    work=Path(work)
    if work.exists() and any(work.iterdir()):raise RuntimeError('Restore into an empty local work folder; never overwrite an active experiment')
    safe_extract(src,work);manifest(work,'SNAPSHOT_MANIFEST.json');(work/'SNAPSHOT_MANIFEST.json').unlink()
    print('Snapshot bytes verified and restored')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['stage','train-select','confirm','export-test','verify-score-seal','evaluate','audit','snapshot','restore']);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--device',default='cuda',choices=['cuda','cpu']);ap.add_argument('--labels',type=Path);ap.add_argument('--file',type=Path);ap.add_argument('--checkpoint-archive',type=Path);a=ap.parse_args();w=a.work.resolve()
    # Drive FUSE is never an active execution location.
    if '/content/drive' in str(w):raise RuntimeError('Use /content local storage; copy archives, never active tiny files, through Drive')
    with Lock(w.parent/(w.name+'.lock')):
        if a.action=='stage':initial(w,a.device)
        elif a.action=='train-select':fit_and_select(w,a.device,a.checkpoint_archive)
        elif a.action=='confirm':confirm_all(w,a.device)
        elif a.action=='export-test':export_test(w,a.device)
        elif a.action=='verify-score-seal':verify_export(w);print('SCORE SEAL VERIFIED')
        elif a.action=='evaluate':
            if a.labels is None:raise ValueError('Missing --labels')
            evaluate(w,a.labels)
        elif a.action=='audit':audit(w)
        elif a.action=='snapshot':snapshot(w,a.file)
        elif a.action=='restore':restore_snapshot(a.file,w)
if __name__=='__main__':main()
