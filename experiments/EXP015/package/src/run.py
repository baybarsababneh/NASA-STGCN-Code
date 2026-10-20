"""One inference-only, frozen EXP013 residual/Hundman supplementary experiment."""
import os,sys
os.environ['MPLBACKEND']='Agg'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1');sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,time,gc
import numpy as np
from integrity import *
from inputs import verify_package,verify_static,recover,required_paths,load_test
from backup import Backup
from export import snapshot,publish_archive
from vendor.frozen.hundman_detector import detect

def unit_id(m,role,seed,ch):return f'{m}__{role}_residual_hundman__{seed}__{ch}'
def unit_identity(run_hash,row,ch,role,seed,source):
    model=read(Path(source)/'models'/f"{row['mission']}__{role}__{seed}/COMPLETE.json")
    return {'run_hash':run_hash,'channel':ch,'mission':row['mission'],'role':role,'seed':seed,
            'checkpoint_sha256':model['checkpoint_sha256'],'state_sha256':model['state_sha256'],
            'TEST_array_sha256':row['sha256'],'target_start':250,'detector':'HUNDMAN_ADAPT_V1',
            'signal':'absolute command-reference residual in released telemetry units'}

def verify_unit(path,row,identity):
    m=manifest(path,'COMMIT.json');a=arrays(Path(path)/'scores.npz');record=read(Path(path)/'record.json')
    if m['identity']!=identity or record['identity']!=identity:raise RuntimeError('Stale score-unit identity')
    expected={'target_index','target','reference','reference_native_target','native_logit','scores_raw','scores_ewma','adaptive_flags','adaptive_issue_index'}
    ix=np.arange(250,row['rows'],dtype=np.int64)
    if set(a)!=expected or not np.array_equal(a['target_index'],ix) or any(v.shape!=ix.shape for v in a.values()):raise RuntimeError('Score schema/support differs')
    if a['target_index'].dtype!=np.int64 or a['adaptive_flags'].dtype!=np.bool_ or a['reference_native_target'].dtype!=np.float32 or a['native_logit'].dtype!=np.float32:raise RuntimeError('FP32 inference/schema differs')
    if any(not np.isfinite(a[k]).all() for k in expected-{'adaptive_flags','adaptive_issue_index','target_index'}):raise RuntimeError('Nonfinite score payload')
    if not np.array_equal(a['scores_raw'],np.abs(a['target']-a['reference'])):raise RuntimeError('Residual construction differs')
    f=a['adaptive_flags'];issue=a['adaptive_issue_index']
    if np.any(issue[f]<ix[f]) or np.any(issue[f]>ix[-1]) or np.any(issue[~f]!=-1) or record['native_replay']['frozen_native_score_replay']!='BITWISE_PASS':raise RuntimeError('Native replay or issue timing differs')
    return record

def verify_score_seal(work,root,identity,source):
    work=Path(work);cfg=read(Path(root)/'config.json');rows=read(Path(root)/'inputs/TEST_ROSTER.json');s=read(work/'SCORE_SEAL.json')
    expected={unit_id(r['mission'],role,seed,ch) for ch,r in rows.items() for role in cfg['roles'] for seed in cfg['seeds']}
    actual={p.name for p in (work/'units').iterdir() if p.is_dir() and not p.name.startswith('.tmp-')}
    if actual!=expected or set(s['roster'])!=expected or len(s['roster'])!=len(expected) or s['run_hash']!=digest(identity) or s['annotation_CSV_parsed']:raise RuntimeError('Incomplete or stale score seal')
    if s['files']!=files(work/'units'):raise RuntimeError('Sealed scores differ')
    for ch,row in rows.items():
        for role in cfg['roles']:
            for seed in cfg['seeds']:verify_unit(work/'units'/unit_id(row['mission'],role,seed,ch),row,unit_identity(digest(identity),row,ch,role,seed,source))
    return s

def execute(root,source_path,work,destination,cache,device='cuda',fixture=False,interrupt_after_units=None):
    from inference import load_model,model_job,full_shape_gate,transform,inverse_reference,predict,verify_reference,memory_record,host_budget
    from report_common import labels
    from report import generate
    import torch
    root,work,destination,cache=map(Path,(root,work,destination,cache));setup_start=time.monotonic()
    mutables=[p.resolve() for p in [work,destination,cache]]
    for original in [root.resolve(),Path(source_path).resolve()]:
        for mutable in mutables:
            if mutable==original or original in mutable.parents or mutable in original.parents:raise RuntimeError('Outputs/cache/backups must be separate from immutable inputs')
    if len(set(mutables))!=3:raise RuntimeError('Work, cache and backup must differ')
    inventory,rows=verify_static(root);cfg=read(root/'config.json')
    if bool(cfg.get('fixture',False))!=fixture:raise RuntimeError('Synthetic inputs cannot become a NASA comparison')
    cache.mkdir(parents=True,exist_ok=True);local_source=cache/'verified_source'
    try:recovered=recover(root,source_path,local_source)
    except Exception as exc:
        write(cache/'MISSING_OR_INVALID_INPUTS.json',{'status':'BLOCKED_IMMUTABLE_INPUTS','reason':str(exc),
              'exact_requested_file':str(source_path),'constituents':required_paths(root),'no_retraining_or_substitution':True})
        raise
    environment=configure(device,fixture)
    identity={'version':VERSION,'configuration':cfg,'environment':environment,'code_manifest_sha256':sha(ROOT/'MANIFEST.json'),
              'immutable_input_inventory_sha256':sha(root/'INPUT_INVENTORY.json'),
              'scientific_decision_sha256':sha(ROOT/'docs/SCIENTIFIC_DECISION.md'),
              'frozen_detector_sha256':sha(ROOT/'vendor/frozen/hundman_detector.py'),
              'frozen_evaluator_sha256':sha(ROOT/'vendor/frozen/metrics.py'),
              'frozen_architecture_sha256':sha(ROOT/'vendor/exp013/architectures.py'),
              'verified_EXP013_input_digest':digest(recovered),'fixture':fixture}
    backup=Backup(work,destination,cache/'backup_archive',identity)
    if not work.exists() or not any(work.iterdir()):backup.restore(environment)
    work.mkdir(parents=True,exist_ok=True)
    with Lock(work/'run.lock'):
        freeze(work/'RUN_FREEZE.json',{'identity':identity,'environment':environment})
        freeze(work/'INPUTS.json',{'embedded_input_hashes':inventory['present'],'recovered_EXP013':recovered,'immutable_originals':True})
        freeze(work/'PROTOCOL_FREEZE.json',{'configuration':cfg,'no_new_training':True,'no_new_calibration':True,
              'no_TEST_label_selection':True,'common_detector':'byte-preserved EXP012 HUNDMAN_ADAPT_V1',
              'residual':'abs(original_TEST[t,0] - (float64(reference[t])*entity_FIT_scale+entity_FIT_mean))',
              'alignment':'ref[t] uses commands t64:t-1; current telemetry t is only used by discriminator and observed residual',
              'native_replay':'original 256 batches starting t64; old EWMA before common t250 crop',
              'new_detector_stream_start':250,'discriminator_hybrid':False,
              'decision_sha256':identity['scientific_decision_sha256'],'annotation_CSV_parsed':False})
        # Final results carry the exact executed small sources and frozen detector/evaluator.
        for folder in ['src','vendor/frozen','vendor/exp013','docs']:
            for p in (ROOT/folder).glob('*.py' if folder!='docs' else '*.md'):
                dst=work/'source'/p.relative_to(ROOT)
                if dst.exists():checked(dst,sha(p))
                else:atomic(dst,p.read_bytes())
        discard_uncommitted(work);guard=LabelGuard(root/'inputs/labeled_anomalies.csv');guard.install()
        last_backup=[time.monotonic()]
        def checkpoint(reason='periodic',force=False):
            if force or time.monotonic()-last_backup[0]>=cfg['backup_interval_seconds']:
                backup.save(reason);last_backup[0]=time.monotonic()
        event(work,'PROTOCOL_FROZEN',fixture=fixture,annotation_CSV_parsed=False,setup_source_seconds=time.monotonic()-setup_start)
        checkpoint('startup',True)
        try:
            if not (work/'SCORE_SEAL.json').exists():
                gates=read(work/'MEMORY_GATE.json') if (work/'MEMORY_GATE.json').exists() else {'records':[],'fixture':fixture}
                total=len(rows)*len(cfg['roles'])*len(cfg['seeds']);done=0;session_units=0
                rates={'inference_seconds':0.,'detector_seconds':0.,'targets':0}
                all_jobs=[(m,r,s) for m in cfg['missions'] for r in cfg['roles'] for s in cfg['seeds']]
                for mission,role,seed in all_jobs:
                    job=model_job(mission,role,seed);mission_rows={ch:r for ch,r in rows.items() if r['mission']==mission}
                    pending=[ch for ch,row in mission_rows.items() if committed(work,unit_id(mission,role,seed,ch),unit_identity(digest(identity),row,ch,role,seed,local_source)) is None]
                    model=ck=st=None
                    if pending:
                        model,ck,st=load_model(local_source,mission,role,seed,root,device)
                        gate=full_shape_gate(model,ck,st,local_source,mission,role,seed,root,device,work)
                        gates['records']=[v for v in gates['records'] if v['job_id']!=job]+[gate]
                        write(work/'MEMORY_GATE.json',gates);checkpoint('full_shape_'+job,True)
                        print('FULL-SHAPE PASS',job,'seconds',round(gate['seconds_first_forward'],3),'peak bytes',gate['measured_memory'].get('peak_reserved_bytes'),'actual GPU',device=='cuda',flush=True)
                    for ch,row in mission_rows.items():
                        jid=unit_id(mission,role,seed,ch);uid=unit_identity(digest(identity),row,ch,role,seed,local_source)
                        previous=committed(work,jid,uid)
                        if previous is not None:
                            verify_unit(previous,row,uid);done+=1;print(f'REUSE {done}/{total} {jid}',flush=True);continue
                        if host_budget(row['rows'],row['columns'])>cfg['maximum_single_channel_host_bytes']:raise RuntimeError('Host budget exceeded before TEST loading')
                        original=load_test(root,ch);meta=read(root/'inputs/freeze/DATA_FREEZE.json')['channels'][ch]
                        z=transform(original,meta,ck['scaler']);entity=ck['channels'].index(ch)
                        if device=='cuda':torch.cuda.reset_peak_memory_stats()
                        begin=time.monotonic();references,logits=predict(model,z,entity,device,cfg,work,ch,checkpoint)
                        inference_seconds=time.monotonic()-begin;replay=verify_reference(local_source,mission,role,seed,ch,logits,root)
                        native_ref=references[250-64:];target=original[250:,0].astype(np.float64)
                        reference=inverse_reference(native_ref,meta,ck['scaler']);ix=np.arange(250,len(original),dtype=np.int64)
                        begin=time.monotonic();out=detect(target,reference,ix,cfg['detector']);detector_seconds=time.monotonic()-begin
                        values={'target_index':ix,'target':target,'reference':reference,'reference_native_target':native_ref,
                                'native_logit':logits[250-64:],'scores_raw':np.abs(target-reference),'scores_ewma':out['smoothed'],
                                'adaptive_flags':out['flags'],'adaptive_issue_index':out['issue_index']}
                        record={'job_id':jid,'identity':uid,'native_replay':replay,'status':'FINITE_SCORES','channel':ch,
                                'mission':mission,'role':role,'seed':seed,'adaptive_status':out['status'],
                                'startup_averaging_lookahead':out['startup_averaging_lookahead'],
                                'inference_seconds':inference_seconds,'detector_seconds':detector_seconds,'memory':memory_record(device),
                                'original_TEST_rows':len(original),'target_start':250,'target_end_exclusive':len(original),
                                'annotation_CSV_parsed':False,'released_telemetry_inverse_scale':{'mean':meta['mean'],'scale':meta['scale']},
                                'native_batch_boundaries':'start64,batch256,retain last remainder','new_detector_start':250,
                                'forecast_history':'commands only; removed role is entity-only constant reference','fixture':fixture}
                        p=commit(work,jid,uid,{'scores.npz':npz_bytes(**values),'record.json':blob(record),'adaptive_windows.json':blob(out['windows'])})
                        verify_unit(p,row,uid);done+=1;session_units+=1
                        rates['inference_seconds']+=inference_seconds;rates['detector_seconds']+=detector_seconds;rates['targets']+=len(ix)
                        event(work,'UNIT_COMMITTED',job_id=jid,inference_seconds=inference_seconds,detector_seconds=detector_seconds,native_replay='BITWISE_PASS',annotation_CSV_parsed=False)
                        checkpoint('completed_units')
                        remaining=sum(v['rows']-250 for c,v in rows.items() for r in cfg['roles'] for s in cfg['seeds'] if not (work/'units'/unit_id(v['mission'],r,s,c)/'COMMIT.json').exists())
                        eta_inf=remaining*rates['inference_seconds']/rates['targets'];eta_det=remaining*rates['detector_seconds']/rates['targets']
                        print(f'COMPLETE {done}/{total} {jid} · inference {inference_seconds:.2f}s, detector {detector_seconds:.2f}s · measured remaining inference {eta_inf/60:.1f} min, detector {eta_det/60:.1f} min; excludes backup/scoring/export; mission and role workloads differ',flush=True)
                        del original,z,references,logits,native_ref,target,reference,values,out;gc.collect()
                        if interrupt_after_units is not None and session_units>=interrupt_after_units:
                            checkpoint('simulated_interrupt',True);raise InterruptedError('Synthetic interruption after committed score unit')
                    if model is not None:del model,ck,st;gc.collect()
                    if device=='cuda':torch.cuda.empty_cache()
                    checkpoint('job_complete_'+job,True)
                roster=[unit_id(row['mission'],role,seed,ch) for ch,row in rows.items() for role in cfg['roles'] for seed in cfg['seeds']]
                freeze(work/'SCORE_SEAL.json',{'version':VERSION,'run_hash':digest(identity),'files':files(work/'units'),'roster':roster,
                      'annotation_CSV_parsed':False,'frozen_EXP013_score_seal_sha256':recovered['score_seal_sha256'],
                      'new_detector_selection':False,'fixture':fixture})
                checkpoint('score_seal',True)
            verify_score_seal(work,root,identity,local_source)
            guard.allowed=True;event(work,'SCORE_SEAL_VERIFIED_BEFORE_ANNOTATION_PARSE')
            events,label_audit=labels(root,True);event(work,'ANNOTATION_CSV_PARSED',label_rows=label_audit['CSV_rows'])
            scoring_start=time.monotonic();summary=generate(root,work,identity,events,label_audit,fixture)
            gate=read(work/'MEMORY_GATE.json');jobs={f'{m}__{r}__{s}' for m in cfg['missions'] for r in cfg['roles'] for s in cfg['seeds']}
            if {v['job_id'] for v in gate['records']}!=jobs:raise RuntimeError('All 18 full-shape receipts required')
            if not fixture and any(not v['measured_memory']['GPU_validation'] for v in gate['records']):raise RuntimeError('CPU receipt cannot pass production A100 gate')
            # Recheck original content after execution; never trust only cached copies.
            guard.active=False
            again,_=verify_static(root);after=recover(root,source_path,local_source)
            if after!=recovered or again!=inventory:raise RuntimeError('Immutable originals changed during the experiment')
            audit={'status':summary['status'],'score_seal_sha256':sha(work/'SCORE_SEAL.json'),'report_seal_sha256':sha(work/'reports/COMPLETE.json'),
                   'input_manifest_sha256':sha(work/'INPUTS.json'),'new_scientific_fits':0,'new_score_streams':len(rows)*9,
                   'frozen_EXP013_native_output_replay':'BITWISE_PASS_ALL_STREAMS','full_shape_exact_restoration':'PASS_ALL_18_FITS',
                   'A100_validation':not fixture,'GPU':environment['GPU'],'GPU_total_bytes':environment['GPU_total_bytes'],
                   'fixture':fixture,'originals_modified':False,'original_content_rechecked_after_execution':True,
                   'discriminator_hybrid_executed':False,'scientific_interpretation':'retrospective exploratory detector-pipeline evidence; no architecture-only or independent operational claim'}
            freeze(work/'FINAL_AUDIT.json',audit)
            event(work,'SCORING_AUDIT_COMPLETE',seconds=time.monotonic()-scoring_start)
            checkpoint('final_reports_audit',True)
            archive=cache/'EXP015_FINAL_RESULTS.zip';beg=time.monotonic();h=snapshot(work,archive)
            publish_archive(archive,destination/'exports'/('EXP015_FINAL_RESULTS_'+h[:16]+'.zip'))
            if not (destination/'FINAL_RESULTS.zip').exists():publish_archive(archive,destination/'FINAL_RESULTS.zip')
            publish_archive(work/'reports/REPORT_READY_SUMMARY.md',destination/'exports'/('REPORT_READY_SUMMARY_'+sha(work/'reports/REPORT_READY_SUMMARY.md')[:16]+'.md'))
            write(destination/'LATEST_EXPORT.json',{'archive':'exports/EXP015_FINAL_RESULTS_'+h[:16]+'.zip','sha256':h,'status':summary['status']})
            print('FINAL EXPORT VERIFIED',destination/'exports'/('EXP015_FINAL_RESULTS_'+h[:16]+'.zip'),'export seconds',round(time.monotonic()-beg,2),'\nOriginal EXP012/EXP013/EXP014 outputs unchanged.',flush=True)
            return summary
        except BaseException as exc:
            event(work,'RUN_STOPPED',type=type(exc).__name__,reason=str(exc))
            try:checkpoint('failure_preserved',True)
            except Exception as failure:print('Backup failed:',str(failure),'No durable success is claimed.',flush=True)
            raise
        finally:guard.active=False

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--backup',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--input-root',type=Path,default=ROOT);ap.add_argument('--fixture',action='store_true')
    a=ap.parse_args();verify_package();execute(a.input_root,a.source,a.work,a.backup,a.cache,'cpu' if a.fixture else 'cuda',a.fixture)
if __name__=='__main__':main()
