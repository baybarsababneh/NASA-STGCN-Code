"""One bounded run: recover, verify, GDN inference, exact detector, evaluate, export."""
import os, sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1');sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, time, zipfile, shutil, gc
import numpy as np
from integrity import *
from inputs import verify_package,verify_static,recover,required_paths,load_test
from backup import Backup
from vendor.frozen.hundman_detector import detect
from report import labels,generate

def unit_id(m,ch): return f'{m}__gdn_hundman__17__{ch}'
def unit_identity(run_hash,row,ch,source):
    model=read(Path(source)/'models'/f"{row['mission']}__gdn__17/COMPLETE.json")
    return {'run_hash':run_hash,'channel':ch,'mission':row['mission'],'seed':17,'checkpoint_sha256':model['checkpoint_sha256'],'state_sha256':model['state_sha256'],'TEST_array_sha256':row['sha256'],'target_start':250,'detector':'HUNDMAN_ADAPT_V1'}

def verify_unit(path,row,identity):
    m=manifest(path,'COMMIT.json')
    if m['identity']!=identity: raise RuntimeError('Score unit identity differs')
    a=arrays(Path(path)/'scores.npz');record=read(Path(path)/'record.json')
    expected=['target_index','target','prediction','prediction_native_target','scores_raw','scores_ewma','adaptive_flags','adaptive_issue_index']
    ix=np.arange(250,row['rows'],dtype=np.int64)
    if set(a)!=set(expected) or not np.array_equal(a['target_index'],ix) or any(v.shape!=ix.shape for v in a.values()): raise RuntimeError('Score schema/support differs')
    if a['adaptive_flags'].dtype!=np.bool_ or a['target_index'].dtype!=np.int64 or a['prediction_native_target'].dtype!=np.float32: raise RuntimeError('Score precision/schema differs')
    if any(not np.isfinite(a[k]).all() for k in ['target','prediction','prediction_native_target','scores_raw','scores_ewma']): raise RuntimeError('Nonfinite score payload')
    if not np.array_equal(a['scores_raw'],np.abs(a['target']-a['prediction'])): raise RuntimeError('Residual construction differs')
    f=a['adaptive_flags'];issue=a['adaptive_issue_index']
    if np.any(issue[f]<ix[f]) or np.any(issue[~f]!=-1) or record['identity']!=identity or record['native_replay']['frozen_native_score_replay']!='BITWISE_PASS': raise RuntimeError('Detector issue/provenance differs')
    return record

def verify_score_seal(work,root,identity,source):
    work=Path(work);rows=read(Path(root)/'inputs/TEST_ROSTER.json');s=read(work/'SCORE_SEAL.json')
    expected={unit_id(r['mission'],ch) for ch,r in rows.items()}
    actual={p.name for p in (work/'units').iterdir() if p.is_dir() and not p.name.startswith('.tmp-')}
    if actual!=expected or set(s['roster'])!=expected or len(s['roster'])!=len(expected) or s['run_hash']!=digest(identity) or s['annotation_CSV_parsed']: raise RuntimeError('Incomplete/stale score seal')
    if s['files']!=files(work/'units'): raise RuntimeError('Sealed score bytes differ')
    if s['frozen_EXP012_subset_sha256']!=sha(Path(root)/'inputs/exp012_scores/SUBSET.json'): raise RuntimeError('Historical input subset differs')
    for ch,row in rows.items():verify_unit(work/'units'/unit_id(row['mission'],ch),row,unit_identity(digest(identity),row,ch,source))
    return s

def snapshot(work,path):
    work=Path(work);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+__import__('uuid').uuid4().hex)
    try:
        records=files(work)
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for n in sorted(records):z.write(work/n,n)
            z.writestr('SNAPSHOT_MANIFEST.json',blob({'version':VERSION,'files':records}))
        with zipfile.ZipFile(tmp) as z:
            for n,h in records.items():
                if __import__('hashlib').sha256(z.read(n)).hexdigest()!=h:raise RuntimeError('Final ZIP verification failed')
        os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)
    return sha(path)

def publish_archive(local,dest):
    h=sha(local);dest=Path(dest)
    if dest.exists():
        checked(dest,h);return
    tmp=dest.with_name(dest.name+'.partial-'+__import__('uuid').uuid4().hex)
    try:
        dest.parent.mkdir(parents=True,exist_ok=True)
        with Path(local).open('rb') as src,tmp.open('wb') as dst:
            shutil.copyfileobj(src,dst,8*2**20);dst.flush();os.fsync(dst.fileno())
        checked(tmp,h);os.replace(tmp,dest);checked(dest,h)
    finally:tmp.unlink(missing_ok=True)

def execute(root,source_path,work,destination,cache,device='cuda',fixture=False,interrupt_after_units=None):
    from gdn_inference import load_model,full_shape_gate,transform,inverse_target,predict,verify_reference,memory_record,host_budget
    import torch
    root,work,destination,cache=map(Path,(root,work,destination,cache))
    w,d,c=map(lambda p:p.resolve(),(work,destination,cache))
    for original in [root.resolve(),Path(source_path).resolve()]:
        for mutable in [w,d,c]:
            if mutable==original or original in mutable.parents or mutable in original.parents: raise RuntimeError('Keep outputs/cache/backup separate from immutable source directories')
    inventory,rows=verify_static(root);cfg=read(root/'config.json')
    if bool(cfg.get('fixture',False))!=fixture: raise RuntimeError('Fixture inputs cannot become a NASA comparison')
    if not fixture and (len(rows)!=81 or sum(r['mission']=='MSL' for r in rows.values())!=27): raise RuntimeError('Frozen mission roster differs')
    cache.mkdir(parents=True,exist_ok=True);local_source=cache/'verified_source'
    try:recovered=recover(root,source_path,local_source)
    except Exception as exc:
        write(cache/'MISSING_OR_INVALID_INPUTS.json',{'status':'BLOCKED_IMMUTABLE_INPUTS','reason':str(exc),'expected_source':str(source_path),'constituents':required_paths(root),'no_retraining_or_substitution':True})
        raise
    environment=configure(device,fixture)
    print('RUNTIME',__import__('json').dumps(environment,sort_keys=True),flush=True)
    identity={'version':VERSION,'configuration':cfg,'environment':environment,'code_manifest_sha256':sha(ROOT/'MANIFEST.json'),'immutable_input_inventory_sha256':sha(root/'INPUT_INVENTORY.json'),'frozen_detector_sha256':sha(ROOT/'vendor/frozen/hundman_detector.py'),'frozen_evaluator_sha256':sha(ROOT/'vendor/frozen/metrics.py'),'verified_EXP013_input_digest':digest(recovered),'fixture':fixture}
    backup=Backup(work,destination,cache/'backup_archive',identity)
    if not work.exists() or not any(work.iterdir()):backup.restore(environment)
    work.mkdir(parents=True,exist_ok=True)
    with Lock(work/'run.lock'):
        freeze(work/'RUN_FREEZE.json',{'identity':identity,'environment':environment})
        freeze(work/'INPUTS.json',{'embedded_input_hashes':inventory['present'],'recovered_EXP013':recovered,'immutable_originals':True})
        freeze(work/'PROTOCOL_FREEZE.json',{'config':cfg,'no_new_training':True,'no_new_calibration':True,'no_TEST_label_selection':True,'common_detector':'exact EXP012 original-unit forecast residuals; fresh detector stream starts at250','native_replay':'preserve native GDN t5 batch boundaries and history, then crop at250','CATCH_Hundman_hybrid':False,'scoring_code_sha256':sha(ROOT/'src/gdn_inference.py')})
        discard_uncommitted(work)
        guard=LabelGuard(root/'inputs/labeled_anomalies.csv')
        last_backup=[time.monotonic()]
        def checkpoint(reason='periodic',force=False):
            if force or time.monotonic()-last_backup[0]>=cfg['backup_interval_seconds']:
                backup.save(reason);last_backup[0]=time.monotonic()
        event(work,'PROTOCOL_FROZEN',annotation_CSV_parsed=False,fixture=fixture)
        checkpoint('startup',True)
        guard.install()
        try:
            if not (work/'SCORE_SEAL.json').exists():
                gates=read(work/'MEMORY_GATE.json') if (work/'MEMORY_GATE.json').exists() else {'records':[],'fixture':fixture}
                done=0;new_times=[];started=time.monotonic()
                for mission in ['MSL','SMAP']:
                    mission_rows={ch:r for ch,r in rows.items() if r['mission']==mission}
                    pending=[ch for ch,r in mission_rows.items() if committed(work,unit_id(mission,ch),unit_identity(digest(identity),r,ch,local_source)) is None]
                    model=ck=st=None
                    if pending:
                        model,ck,st=load_model(local_source,mission,root,device)
                        gate=full_shape_gate(model,ck,st,local_source,mission,root,device,work)
                        gates['records']=[r for r in gates['records'] if r['mission']!=mission]+[gate]
                        write(work/'MEMORY_GATE.json',gates);checkpoint('full_shape_'+mission,True)
                    for ch,row in mission_rows.items():
                        jid=unit_id(mission,ch);uid=unit_identity(digest(identity),row,ch,local_source)
                        previous=committed(work,jid,uid)
                        if previous is not None:
                            verify_unit(previous,row,uid);done+=1;print(f'REUSE {done}/{len(rows)} {jid}',flush=True);continue
                        t0=time.monotonic()
                        if host_budget(row['rows'],row['columns'])>cfg['maximum_single_channel_host_bytes']:raise RuntimeError('Channel host-memory estimate exceeds budget before array loading')
                        original=load_test(root,ch)
                        meta=read(root/'inputs/freeze/DATA_FREEZE.json')['channels'][ch]
                        z=transform(original,meta,ck['scaler'])
                        if device=='cuda':torch.cuda.reset_peak_memory_stats()
                        pred,errors=predict(model,z,device,cfg,work,ch,checkpoint)
                        replay=verify_reference(local_source,mission,ch,errors,root)
                        # Native exports preserve smoothing fromt5. Hundman is a
                        # separate frozen detector and receives only t250 onward.
                        pred=pred[250-5:];target=original[250:,0].astype(np.float64)
                        prediction=inverse_target(pred,meta,ck['scaler']);ix=np.arange(250,len(original),dtype=np.int64)
                        output=detect(target,prediction,ix,cfg['detector'])
                        values={'target_index':ix,'target':target,'prediction':prediction,'prediction_native_target':pred,'scores_raw':np.abs(target-prediction),'scores_ewma':output['smoothed'],'adaptive_flags':output['flags'],'adaptive_issue_index':output['issue_index']}
                        if device=='cuda':torch.cuda.synchronize()
                        record={'job_id':jid,'identity':uid,'native_replay':replay,'status':'FINITE_SCORES','channel':ch,'mission':mission,'seed':17,'adaptive_status':output['status'],'startup_averaging_lookahead':output['startup_averaging_lookahead'],'seconds':time.monotonic()-t0,'memory':memory_record(device),'original_TEST_rows':len(original),'target_start':250,'target_end_exclusive':len(original),'annotation_CSV_parsed':False,'target_units':'released NASA telemetry units; both frozen scaling stages inverted','native_batch_boundaries':'start5, batch32; last remainder retained','new_detector_start':250,'fixture':fixture}
                        p=commit(work,jid,uid,{'scores.npz':npz_bytes(**values),'record.json':blob(record),'adaptive_windows.json':blob(output['windows'])})
                        verify_unit(p,row,uid);done+=1;new_times.append(time.monotonic()-t0)
                        event(work,'UNIT_COMMITTED',job_id=jid,seconds=new_times[-1],native_replay='BITWISE_PASS',annotation_CSV_parsed=False)
                        checkpoint('periodic_completed_units')
                        eta=(len(rows)-done)*float(np.mean(new_times));print(f'COMPLETE {done}/{len(rows)} {jid} · {new_times[-1]:.2f}s · measured mean unit ETA {eta/60:.1f} min (channel lengths differ; excludes setup/backup/evaluation)',flush=True)
                        del original,z,pred,errors,target,prediction,values,output;gc.collect()
                        if interrupt_after_units is not None and len(new_times)>=interrupt_after_units:
                            checkpoint('simulated_interrupt',True);raise InterruptedError('Synthetic interruption after committed scientific unit')
                    if model is not None: del model,ck,st;gc.collect()
                    if device=='cuda':torch.cuda.empty_cache()
                    checkpoint('mission_complete_'+mission,True)
                expected=[unit_id(r['mission'],ch) for ch,r in rows.items()]
                freeze(work/'SCORE_SEAL.json',{'version':VERSION,'run_hash':digest(identity),'files':files(work/'units'),'roster':expected,'annotation_CSV_parsed':False,'frozen_EXP012_subset_sha256':sha(root/'inputs/exp012_scores/SUBSET.json'),'frozen_EXP013_score_seal_sha256':recovered['score_seal_sha256'],'new_detector_selection':False,'fixture':fixture})
                checkpoint('score_seal',True)
            verify_score_seal(work,root,identity,local_source)
            # Recheck the original subset before allowing any new annotation read.
            for n,h in read(root/'inputs/exp012_scores/SUBSET.json')['files'].items(): checked(root/'inputs/exp012_scores'/n,h)
            guard.allowed=True;event(work,'SCORE_SEAL_VERIFIED_BEFORE_ANNOTATION_PARSE')
            ev,audit=labels(root,True);event(work,'ANNOTATION_CSV_PARSED',label_rows=audit['CSV_rows'])
            summary=generate(root,work,identity,ev,audit,fixture)
            gate=read(work/'MEMORY_GATE.json')
            if {r['mission'] for r in gate['records']}!={'MSL','SMAP'}: raise RuntimeError('Both full-shape restoration checks are required')
            if not fixture and any(not r['measured_memory']['GPU_validation'] for r in gate['records']): raise RuntimeError('CPU receipt cannot satisfy real A100 production gate')
            audit={'status':summary['status'],'score_seal_sha256':sha(work/'SCORE_SEAL.json'),'report_seal_sha256':sha(work/'reports/COMPLETE.json'),'input_manifest_sha256':sha(work/'INPUTS.json'),'new_scientific_fits':0,'new_score_streams':len(rows),'frozen_EXP012_exact_count_regression':'PASS','frozen_GDN_native_output_replay':'BITWISE_PASS_ALL_CHANNELS','full_shape_exact_restoration':'PASS_BOTH_MISSIONS','A100_validation':not fixture,'fixture':fixture,'originals_modified':False,'scientific_interpretation':'retrospective descriptive comparison, not independent confirmation','exclusions_preserved':['STGLR_FIDELITY_GAP','OmniAnomaly_LEGACY_ENVIRONMENT','CATCH_HUNDMAN_NOT_NECESSARY']}
            freeze(work/'FINAL_AUDIT.json',audit);checkpoint('final_reports_and_audit',True)
            archive=cache/'EXP014_FINAL_RESULTS.zip';archive_hash=snapshot(work,archive)
            # Logs grow on later verified resumes. Each immutable final export
            # has its own content-addressed name; the first stable alias is retained.
            publish_archive(archive,destination/'exports'/('EXP014_FINAL_RESULTS_'+archive_hash[:16]+'.zip'))
            if not (destination/'FINAL_RESULTS.zip').exists():publish_archive(archive,destination/'FINAL_RESULTS.zip')
            publish_archive(work/'reports/REPORT_READY_SUMMARY.md',destination/'exports'/('REPORT_READY_SUMMARY_'+sha(work/'reports/REPORT_READY_SUMMARY.md')[:16]+'.md'))
            write(destination/'LATEST_EXPORT.json',{'archive':str('exports/EXP014_FINAL_RESULTS_'+archive_hash[:16]+'.zip'),'sha256':archive_hash,'status':summary['status']})
            print('COMPLETE',summary['status'],'\nResults:',destination/'exports'/('EXP014_FINAL_RESULTS_'+archive_hash[:16]+'.zip'),'\nOriginal EXP012/EXP013 outputs unchanged.',flush=True)
            return summary
        except BaseException as exc:
            event(work,'RUN_STOPPED',type=type(exc).__name__,reason=str(exc))
            try:checkpoint('failure_preserved',True)
            except Exception as backup_error:print('Backup failed:',str(backup_error),'Local committed units remain intact; no durable success is claimed.',flush=True)
            raise
        finally:
            guard.active=False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--backup',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--input-root',type=Path,default=ROOT);ap.add_argument('--fixture',action='store_true')
    a=ap.parse_args();verify_package()
    execute(a.input_root,a.source,a.work,a.backup,a.cache,'cpu' if a.fixture else 'cuda',a.fixture)
if __name__=='__main__': main()
