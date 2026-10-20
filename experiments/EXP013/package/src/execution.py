"""E01 orchestration around the unchanged frozen scientific pipeline."""
import contextlib, gc, json, os, shutil, subprocess, sys, threading, time
from pathlib import Path
import core, pipeline, scoring, execution_runtime
from core import ROOT,sha,read,write,manifest,configure,Lock
from recovery import Recovery
from drive_transport import AMENDMENT

class Runner:
    def __init__(self,work,cache,transport,device='cuda',interval_seconds=600):
        if '/content/drive' in str(Path(work).resolve()):raise RuntimeError('Drive FUSE cannot be the live experiment directory')
        self.work=Path(work);self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)
        self.device=device;self.interval=interval_seconds;self.last_progress=0.
        self.start=time.monotonic();self.phase='setup';self.job=None
        self.log=self.cache/'PROGRESS.jsonl';self.lock=None;self.stop_heartbeat=threading.Event()
        identity={'amendment':AMENDMENT,'package_manifest_sha256':sha(ROOT/'PACKAGE_MANIFEST.json'),
                  'protocol_sha256':sha(ROOT/'PROTOCOL.json'),
                  'parent_bundle_sha256':'f94d70b2fbdaa70c22616650ff2993378b6156fa6a1e34ea8ee35e24e2278a5e'}
        self.recovery=Recovery(transport,self.work,self.cache/'archives',identity,keep=2)
        self.last_success=time.monotonic();self.measured_checkpoint_seconds=[];self.phase_rates={}

    def log_record(self,v):
        v={'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**v}
        with self.log.open('a') as f:f.write(json.dumps(v,sort_keys=True)+'\n');f.flush()
        print(json.dumps(v,sort_keys=True),flush=True)

    def completed(self):
        return len(list((self.work/'models').glob('*/COMPLETE.json')))

    def checkpoint_due(self):return time.monotonic()-self.last_success>=self.interval

    def transfer_progress(self,done,total,elapsed):
        self.log_record({'phase':'BACKUP_UPLOAD','bytes':done,'total_bytes':total,
            'elapsed_seconds':round(elapsed,1),'transfer_eta_seconds':round((total-done)*elapsed/max(1,done),1),
            'training_paused_at_safe_checkpoint':True})

    def backup(self,reason,force=False):
        if not force and not self.checkpoint_due():return
        self.phase='durable_backup'
        self.log_record({'phase':'BACKUP_START','reason':reason,'completed_fits':self.completed(),
            'seconds_since_previous_verified_backup':round(time.monotonic()-self.last_success,1),
            'training_paused_at_safe_checkpoint':True})
        # Logging stays outside scientific model directories and sealed study
        # membership. Copy only at this synchronous, quiescent boundary.
        if self.log.exists():core.atomic(self.work/'EXECUTION_PROGRESS.jsonl',self.log.read_bytes())
        before=time.monotonic()
        try:self.recovery.backup(reason,progress=self.transfer_progress)
        except Exception as ex:
            self.log_record({'phase':'BACKUP_FAILED_EXECUTION_STOPPED','error':str(ex),
                'last_good_backup_retained':True,'human_action':'Restore connectivity or quota, then rerun. Training will not continue without protection.'})
            raise
        self.measured_checkpoint_seconds.append(time.monotonic()-before)
        self.last_success=time.monotonic()

    def progress(self,job,phase,epoch,batch,batches,start,epochs,epoch_elapsed=0.,force=False,**extra):
        self.phase=phase;self.job=job;now=time.monotonic()
        if not force and batch not in [1,batches] and now-self.last_progress<30:return
        elapsed=now-start+epoch_elapsed
        key=(job,phase,epoch)
        previous=self.phase_rates.get(key)
        rate=extra.get('batch_seconds')
        if rate is None and previous and batch>previous['batch']:
            rate=(now-previous['time'])/(batch-previous['batch'])
        if rate is not None and previous and previous['rate'] is not None:
            rate=.3*rate+.7*previous['rate']
        self.phase_rates[key]={'batch':batch,'time':now,'rate':rate}
        record={'phase':phase,'job':job,'epoch':epoch,'epoch_cap':epochs,
            'batch':batch,'batches':batches,'epoch_elapsed_seconds':round(elapsed,2),
            'remaining_phase_seconds_estimate':None if rate is None else round((batches-batch)*rate,1),
            'estimate_is_observed_rolling_not_guaranteed':True,
            'completed_fits':self.completed(),'planned_fits':22,
            'seconds_since_verified_backup':round(now-self.last_success,1),**extra}
        if self.device=='cuda':
            import torch
            record.update(GPU_allocated_GiB=round(torch.cuda.memory_allocated()/2**30,3),
                          GPU_reserved_GiB=round(torch.cuda.memory_reserved()/2**30,3),
                          GPU_peak_GiB=round(torch.cuda.max_memory_allocated()/2**30,3))
        self.log_record(record);self.last_progress=now

    def _heartbeat(self):
        while not self.stop_heartbeat.wait(30):
            # No state file is changed by this thread. Backup reads a static copy
            # made by the main thread; heartbeat gives liveness during long calls.
            print('EXP013 E01 heartbeat: phase='+self.phase+' job='+str(self.job)+
                ' completed_fits='+str(self.completed())+'/22 session_seconds='+
                str(round(time.monotonic()-self.start))+' last_verified_backup_age_seconds='+
                str(round(time.monotonic()-self.last_success)),flush=True)

    @contextlib.contextmanager
    def active(self):
        manifest(ROOT,'PACKAGE_MANIFEST.json')
        with Lock(self.work.parent/(self.work.name+'.lock')):
            if execution_runtime.ACTIVE is not None:raise RuntimeError('Another managed experiment is active')
            execution_runtime.ACTIVE=self;self.stop_heartbeat.clear()
            heartbeat=threading.Thread(target=self._heartbeat,daemon=True);heartbeat.start()
            try:yield
            finally:self.stop_heartbeat.set();heartbeat.join(timeout=2);execution_runtime.ACTIVE=None

    def startup(self):
        environment=configure(self.device);manifest(ROOT,'PACKAGE_MANIFEST.json')
        if not self.work.exists() or not any(self.work.iterdir()):
            self.recovery.restore(environment)
        if (self.work/'RUN_FREEZE.json').exists():
            pipeline.verify_run(self.work,self.device)
            if (self.work/'STUDY_FREEZE.json').exists():
                pipeline.verify_study(self.work);self._mark('STUDY_SEALED')
            if (self.work/'scores/SCORE_SEAL.json').exists():self.verify_score_seal()
            self.log_record({'phase':'VERIFIED_LOCAL_OR_RESTORED_REUSE','completed_fits':self.completed()})
        else:
            pipeline.initial(self.work,self.device)
            self.log_record({'phase':'NEW_RUN_NO_RECOVERABLE_OLD_TRAINING','completed_fits':0})
        # Include the EXISTING hash-pinned source archives, avoiding a new network
        # dependency. The unchanged upstream adapter still re-derives all hashes.
        from upstream import PINS
        for name,pin in PINS.items():
            src=ROOT/'inputs'/('pinned_'+name+'.zip');core.checked(src,pin['sha256'])
            dst=self.work/'official'/(name+'.zip')
            if dst.exists():core.checked(dst,pin['sha256'])
            else:
                dst.parent.mkdir(parents=True,exist_ok=True);core.atomic(dst,src.read_bytes())
        self.backup('startup',force=True)

    def _mark(self,action):
        # A VM/kernel interruption can occur after an atomic scientific seal
        # write but before the next ACCESS append. Verification remains mandatory;
        # append the observed verification before advancing, never forge a date.
        access=self.work/'ACCESS.jsonl'
        actions=[] if not access.exists() else [json.loads(line)['action'] for line in access.read_text().splitlines()]
        if action not in actions:core.event(self.work,action,NASA_labels_opened=False,engineering_resume_verified=True)

    def verify_score_seal(self):
        result=pipeline.verify_export(self.work)
        self._mark('NASA_SCORE_SEAL_VERIFIED')
        return result

    def engineering(self):
        self.phase='engineering_validation'
        existing=self.work/'E01_ENGINEERING_GATE.json'
        if existing.exists():
            gate=read(existing)
            if gate.get('status')!='PASS' or gate.get('environment')!=configure(self.device) or gate.get('package_manifest_sha256')!=sha(ROOT/'PACKAGE_MANIFEST.json') or (self.device=='cuda' and not gate.get('actual_GPU_checks_run')):
                raise RuntimeError('Stored E01 engineering gate is incompatible')
            parent=read(self.work/'ENGINEERING_GATE.json')
            if parent.get('status')!='PASS' or parent.get('environment')!=gate['environment'] or parent.get('package_manifest_sha256')!=gate['package_manifest_sha256']:
                raise RuntimeError('Stored parent engineering gate is incompatible')
            print('REUSE VERIFIED E01 AND ORIGINAL ENGINEERING GATES; no repeated validation work',flush=True)
            return
        log=self.work/'E01_ENGINEERING_STDOUT.log'
        # Keep GPU work in the notebook kernel. A kernel restart must not leave
        # an orphan GPU subprocess consuming memory or writing gate state.
        import engineering,runpy
        class Tee:
            def __init__(self,original,file):self.original,self.file=original,file
            def write(self,s):self.original.write(s);self.file.write(s);self.file.flush()
            def flush(self):self.original.flush();self.file.flush()
        saved_active=execution_runtime.ACTIVE;saved_argv=sys.argv
        try:
            execution_runtime.ACTIVE=None  # Fixtures never back up production state.
            with log.open('a') as out,contextlib.redirect_stdout(Tee(sys.stdout,out)),contextlib.redirect_stderr(Tee(sys.stderr,out)):
                sys.argv=['engineering.py','--work',str(self.work),'--device',self.device]
                engineering.main()
                # Original tests remain byte-identical. E01 adds exact FIT/STOP
                # resume, source-oracle parity and durable reset simulations.
                sys.argv=['test_execution.py','--device',self.device,'--official-work',str(self.work)]
                try:runpy.run_path(str(ROOT/'tests/test_execution.py'),run_name='__main__')
                except SystemExit as result:
                    if result.code not in [None,0]:raise RuntimeError('E01 exact batch-resume/compatibility tests failed; scientific execution forbidden')
                sys.argv=['test_release.py']
                try:runpy.run_path(str(ROOT/'tests/test_release.py'),run_name='__main__')
                except SystemExit as result:
                    if result.code not in [None,0]:raise RuntimeError('E01 release/report persistence tests failed; scientific execution forbidden')
        finally:execution_runtime.ACTIVE=saved_active;sys.argv=saved_argv
        gc.collect()
        if self.device=='cuda':
            import torch
            torch.cuda.empty_cache()
        write(self.work/'E01_ENGINEERING_GATE.json',{'status':'PASS','package_manifest_sha256':sha(ROOT/'PACKAGE_MANIFEST.json'),'environment':configure(self.device),'actual_GPU_checks_run':self.device=='cuda','not_scientific_performance_evidence':True})
        self.backup('engineering_gates',force=True)

    def preflight(self):
        from performance import preflight
        self.phase='performance_preflight'
        result=preflight(self.work,self.device,self.recovery.transport,self.cache)
        self.backup('measured_performance_preflight',force=True)
        return result

    @contextlib.contextmanager
    def scientific_hooks(self):
        originals={name:getattr(pipeline,name) for name in ['select_bank','threshold_all','confirm','seal']}
        from execution_scoring import ScoreMemoizer
        original_scoring=scoring.scores;original_pipeline_scoring=pipeline.scores
        cache=ScoreMemoizer(self.work,original_scoring)
        scoring.scores=cache;pipeline.scores=cache
        def wrap(name):
            fn=originals[name]
            def invoke(*args,**kw):
                self.phase=name;result=fn(*args,**kw)
                if name=='seal' and Path(args[0])==self.work/'scores':self.verify_score_seal()
                self.backup('committed_'+name,force=False)
                return result
            return invoke
        for name in originals:setattr(pipeline,name,wrap(name))
        try:yield
        finally:
            for name,value in originals.items():setattr(pipeline,name,value)
            scoring.scores=original_scoring;pipeline.scores=original_pipeline_scoring

    def develop_and_seal(self):
        gate=read(self.work/'E01_ENGINEERING_GATE.json');perf=read(self.work/'PERFORMANCE_PREFLIGHT.json')
        if gate['status']!='PASS' or gate['environment']!=configure(self.device) or gate['package_manifest_sha256']!=sha(ROOT/'PACKAGE_MANIFEST.json'):raise RuntimeError('E01 runtime gate identity differs')
        if perf['status']!='PASS' or perf['environment']!=gate['environment'] or perf['package_manifest_sha256']!=gate['package_manifest_sha256']:raise RuntimeError('A measured, compatible performance preflight is required')
        if self.device=='cuda' and not gate['actual_GPU_checks_run']:raise RuntimeError('Actual E01 GPU resume tests required')
        with self.scientific_hooks():
            self.phase='train-select';pipeline.fit_and_select(self.work,self.device)
            pipeline.verify_study(self.work);self._mark('STUDY_SEALED');self.backup('study_frozen',force=True)
            self.phase='synthetic_confirmation';pipeline.confirm_all(self.work,self.device)
            self.backup('confirmation_complete',force=True)
            self.phase='NASA_TEST_score_export';pipeline.export_test(self.work,self.device)
            self.verify_score_seal();self.backup('verified_score_seal',force=True)
        self.phase='AWAITING_LABEL_CSV_AFTER_VERIFIED_SEAL'
        self.log_record({'phase':self.phase,'completed_fits':self.completed(),
            'required_human_action':'Supply the exact historical NASA annotation CSV in the next notebook cell.',
            'score_seal_sha256':sha(self.work/'scores/SCORE_SEAL.json')})

    def evaluate_and_finish(self,labels):
        self.verify_score_seal()  # Must pass even on completed-report reuse.
        self.phase='final_evaluation';pipeline.evaluate(self.work,labels)
        result=pipeline.audit(self.work)
        self.backup('evaluated_final_audit',force=True)
        return result
