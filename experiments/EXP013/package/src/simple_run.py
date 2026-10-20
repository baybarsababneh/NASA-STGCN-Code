"""E02 straightforward runner: one process/action, ordinary mounted Drive backups.

The original pipeline owns every scientific boundary and the common evaluator.
This module owns progress, safe checkpoint hooks and a new honest run identity.
"""
import argparse, contextlib, gc, json, os, shutil, time
from pathlib import Path
import torch
import core, pipeline, data, train, upstream, scoring, execution_runtime as runtime
from simple_backup import Backup, VERSION
from simple_gate import smoke, hardware_gate, verify_hardware_gate

class Runner:
    def __init__(self,work,drive,cache,device='cuda',interval=600):
        self.work,self.drive,self.cache=map(Path,(work,drive,cache)); self.device=device; self.interval=interval
        if '/content/drive' in str(self.work.resolve()):
            raise RuntimeError('Drive FUSE cannot be the live scientific directory; use local /content')
        self.cache.mkdir(parents=True,exist_ok=True)
        self.last_backup=time.monotonic(); self.last_progress=0.; self.batch=0; self.action='startup'; self.trace=False
        self.log=self.cache/'progress.jsonl'; self.status=self.cache/'operation.json'
        self.operations=[]
        self.identity={'amendment':VERSION,'package_manifest_sha256':core.sha(core.ROOT/'PACKAGE_MANIFEST.json')}
        self.backups=Backup(self.work,self.drive,self.cache/'backup',self.identity)
        self.copied_official=False

    def log_record(self,record):
        record={'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**record}
        with self.log.open('a') as f: f.write(json.dumps(record,sort_keys=True)+'\n'); f.flush()
        print(json.dumps(record,sort_keys=True),flush=True)

    @contextlib.contextmanager
    def operation(self,name,deadline=180,**details):
        start=time.monotonic(); token=__import__('uuid').uuid4().hex
        entry={'name':name,'start_monotonic':start,'deadline':deadline,'token':token,**details}
        self.operations.append(entry);core.write(self.status,{'pid':os.getpid(),'operations':self.operations})
        measured=self.device=='cuda' and (self.trace and details.get('batch',self.batch)<=2 or self.action=='hardware')
        announce=self.trace and details.get('batch',self.batch)<=2 or self.action in ['startup','hardware'] or 'checkpoint' in name
        if announce:self.log_record({'operation':name,'state':'START','deadline_seconds':deadline,**details})
        try:
            yield
            if measured:
                torch.cuda.synchronize()
            if announce:
                record={'operation':name,'state':'END','seconds':round(time.monotonic()-start,4),'GPU_completion_synchronised':measured,**details}
                import resource
                record['process_peak_RSS_GiB']=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,3)
                if self.device=='cuda': record.update(GPU_allocated_GiB=round(torch.cuda.memory_allocated()/2**30,3),GPU_peak_GiB=round(torch.cuda.max_memory_allocated()/2**30,3))
                self.log_record(record)
        except BaseException as ex:
            self.log_record({'operation':name,'state':'FAILED','seconds':round(time.monotonic()-start,4),'error':str(ex),**details}); raise
        finally:
            self.operations.remove(entry);core.write(self.status,{'pid':os.getpid(),'operations':self.operations})

    def completed(self): return len(list((self.work/'models').glob('*/COMPLETE.json')))

    def verify_gradients(self,model):
        if self.action=='smoke' and self.batch<2:
            with self.operation('finite_gradient_check'):
                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                    raise FloatingPointError('Smoke full-batch gradient is nonfinite')

    def checkpoint_due(self):return time.monotonic()-self.last_backup>=self.interval

    def backup(self,reason,force=False):
        if not force and not self.checkpoint_due():return
        with self.operation('durable_backup',deadline=900,reason=reason):
            if self.log.exists():core.atomic(self.work/'EXECUTION_PROGRESS.jsonl',self.log.read_bytes())
            self.backups.save(reason)
        self.last_backup=time.monotonic()

    def progress(self,job,phase,epoch,batch,batches,start,epochs,epoch_elapsed=0.,**extra):
        self.batch=batch; now=time.monotonic()
        if batch not in [1,batches] and now-self.last_progress<10:return
        elapsed=now-start+epoch_elapsed
        rate=extra.get('batch_seconds'); completed=len(list((self.work/'models').glob('*/COMPLETE.json')))
        record={'job':job,'phase':phase,'epoch':epoch,'epoch_cap':epochs,'batch':batch,'batches':batches,
                'elapsed_epoch_seconds':round(elapsed,2),'remaining_FIT_seconds_estimate':None if not rate else round((batches-batch)*rate,1),
                'completed_fits':completed,'planned_fits':22,'verified_backup_age_seconds':round(now-self.last_backup,1),**extra}
        if self.device=='cuda': record.update(GPU_allocated_GiB=round(torch.cuda.memory_allocated()/2**30,3),GPU_reserved_GiB=round(torch.cuda.memory_reserved()/2**30,3),GPU_peak_GiB=round(torch.cuda.max_memory_allocated()/2**30,3))
        self.log_record(record); self.last_progress=now

    def startup(self):
        with self.operation('source_and_environment_verification',deadline=300):
            if self.device=='cuda' and not torch.cuda.is_available():
                raise RuntimeError('Select a CUDA GPU. Memory feasibility is measured at the frozen full batch sizes.')
            core.manifest(core.ROOT,'PACKAGE_MANIFEST.json'); environment=core.configure(self.device)
            if self.device=='cuda' and 'A100' not in environment['GPU']:
                raise RuntimeError('This runner requires the requested A100 device; keep the frozen settings')
        if not self.work.exists() or not any(self.work.iterdir()):self.backups.restore(environment)
        if (self.work/'RUN_FREEZE.json').exists():
            identity=pipeline.verify_run(self.work,self.device)
        else:
            with self.operation('TRAIN_only_stage',deadline=300): _,_,identity=pipeline.initial(self.work,self.device)
        core.freeze(self.work/'EXECUTION_AMENDMENT.json',{
            'amendment':VERSION,'package_manifest_sha256':self.identity['package_manifest_sha256'],
            'native_convolution_checkpoint_continuation_authorised':False})
        # Cache official archives locally without relying on runtime network.
        for name,pin in upstream.PINS.items():
            src=core.ROOT/'inputs'/('pinned_'+name+'.zip'); core.checked(src,pin['sha256'])
            dst=self.work/'official'/(name+'.zip')
            if dst.exists():core.checked(dst,pin['sha256'])
            else:core.atomic(dst,src.read_bytes())
        if (self.work/'STUDY_FREEZE.json').exists():pipeline.verify_study(self.work)
        if (self.work/'scores/SCORE_SEAL.json').exists():pipeline.verify_export(self.work)
        self.backup('startup',force=True)
        return identity

    @contextlib.contextmanager
    def active(self):
        with core.Lock(self.work.parent/(self.work.name+'.lock')):
            if runtime.ACTIVE is not None:raise RuntimeError('Another runner is active')
            runtime.ACTIVE=self
            try:yield
            finally:runtime.ACTIVE=None

    @contextlib.contextmanager
    def scientific_hooks(self):
        # Reuse the already audited score cache; no changes to native arithmetic.
        from execution_scoring import ScoreMemoizer
        saved={k:getattr(pipeline,k) for k in ['select_bank','threshold_all','confirm','seal']}
        score_fn=scoring.scores; pipeline_score=pipeline.scores
        cache=ScoreMemoizer(self.work,score_fn); scoring.scores=cache; pipeline.scores=cache
        def wrap(name,fn):
            def call(*args,**kw):
                result=fn(*args,**kw)
                self.backup('scientific_commit_'+name)
                return result
            return call
        for name,fn in saved.items():setattr(pipeline,name,wrap(name,fn))
        try:yield
        finally:
            for name,fn in saved.items():setattr(pipeline,name,fn)
            scoring.scores=score_fn; pipeline.scores=pipeline_score

    def run(self,action,labels=None):
        with self.active():
            identity=self.startup();self.action=action;self.trace=action=='smoke'
            if action in ['smoke','hardware']:
                if (self.work/'STUDY_FREEZE.json').exists():
                    pipeline.verify_study(self.work);print('Study sealed: no additional fitting or probes',flush=True);return
                with self.operation('load_TRAIN_and_official_sources',deadline=300):
                    raw,meta=data.training_data(self.work); paths=upstream.prepare(self.work)
                if action=='smoke':smoke(self.work,raw,meta,paths,self.device,identity)
                else:
                    gate=core.read(self.work/'SMOKE_GATE.json')
                    if gate['identity']!=identity or gate['status']!='PASS' or self.device=='cuda' and not gate['actual_GPU']:
                        raise RuntimeError('Actual real-epoch smoke required first')
                    for n,h in gate['files'].items():core.checked(self.work/n,h)
                    hardware_gate(self.work,raw,meta,paths,self.device,identity)
                self.backup(action+'_complete',force=True)
            elif action=='develop':
                if not (self.work/'ENGINEERING_GATE.json').exists():raise RuntimeError('Full-batch hardware gate required')
                verify_hardware_gate(self.work,identity)
                with self.scientific_hooks():
                    pipeline.fit_and_select(self.work,self.device);pipeline.verify_study(self.work)
                    self.backup('study_frozen',force=True)
                    pipeline.confirm_all(self.work,self.device);self.backup('synthetic_confirmation',force=True)
                    pipeline.export_test(self.work,self.device);pipeline.verify_export(self.work)
                    self.backup('verified_TEST_score_seal',force=True)
                print('SCORE SEAL VERIFIED. Supply the historical annotation CSV only now.',flush=True)
            elif action=='verify-seal':pipeline.verify_export(self.work);print('SCORE SEAL VERIFIED',flush=True)
            elif action=='evaluate':
                pipeline.verify_export(self.work)
                if labels is None:raise RuntimeError('Annotation path required after seal verification')
                pipeline.evaluate(self.work,labels);pipeline.audit(self.work)
                self.backup('final_report_and_audit',force=True)
                # Keep an easy-to-download final archive, once, after the final audit.
                final=self.cache/'EXP013_E03_final_results.zip';pipeline.snapshot(self.work,final)
                destination=self.drive/'FINAL_RESULTS.zip'
                pending=self.drive/'FINAL_RESULTS.zip.partial'
                shutil.copyfile(final,pending);core.checked(pending,core.sha(final));os.replace(pending,destination)
                core.write(self.drive/'FINAL_RESULTS_RECEIPT.json',{'sha256':core.sha(final),'bytes':final.stat().st_size})
            elif action!='startup':raise ValueError(action)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['startup','smoke','hardware','develop','verify-seal','evaluate'])
    parser.add_argument('--work',type=Path,required=True);parser.add_argument('--drive',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--labels',type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');parser.add_argument('--parent-pid',type=int)
    args=parser.parse_args()
    if args.parent_pid:
        from process_guard import arm_parent_death_guard
        arm_parent_death_guard(args.parent_pid)
    if args.device=='cpu' and not core.read(core.ROOT/'PROTOCOL.json').get('fixture_only'):
        raise RuntimeError('CPU CLI cannot authorise or execute scientific development; use isolated tests')
    runner=Runner(args.work,args.drive,args.cache,args.device)
    runner.run(args.action,args.labels)

if __name__=='__main__': main()
