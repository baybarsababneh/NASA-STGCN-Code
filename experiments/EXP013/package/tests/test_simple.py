"""Targeted E02 regressions. All arrays/events are fictional, CPU only."""
import ast, copy, importlib.util, json, shutil, sys, tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch, numpy as np
import core, train, data, pipeline, upstream, execution_runtime as runtime
from execution_checkpoint import load_torch
from simple_backup import Backup, EnvironmentMismatch
from simple_run import Runner
from simple_gate import smoke, probe, exact, STATE_KEYS
from simple_launch import launch, expired_operation

PATHS={}
def fictional(mission='MSL'):
    raw={k:np.random.default_rng(i+83).normal(0,.3,(480,6)).astype(np.float32) for i,k in enumerate(['f_a','f_b'])}
    meta={k:{'mission':mission,'rows':480,'columns':6,'fit_end':288,'stop_end':384,'cal_end':432} for k in raw}
    return raw,meta

REAL_FACTORY=train.model_for
def tiny(role,mission,channels,nf,paths,device):
    if role=='catch':
        with torch.device('meta'):m,c=REAL_FACTORY(role,mission,channels,nf,paths,device)
        cls=type(m);del m;c=copy.deepcopy(c)
        c['config'].update(cf_dim=4,d_model=4,d_ff=8,e_layers=1,n_heads=1,head_dim=4,batch_size=16,num_epochs=2)
        m=cls(SimpleNamespace(**c['config']))
    else:m,c=REAL_FACTORY(role,mission,channels,nf,paths,device);c=copy.deepcopy(c)
    c.update(epochs=2,patience=20,batch_size=16)
    return m,c

class SimpleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):core.configure('cpu')

    def backup(self,p):
        w=p/'work';w.mkdir();core.write(w/'RUN_FREEZE.json',{'environment':{'GPU_total_bytes':42405855232}})
        core.write(w/'state.json',{'epoch':1});(w/'last.pt').write_bytes(b'fictional checkpoint 1')
        return Backup(w,p/'drive',p/'cache',{'source':'fixture'}),w

    def test_incremental_checkpoint_reuse_two_generations(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);b,w=self.backup(p);a=b.save('first');core.write(w/'state.json',{'epoch':2});c=b.save('second')
            self.assertEqual(a['objects']['checkpoints'],c['objects']['checkpoints'])
            core.write(w/'state.json',{'epoch':3});b.save('third')
            self.assertEqual(len(list((p/'drive/commits').iterdir())),2)
            shutil.rmtree(w);self.assertTrue(b.restore({'GPU_total_bytes':42405855232}));self.assertEqual(core.read(w/'state.json')['epoch'],3)

    def test_interrupted_upload_keeps_previous_commit(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);b,w=self.backup(p);b.save('first');(w/'last.pt').write_bytes(b'new checkpoint 2')
            original=shutil.copyfileobj
            def interrupted(src,dst,*args):dst.write(src.read(7));raise OSError('simulated interrupted Drive copy')
            with patch.object(shutil,'copyfileobj',side_effect=interrupted):
                with self.assertRaises(OSError):b.save('interrupted')
            shutil.rmtree(w);b.restore({'GPU_total_bytes':42405855232});self.assertEqual((w/'last.pt').read_bytes(),b'fictional checkpoint 1')

    def test_corrupt_latest_rejected_older_verified_generation_restored(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);b,w=self.backup(p);b.save('first');(w/'last.pt').write_bytes(b'checkpoint 2');r=b.save('second')
            (p/'drive/objects'/r['objects']['checkpoints']['last.pt']).write_bytes(b'corruption')
            shutil.rmtree(w);self.assertTrue(b.restore({'GPU_total_bytes':42405855232}));self.assertEqual((w/'last.pt').read_bytes(),b'fictional checkpoint 1')

    def test_80GB_40GB_mismatch_rejected_without_identity_rewrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);b,w=self.backup(p);core.write(w/'RUN_FREEZE.json',{'environment':{'GPU_total_bytes':85094825984}});b.save('80GB')
            commits={f:core.sha(f) for f in (p/'drive').rglob('*') if f.is_file()};shutil.rmtree(w)
            with self.assertRaises(EnvironmentMismatch):b.restore({'GPU_total_bytes':42405855232})
            self.assertFalse(w.exists());self.assertEqual(commits,{f:core.sha(f) for f in (p/'drive').rglob('*') if f.is_file()})

    def test_source_identity_mismatch_and_partial_commit_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);b,w=self.backup(p);b.save('first');shutil.rmtree(w)
            bad=Backup(w,p/'drive',p/'cache2',{'source':'wrong'})
            with self.assertRaises(RuntimeError):bad.restore({'GPU_total_bytes':42405855232})
            pending=p/'drive/commits/incomplete.partial';pending.mkdir();(pending/'manifest.json').write_text('{}')
            self.assertTrue(b.restore({'GPU_total_bytes':42405855232}))

    def test_complete_VM_reset_exact_resume_and_completed_reuse(self):
        raw,meta=fictional()
        with tempfile.TemporaryDirectory() as td,patch.object(train,'model_for',tiny):
            p=Path(td);w=p/'local/work';w.mkdir(parents=True);core.write(w/'RUN_FREEZE.json',{'environment':core.configure('cpu')})
            runner=Runner(w,p/'drive',p/'cache','cpu');identity={'fictional':True}
            with runner.active():train.train_job(w,raw,meta,'MSL','graph',17,PATHS,'cpu',identity,interrupt_after_batches=2)
            shutil.rmtree(p/'local');shutil.rmtree(p/'cache')
            fresh=Runner(w,p/'drive',p/'new_cache','cpu');self.assertTrue(fresh.backups.restore(core.configure('cpu')))
            resumed=train.train_job(w,raw,meta,'MSL','graph',17,PATHS,'cpu',identity)
            whole=train.train_job(p/'whole',raw,meta,'MSL','graph',17,PATHS,'cpu',identity)
            self.assertEqual(resumed['state_sha256'],whole['state_sha256'])
            with patch.object(train,'validate',side_effect=AssertionError('completed fit retrained')):
                self.assertEqual(train.train_job(w,raw,meta,'MSL','graph',17,PATHS,'cpu',identity)['state_sha256'],whole['state_sha256'])

    def test_real_path_smoke_completes_epoch_replays_and_retains_next_batch(self):
        raw,meta=fictional()
        with tempfile.TemporaryDirectory() as td,patch.object(train,'model_for',tiny):
            p=Path(td);identity={'fixture':True};w=p/'work'
            record=smoke(w,raw,meta,PATHS,'cpu',identity)
            state=load_torch(w/'models/MSL__graph__17/last.pt')
            self.assertEqual((state['epoch'],state['phase'],state['cursor']),(1,'fit',1))
            self.assertEqual(len(state['history']),1);self.assertTrue(record['full_real_epoch_completed'])
            with patch.object(train,'train_job',side_effect=AssertionError('smoke repeated')):smoke(w,raw,meta,PATHS,'cpu',identity)
            train.train_job(w,raw,meta,'MSL','graph',17,PATHS,'cpu',identity,epoch_limit=2)
            train.train_job(p/'whole',raw,meta,'MSL','graph',17,PATHS,'cpu',identity,epoch_limit=2)
            a=load_torch(w/'models/MSL__graph__17/last.pt');b=load_torch(p/'whole/models/MSL__graph__17/last.pt')
            for k in STATE_KEYS:self.assertTrue(exact(a.get(k),b.get(k)),k)

    def test_smoke_replay_interruption_reuses_full_epoch(self):
        raw,meta=fictional()
        with tempfile.TemporaryDirectory() as td,patch.object(train,'model_for',tiny):
            p=Path(td);original=train.train_job;calls=[]
            def interrupt(*args,**kw):
                calls.append(kw)
                if len(calls)==3:raise RuntimeError('simulated replay interruption')
                return original(*args,**kw)
            with patch.object(train,'train_job',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'replay interruption'):smoke(p/'w',raw,meta,PATHS,'cpu',{'fixture':True})
            epoch_hash=core.sha(p/'w/engineering/smoke/epoch1.pt')
            smoke(p/'w',raw,meta,PATHS,'cpu',{'fixture':True})
            self.assertEqual(core.sha(p/'w/engineering/smoke/epoch1.pt'),epoch_hash)

    def test_smoke_adopts_saved_complete_epoch_without_repeating_epoch(self):
        raw,meta=fictional()
        with tempfile.TemporaryDirectory() as td,patch.object(train,'model_for',tiny):
            w=Path(td)/'w';identity={'fixture':True}
            train.train_job(w,raw,meta,'MSL','graph',17,PATHS,'cpu',identity,epoch_limit=1)
            original=train.train_job;calls=[]
            def resumed(*a,**kw):
                self.assertNotIn('epoch_limit',kw);calls.append(kw);return original(*a,**kw)
            with patch.object(train,'train_job',side_effect=resumed):smoke(w,raw,meta,PATHS,'cpu',identity)
            self.assertEqual(len(calls),2)
            self.assertTrue(all(v['interrupt_after_batches']==1 for v in calls))

    def test_full_batch_probe_exact_resume_all_five_roles(self):
        raw,meta=fictional('SMAP')
        with tempfile.TemporaryDirectory() as td,patch.object(train,'model_for',tiny):
            for role in train.NEW_ROLES+['gdn','catch']:
                r=probe(Path(td)/'work',raw,meta,PATHS,'SMAP',role,'cpu')
                self.assertTrue(r['checkpoint_replay_exact']);self.assertEqual(r['batch_size'],16)
                self.assertEqual(r['CATCH_carried_gradient_mask_step_tested'],role=='catch')

    def test_corrected_role_native_comparison_all_roles(self):
        from convolution_regression import validate_role
        raw,meta=fictional('SMAP')
        with patch.object(train,'model_for',tiny):
            for role in train.NEW_ROLES:
                result=validate_role(raw,meta,PATHS,'SMAP',role,'cpu')
                self.assertEqual(result['status'],'PASS')
                self.assertTrue(result['corrected_replay_bitwise_equal'])
                self.assertFalse(result['actual_GPU'])
    def test_corrected_role_missing_numerical_evidence_blocks_training(self):
        from simple_gate import verify_hardware_gate
        with tempfile.TemporaryDirectory() as td:
            w=Path(td);identity={'environment':{'device':'cpu'},'package_manifest_sha256':'fixture'}
            g={'status':'PASS',**identity,'full_batches':[{'mission':m,'role':r,'identity':identity,'status':'PASS','batch_size':256 if r in train.NEW_ROLES else 32 if r=='gdn' else 128,'checkpoint_replay_exact':True} for m in ['MSL','SMAP'] for r in train.NEW_ROLES+['gdn','catch']]}
            core.write(w/'ENGINEERING_GATE.json',g)
            with self.assertRaisesRegex(RuntimeError,'Corrected-role'):verify_hardware_gate(w,identity)
    def test_scientific_factory_schedule_and_sources_unchanged(self):
        parent=core.ROOT/'frozen_originals/train.py'
        a=ast.parse(parent.read_text());b=ast.parse((core.ROOT/'src/train.py').read_text())
        for name in ['jobs','model_for','job_name']:
            fn=lambda tree:next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name==name)
            self.assertEqual(ast.dump(fn(a),include_attributes=False),ast.dump(fn(b),include_attributes=False))
        self.assertEqual(len(train.jobs()),22)

    def test_failed_seal_blocks_even_label_lookup_in_runner(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);r=Runner(p/'w',p/'d',p/'c','cpu')
            with patch.object(r,'startup',return_value={}),patch.object(pipeline,'verify_export',side_effect=RuntimeError('bad seal')),patch.object(pipeline,'evaluate') as evaluate:
                with self.assertRaisesRegex(RuntimeError,'bad seal'):r.run('evaluate',p/'never_open')
                evaluate.assert_not_called()

    def test_stage_order_and_persistent_commits(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);r=Runner(p/'w',p/'d',p/'c','cpu');core.write(r.work/'ENGINEERING_GATE.json',{});order=[]
            with patch.object(r,'startup',return_value={}),patch('simple_run.verify_hardware_gate'),patch.object(r,'backup',side_effect=lambda reason,**kw:order.append(reason)),patch.object(pipeline,'fit_and_select',side_effect=lambda *a:order.append('select')),patch.object(pipeline,'verify_study'),patch.object(pipeline,'confirm_all',side_effect=lambda *a:order.append('CONF')),patch.object(pipeline,'export_test',side_effect=lambda *a:order.append('TEST')),patch.object(pipeline,'verify_export',side_effect=lambda *a:order.append('seal')),patch.object(pipeline,'read_labels') as labels:
                r.run('develop');labels.assert_not_called()
            self.assertEqual(order,['select','study_frozen','CONF','synthetic_confirmation','TEST','seal','verified_TEST_score_seal'])

    def test_hardware_failure_cannot_grant_gate(self):
        from simple_gate import hardware_gate
        raw,meta=fictional()
        with tempfile.TemporaryDirectory() as td,patch('simple_gate.workload',return_value=[{'mission':'MSL','role':'catch','batch_size':128,'parameters':210881152}]),patch('simple_gate.probe',side_effect=torch.OutOfMemoryError('fixture capacity failure')):
            w=Path(td)
            with self.assertRaises(torch.OutOfMemoryError):hardware_gate(w,raw,meta,PATHS,'cpu',{})
            self.assertFalse((w/'ENGINEERING_GATE.json').exists())

    def test_launcher_timeout_and_SIGKILL_no_gate(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(RuntimeError,'Bounded invocation'):
                launch([sys.executable,'-u','-c','import time; print("started",flush=True); time.sleep(30)'],td,total_timeout=.2)
            with self.assertRaisesRegex(RuntimeError,'Cause is not inferred'):
                launch([sys.executable,'-u','-c','import os,signal; os.kill(os.getpid(),signal.SIGKILL)'],td,total_timeout=5)

    def test_nested_deadlines_and_foreign_pid_ignored(self):
        status={'pid':2,'operations':[{'name':'epoch','start_monotonic':0,'deadline':20},{'name':'forward','start_monotonic':9,'deadline':10}]}
        self.assertEqual(expired_operation(status,2,21)['name'],'epoch');self.assertIsNone(expired_operation(status,3,999))

    def test_operation_watchdog_terminates_blocked_worker(self):
        with tempfile.TemporaryDirectory() as td:
            status=Path(td)/'operation.json'
            script="import os,json,time,sys; open(sys.argv[1],'w').write(json.dumps({'pid':os.getpid(),'operations':[{'name':'forward','start_monotonic':time.monotonic(),'deadline':.1}]})); print('forward start',flush=True); time.sleep(30)"
            with self.assertRaisesRegex(RuntimeError,'Operation deadline exceeded: forward'):
                launch([sys.executable,'-u','-c',script,str(status)],td,total_timeout=10)

    def test_partial_hardware_gate_and_changed_batch_rejected(self):
        from simple_gate import verify_hardware_gate
        with tempfile.TemporaryDirectory() as td:
            w=Path(td);identity={'environment':{'device':'cpu'},'package_manifest_sha256':'fixture'}
            g={'status':'PASS',**identity,'full_batches':[]}
            core.write(w/'ENGINEERING_GATE.json',g)
            with self.assertRaisesRegex(RuntimeError,'All ten'):verify_hardware_gate(w,identity)
            g['full_batches']=[{'mission':m,'role':r,'identity':identity,'status':'PASS','batch_size':256 if r in train.NEW_ROLES else 32 if r=='gdn' else 128,'checkpoint_replay_exact':True} for m in ['MSL','SMAP'] for r in train.NEW_ROLES+['gdn','catch']]
            for r in g['full_batches']:
                if r['role'] in train.NEW_ROLES:r['convolution_regression']={'status':'PASS','batch_size':256,'corrected_replay_bitwise_equal':True,'actual_GPU':False}
            g['full_batches'][-1]['batch_size']=64;core.write(w/'ENGINEERING_GATE.json',g)
            with self.assertRaisesRegex(RuntimeError,'scientific settings'):verify_hardware_gate(w,identity)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--official-work',type=Path,required=True);a=p.parse_args();PATHS=upstream.prepare(a.official_work)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(SimpleTests))
    raise SystemExit(0 if result.wasSuccessful() else 1)
