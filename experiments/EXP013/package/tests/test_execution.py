"""E01 synthetic interruption, source equivalence and durable recovery tests."""
import argparse,copy,importlib.util,json,os,shutil,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch,numpy as np
from torch.utils.data import DataLoader,WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
import core,data,train,upstream,execution_runtime,pipeline
from execution_data import gather,epoch_plan
from execution_checkpoint import save_torch,load_torch
from recovery import Recovery,LocalTransport,IncompatibleRecovery
from core import configure,rng,restore_rng,seeds,state_hash,write,read,sha,ROOT

DEVICE='cpu';PATHS={}
spec=importlib.util.spec_from_file_location('frozen_original_train',ROOT/'frozen_originals/train.py')
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)

def fictional(nf=6,mission='SMAP',T=480):
    gen=np.random.default_rng(333)
    raw={c:gen.normal(0,.3,(T,nf)).astype(np.float32) for c in ['fake_a','fake_b']}
    meta={c:{'mission':mission,'rows':T,'columns':nf,'fit_end':int(.6*T),'stop_end':int(.8*T),'cal_end':int(.9*T)} for c in raw}
    return raw,meta

REAL_MODEL_FOR=train.model_for
def tiny(role,mission,channels,nf,paths,device):
    model,cfg=REAL_MODEL_FOR(role,mission,channels,nf,paths,device);cfg=copy.deepcopy(cfg)
    cfg.update(epochs=2,patience=20,batch_size=16)
    if role=='catch':
        cfg['config'].update(cf_dim=4,d_model=4,d_ff=8,e_layers=1,n_heads=1,head_dim=4,batch_size=16,num_epochs=2)
        cls=type(model);del model;model=cls(SimpleNamespace(**cfg['config']))
    return model,cfg

def exact(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure(DEVICE)

    def test_kernel_death_terminates_hash_seed_pinned_worker(self):
        with tempfile.TemporaryDirectory() as td:
            ready=Path(td)/'worker_ready.json'
            worker="import os,sys,time,json; from process_guard import arm_parent_death_guard; arm_parent_death_guard(int(sys.argv[1])); open(sys.argv[2],'w').write(json.dumps({'pid':os.getpid(),'hashseed':os.environ.get('PYTHONHASHSEED')})); time.sleep(60)"
            launcher=("import os,sys,subprocess,time; from pathlib import Path; "
                "p=subprocess.Popen([sys.executable,'-c',sys.argv[1],str(os.getpid()),sys.argv[2]],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                "deadline=time.monotonic()+8\n"
                "while not Path(sys.argv[2]).exists() and time.monotonic()<deadline:time.sleep(.02)\n"
                "assert Path(sys.argv[2]).exists(); os._exit(0)")
            subprocess.run([sys.executable,'-c',launcher,worker,str(ready)],check=True,timeout=12,
                env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'PYTHONHASHSEED':'17'})
            details=read(ready);self.assertEqual(details['hashseed'],'17')
            status=Path('/proc')/str(details['pid'])/'stat';deadline=time.monotonic()+5
            while status.exists() and status.read_text().split()[2]!='Z' and time.monotonic()<deadline:time.sleep(.05)
            self.assertTrue(not status.exists() or status.read_text().split()[2]=='Z','Orphan computation survived its owning kernel')

    def test_vectorised_batch_bytes_all_roles_and_native_dimensions(self):
        for mission,nf in [('MSL',55),('SMAP',25)]:
            raw,meta=fictional(nf,mission)
            for role in ['graph','removed','fusion','gdn','catch']:
                ds=data.Windows(raw,meta,sorted(raw),role,'fit')
                indices=[len(ds)-1,0,1,len(ds)//2,0]
                self.assertTrue(exact(gather(ds,indices),tuple(default_collate([ds[i] for i in indices]))))

    def test_original_sampler_order_and_generator_states(self):
        raw,meta=fictional();ds=data.Windows(raw,meta,sorted(raw),'graph')
        for weighted in [False,True]:
            def loader():
                g=torch.Generator().manual_seed(43)
                sampler=WeightedRandomSampler([1/len(ds)]*len(ds),len(ds),replacement=True,generator=g) if weighted else None
                return DataLoader(ds,batch_size=17,shuffle=not weighted,sampler=sampler,generator=g,num_workers=0),g
            a,ga=loader();b,gb=loader();before=rng()
            expected=[tuple(v.clone() for v in batch) for batch in a];global_after=rng()
            restore_rng(before);plan=epoch_plan(b);actual=[gather(ds,ix) for ix in plan]
            self.assertTrue(exact(expected,actual));self.assertTrue(torch.equal(ga.get_state(),gb.get_state()));self.assertTrue(exact(global_after,rng()))

    def test_checkpoint_format_matches_original_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            a=Path(td)/'a.pt';b=Path(td)/'b.pt';value={'t':torch.arange(71),'rng':rng()}
            core.save_torch(a,value);save_torch(b,value)
            self.assertEqual(a.read_bytes(),b.read_bytes());self.assertTrue(exact(load_torch(b),value))
            payload=bytearray(b.read_bytes());payload[-15]^=1;b.write_bytes(payload)
            with self.assertRaises(RuntimeError):load_torch(b)

    def test_uninterrupted_training_matches_frozen_original_all_five_roles(self):
        raw,meta=fictional()
        with tempfile.TemporaryDirectory() as td,patch.object(train,'model_for',tiny),patch.object(original,'model_for',tiny):
            for role in ['graph','removed','fusion','gdn','catch']:
                a=Path(td)/('original_'+role);b=Path(td)/('amended_'+role);identity={'fixture_only':True}
                oa=original.train_job(a,raw,meta,'SMAP',role,17,PATHS,DEVICE,identity)
                ob=train.train_job(b,raw,meta,'SMAP',role,17,PATHS,DEVICE,identity)
                self.assertEqual(oa['state_sha256'],ob['state_sha256'],role);self.assertEqual(oa['epoch'],ob['epoch'],role)
                pa=a/'models'/('SMAP__'+role+'__17')
                pb=b/'models'/('SMAP__'+role+'__17')
                self.assertEqual((pa/'selected.pt').read_bytes(),(pb/'selected.pt').read_bytes(),role)
                aa=read(pa/'history.json');bb=read(pb/'history.json')
                for h in aa+bb:h.pop('seconds')
                self.assertEqual(aa,bb,role)

    def test_exact_fit_and_STOP_resume_all_roles_including_optimizer_RNG_mask(self):
        raw,meta=fictional()
        with tempfile.TemporaryDirectory() as td,patch.object(train,'model_for',tiny):
            for role in ['graph','removed','fusion','gdn','catch']:
                for phase in ['fit','stop']:
                    a=Path(td)/('whole_'+role+phase);b=Path(td)/('resume_'+role+phase);identity={'fixture_only':True}
                    train.train_job(a,raw,meta,'SMAP',role,17,PATHS,DEVICE,identity,epoch_limit=2)
                    result=train.train_job(b,raw,meta,'SMAP',role,17,PATHS,DEVICE,identity,interrupt_after_batches=2,interrupt_phase=phase)
                    self.assertEqual(result['status'],'PAUSED_FOR_TEST')
                    train.train_job(b,raw,meta,'SMAP',role,17,PATHS,DEVICE,identity,epoch_limit=2)
                    aa=load_torch(a/'models'/('SMAP__'+role+'__17')/'last.pt');bb=load_torch(b/'models'/('SMAP__'+role+'__17')/'last.pt')
                    for key in ['model','optimizer','mask_optimizer','mask_grads','schedulers','rng','loader_rng','best','bad','epoch','phase']:
                        self.assertTrue(exact(aa[key],bb[key]),(role,phase,key))
                    for h in aa['history']+bb['history']:h.pop('seconds')
                    self.assertEqual(aa['history'],bb['history'],(role,phase))

    def test_mixed_mission_training_scaler_receives_only_selected_mission(self):
        raw,meta=fictional(55,'MSL');r2,m2=fictional(25,'SMAP');r2={'other_'+k:v for k,v in r2.items()};m2={'other_'+k:v for k,v in m2.items()};raw.update(r2);meta.update(m2)
        for role in ['gdn','catch']:
            recorded={}
            def inspect(values,scaler):
                recorded['keys']=set(values);transformed=data.transform(values,scaler)
                self.assertTrue(all(v.shape[1]==55 for v in transformed.values()))
                raise RuntimeError('fixture reached verified mission-only transform')
            with tempfile.TemporaryDirectory() as td,patch.object(train,'model_for',return_value=(torch.nn.Linear(55,1),{'fixture_only':True})),patch.object(train,'transform',side_effect=inspect):
                with self.assertRaisesRegex(RuntimeError,'fixture reached'):train.train_job(td,raw,meta,'MSL',role,17,PATHS,DEVICE,{'fixture_only':True})
                self.assertEqual(recorded['keys'],{'fake_a','fake_b'})

    def test_interrupted_upload_retains_previous_and_rejects_partial(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);w=p/'work';w.mkdir();write(w/'state.json',{'epoch':1})
            backend=LocalTransport(p/'drive');r=Recovery(backend,w,p/'cache',{'fixture':1})
            good,_=r.backup('first');write(w/'state.json',{'epoch':2});backend.fail_after=1
            with self.assertRaises(OSError):r.backup('interrupted')
            self.assertEqual(r.latest()[0]['id'],good['id']);self.assertEqual(len(r.candidates()),1)
            backend.fail_after=None;r.backup('retry');self.assertEqual(len(r.candidates()),2)

    def test_corrupt_archive_or_receipt_rejected_then_verified_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);w=p/'work';w.mkdir();write(w/'state.json',{'epoch':1});backend=LocalTransport(p/'drive');r=Recovery(backend,w,p/'cache',{'fixture':1})
            first,_=r.backup('first');write(w/'state.json',{'epoch':2});latest,receipt=r.backup('second')
            archive=p/'drive'/(receipt['objects'][0]['remote_id']+'.bin');archive.write_bytes(archive.read_bytes()+b'corruption')
            self.assertEqual(r.latest()[0]['id'],first['id'])
            oldbytes=(p/'drive'/(first['id']+'.bin')).read_bytes();(p/'drive'/(first['id']+'.bin')).write_bytes(oldbytes+b'corruption')
            with self.assertRaises(RuntimeError):r.latest()

    def test_interrupted_commit_and_ambiguous_completed_upload(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);w=p/'work';w.mkdir();write(w/'state.json',{'epoch':1});backend=LocalTransport(p/'drive');r=Recovery(backend,w,p/'cache',{'fixture':1})
            good,_=r.backup('first');write(w/'state.json',{'epoch':2});backend.fail_after=1;backend.fail_kind='commit'
            with self.assertRaises(OSError):r.backup('incomplete_receipt')
            self.assertEqual(r.latest()[0]['id'],good['id'])
            backend.fail_after=None;real=backend.finalize
            def ambiguous(identifier,name):
                value=real(identifier,name)
                if value['properties']['kind']=='commit':raise OSError('Lost HTTP response after server completed commit')
                return value
            with patch.object(backend,'finalize',side_effect=ambiguous):
                with self.assertRaises(OSError):r.backup('ambiguous_completion')
            self.assertNotEqual(r.latest()[0]['id'],good['id'])
            shutil.rmtree(w);self.assertTrue(r.restore());self.assertEqual(read(w/'state.json'),{'epoch':2})

    def test_incremental_large_object_reuse_and_hash_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);w=p/'work';w.mkdir();write(w/'state.json',{'epoch':1})
            with (w/'selected.pt').open('wb') as f:
                for _ in range(33):f.write(bytes(range(256))*4096)
            backend=LocalTransport(p/'drive');r=Recovery(backend,w,p/'cache',{'fixture':1})
            first,a=r.backup('first');large=next(v for v in a['objects'] if 'selected.pt' in v['members'])
            write(w/'state.json',{'epoch':2});second,b=r.backup('metadata_only_update')
            repeated=next(v for v in b['objects'] if 'selected.pt' in v['members'])
            self.assertEqual(large,repeated);self.assertLess(b['uploaded_bytes'],2**20)
            before=sha(w/'selected.pt');shutil.rmtree(w);self.assertTrue(r.restore());self.assertEqual(sha(w/'selected.pt'),before)

    def test_verified_seal_marker_repair_never_precedes_verification(self):
        from execution import Runner
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);w=p/'work';w.mkdir();backend=LocalTransport(p/'drive');runner=Runner(w,p/'cache',backend,device=DEVICE)
            with patch.object(pipeline,'verify_export',side_effect=RuntimeError('corrupt scores')):
                with self.assertRaises(RuntimeError):runner.verify_score_seal()
                self.assertFalse((w/'ACCESS.jsonl').exists())
            with patch.object(pipeline,'verify_export',return_value={'verified':True}) as verify:
                runner.verify_score_seal();runner.verify_score_seal();self.assertEqual(verify.call_count,2)
            lines=[json.loads(v) for v in (w/'ACCESS.jsonl').read_text().splitlines()]
            self.assertEqual(len(lines),1);self.assertEqual(lines[0]['action'],'NASA_SCORE_SEAL_VERIFIED')

    def test_complete_VM_reset_exact_training_restore_and_completed_job_reuse(self):
        raw,meta=fictional()
        with tempfile.TemporaryDirectory() as td,patch.object(train,'model_for',tiny):
            p=Path(td);w=p/'local/work';w.mkdir(parents=True);write(w/'RUN_FREEZE.json',{'environment':configure(DEVICE)})
            backend=LocalTransport(p/'drive');recovery=Recovery(backend,w,p/'local/cache',{'fixture':1})
            class Active:
                def checkpoint_due(self):return False
                def progress(self,*a,**kw):pass
                def backup(self,reason,force=False):
                    if force:recovery.backup(reason)
            execution_runtime.ACTIVE=Active()
            try:train.train_job(w,raw,meta,'SMAP','graph',17,PATHS,DEVICE,{'fixture_only':True},interrupt_after_batches=2)
            finally:execution_runtime.ACTIVE=None
            shutil.rmtree(p/'local')  # Only remote objects survive VM destruction.
            restored=Recovery(backend,w,p/'new_cache',{'fixture':1});self.assertTrue(restored.restore(configure(DEVICE)))
            result=train.train_job(w,raw,meta,'SMAP','graph',17,PATHS,DEVICE,{'fixture_only':True})
            whole=train.train_job(p/'whole',raw,meta,'SMAP','graph',17,PATHS,DEVICE,{'fixture_only':True})
            self.assertEqual(result['state_sha256'],whole['state_sha256'])
            selected=w/'models/SMAP__graph__17/selected.pt';before=sha(selected)
            with patch.object(train,'validate',side_effect=AssertionError('completed fit was retrained')):
                train.train_job(w,raw,meta,'SMAP','graph',17,PATHS,DEVICE,{'fixture_only':True})
            self.assertEqual(before,sha(selected))

    def test_source_and_environment_compatibility_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);w=p/'work';w.mkdir();write(w/'RUN_FREEZE.json',{'environment':{'torch':'frozen'}})
            backend=LocalTransport(p/'drive');r=Recovery(backend,w,p/'cache',{'source':'frozen'});r.backup('valid')
            with self.assertRaises(IncompatibleRecovery):Recovery(backend,p/'empty',p/'cache2',{'source':'edited'}).latest()
            with self.assertRaises(IncompatibleRecovery):r.latest({'torch':'different'})
            with self.assertRaises(RuntimeError):r.restore()

    def test_retention_preserves_two_verified_states_and_shared_objects(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);w=p/'work';w.mkdir();backend=LocalTransport(p/'drive');r=Recovery(backend,w,p/'cache',{'fixture':1})
            for i in range(4):write(w/'state.json',{'epoch':i});r.backup('epoch'+str(i))
            self.assertEqual(len(r.candidates()),2)
            for remote in r.candidates():r._read_record(remote)
            shutil.rmtree(w);self.assertTrue(r.restore());self.assertEqual(read(w/'state.json'),{'epoch':3})

    def test_preserved_fit_STOP_CONF_TEST_and_label_gate(self):
        raw,meta=fictional()
        for role in ['graph','removed','fusion','gdn','catch']:
            for phase in ['fit','stop']:
                ds=data.Windows(raw,meta,sorted(raw),role,phase)
                for ch,t,e in ds.rows:
                    self.assertLess(t,meta[ch]['fit_end'] if phase=='fit' else meta[ch]['stop_end'])
                    self.assertLess(t,meta[ch]['cal_end'])
        with tempfile.TemporaryDirectory() as td,patch.object(pipeline,'verify_export',side_effect=RuntimeError('seal absent')),patch.object(pipeline,'read_labels') as parse:
            with self.assertRaises(RuntimeError):pipeline.evaluate(Path(td),Path(td)/'NASA_LABELS_NEVER_OPEN')
            parse.assert_not_called()

    def test_score_call_cache_exact_values_RNG_and_future_isolation(self):
        from execution_scoring import ScoreMemoizer
        import scoring
        raw,meta=fictional()
        with tempfile.TemporaryDirectory() as td:
            w=Path(td);write(w/'DATA_FREEZE.json',{'fixture_only':True})
            for role in ['graph','gdn','catch']:
                model,cfg=tiny(role,'SMAP',sorted(raw),6,PATHS,DEVICE);model.to(DEVICE)
                model._execution_selected_identity={'job':role,'state_sha256':state_hash(model),'checkpoint_sha256':'fixture'}
                look=64 if role=='graph' else 5 if role=='gdn' else 191
                a=raw['fake_a'];start=look;end=look+37
                expected=scoring.scores(model,role,a,0,start,end,DEVICE,PATHS,cfg,29)
                saved=rng();memo=ScoreMemoizer(w,scoring.scores)
                actual=memo(model,role,a,0,start,end,DEVICE,PATHS,cfg,29)
                self.assertTrue(np.array_equal(expected,actual));self.assertTrue(exact(saved,rng()))
                b=a.copy();b[end:]+=999
                with patch.object(memo,'original',side_effect=AssertionError('completed score call repeated')):
                    repeated=memo(model,role,b,0,start,end,DEVICE,PATHS,cfg,29)
                self.assertTrue(np.array_equal(expected,repeated));self.assertTrue(exact(saved,rng()))
            corrupt=next((w/'score_call_cache').glob('*/values.npz'));corrupt.write_bytes(corrupt.read_bytes()+b'tamper')
            # Direct manifest verification rejects even ZIP formats that tolerate
            # trailing bytes; the scientific wrapper performs this same check.
            with self.assertRaises(RuntimeError):core.manifest(corrupt.parent,'COMMIT.json')

    def test_orchestrator_orders_study_CONFIRM_TEST_and_label_gate(self):
        from execution import Runner
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);w=p/'work';w.mkdir();runner=Runner(w,p/'cache',LocalTransport(p/'drive'),device=DEVICE)
            env=configure(DEVICE);identity={'status':'PASS','environment':env,'package_manifest_sha256':sha(ROOT/'PACKAGE_MANIFEST.json'),'actual_GPU_checks_run':DEVICE=='cuda'}
            write(w/'E01_ENGINEERING_GATE.json',identity);write(w/'PERFORMANCE_PREFLIGHT.json',identity)
            order=[]
            def export(*a,**kw):order.append('TEST');write(w/'scores/SCORE_SEAL.json',{'fixture_only':True})
            with patch.object(pipeline,'fit_and_select',side_effect=lambda *a,**k:order.append('study')),patch.object(pipeline,'verify_study',return_value={}),patch.object(pipeline,'confirm_all',side_effect=lambda *a,**k:order.append('CONF')),patch.object(pipeline,'export_test',side_effect=export),patch.object(pipeline,'verify_export',side_effect=lambda *a,**k:order.append('verified_seal') or {}),patch.object(pipeline,'read_labels') as labels,patch.object(runner,'backup',side_effect=lambda reason,force=False:order.append(reason)):
                runner.develop_and_seal();labels.assert_not_called()
            self.assertLess(order.index('study'),order.index('study_frozen'))
            self.assertLess(order.index('study_frozen'),order.index('CONF'))
            self.assertLess(order.index('confirmation_complete'),order.index('TEST'))
            self.assertLess(order.index('verified_seal'),order.index('verified_score_seal'))
            self.assertEqual(runner.phase,'AWAITING_LABEL_CSV_AFTER_VERIFIED_SEAL')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu',choices=['cpu','cuda']);parser.add_argument('--official-work',type=Path,required=True);args,rest=parser.parse_known_args()
    DEVICE=args.device;PATHS=upstream.prepare(args.official_work)
    if rest:raise RuntimeError('Unexpected E01 test arguments: '+str(rest))
    # An explicit suite also works when Colab runs this file inside its kernel
    # with runpy; unittest.main() would inspect the notebook's __main__ module.
    suite=unittest.TestLoader().loadTestsFromTestCase(ExecutionTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
