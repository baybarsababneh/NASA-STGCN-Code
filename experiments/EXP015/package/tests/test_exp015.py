import os,sys,io,time,tempfile,shutil,zipfile,subprocess
from pathlib import Path
import unittest
import numpy as np
import torch
from integrity import ROOT,read,write,sha,blob,files,seal,manifest,arrays,commit,committed,discard_uncommitted,rng,restore_rng,rng_equal,state_hash,LabelGuard
from inputs import Source,required_paths,recover
from backup import Backup,EnvironmentMismatch
from checkpoint import save_torch,load_torch
from inference import transform,inverse_reference,gather,ewma,host_budget
from vendor.exp013.architectures import CommandReferenceGCN
from vendor.frozen import metrics
from vendor.frozen.hundman_detector import detect

class ScientificTests(unittest.TestCase):
    measurements=[]

    def test_exact_frozen_sources_and_input_roster(self):
        c=read(ROOT/'INPUT_CONTRACT.json')
        for name,key in [('vendor/frozen/hundman_detector.py','frozen_detector_sha256'),('vendor/frozen/metrics.py','frozen_evaluator_sha256'),('vendor/exp013/architectures.py','frozen_architecture_sha256')]:self.assertEqual(sha(ROOT/name),c[key])
        self.assertEqual(len(required_paths(ROOT)),2260)
        self.assertEqual(sum(n.endswith('selected.pt') for n in c['known_required_files']),18)
        self.assertEqual(c['frozen_detector_sha256'],'5b238ae7226a60e092b4d0a0954ca276733f3f9a27c0766d98282512dc692714')
        self.assertEqual(c['frozen_evaluator_sha256'],'d8de4c09a197a9647109369ac77082b0588c8b2ad2f1b940bd479bc47cdbceef')

    def test_eight_established_detector_regressions(self):
        cfg=read(ROOT/'config.json')['detector'];index=read(ROOT/'tests/golden/INDEX.json')
        exp014_seal=read(ROOT/'evidence/EXP014_SCORE_SEAL.json')['files']
        exp012_seal=read(ROOT/'evidence/EXP012_SCORE_EXPORT_FREEZE.json')['files']
        for r in index:
            with self.subTest(experiment=r['experiment'],job=r['job']):
                base=ROOT/'tests/golden'/r['experiment'].lower()/r['job']
                original=exp014_seal if r['experiment']=='EXP014' else exp012_seal
                for n,h in r['source_files'].items():
                    self.assertEqual(sha(base/n),h)
                    original_path=(r['job']+'/'+n) if r['experiment']=='EXP014' else ('units/'+r['job']+'/'+n)
                    self.assertEqual(original[original_path],h)
                a=arrays(base/'scores.npz');beg=time.monotonic();result=detect(a['target'],a['prediction'],a['target_index'],cfg)
                for key,expected in [('smoothed','scores_ewma'),('flags','adaptive_flags'),('issue_index','adaptive_issue_index')]:self.assertTrue(np.array_equal(result[key],a[expected]),key)
                self.assertEqual(result['windows'],read(base/'adaptive_windows.json'))
                self.measurements.append({'kind':'frozen_detector_regression_CPU','experiment':r['experiment'],'job':r['job'],'samples':len(a['target']),'seconds':time.monotonic()-beg,'bitwise':True})

    def test_short_zero_floor_and_tail_detector_semantics(self):
        for n in [1,69,70,149,1799,1800,2500,3113]:
            ix=np.arange(250,250+n,dtype=np.int64);target=np.sin(np.arange(n)/13)*.1;prediction=target.copy()
            zero=detect(target,prediction,ix)
            self.assertFalse(zero['flags'].any());self.assertTrue(np.all(zero['issue_index']==-1))
            self.assertEqual(zero['status'],'NO_OUTPUT_SHORT_STREAM' if n<70 else 'OUTPUT_COMPLETE')
            if n>=70:self.assertEqual(zero['windows'][-1]['end_exclusive'],250+n)
        n=3113;ix=np.arange(250,250+n,dtype=np.int64);target=np.sin(np.arange(n)/11)*.2;prediction=target.copy();prediction[1200:1230]+=2;prediction[3040:]+=4
        r=detect(target,prediction,ix);f=r['flags']
        self.assertTrue(f.any());self.assertTrue(np.all(r['issue_index'][f]>=ix[f]));self.assertTrue(np.all(r['issue_index'][f]<=ix[-1]));self.assertEqual(r['windows'][-1]['end_exclusive'],250+n)

    def test_inverse_scale_original_target_precision(self):
        meta={'mean':.55,'scale':.001};original=np.array([[.551234567895,.3,.7],[.549765432195,.2,.8]],np.float64)
        z=transform(original,meta);self.assertEqual(z.dtype,np.float32);self.assertTrue(np.array_equal(z[:,1:],original[:,1:].astype(np.float32)))
        self.assertTrue(np.array_equal(z[:,0],((original[:,0]-meta['mean'])/meta['scale']).astype(np.float32)))
        p=z[:,0];reference=inverse_reference(p,meta);error=np.abs(original[:,0]-reference)
        self.assertTrue(np.array_equal(reference,p.astype(np.float64)*meta['scale']+meta['mean']))
        self.assertTrue(np.any(error>0));self.assertFalse(np.array_equal(error,np.abs(inverse_reference(z[:,0],meta)-reference)))
        with self.assertRaises(RuntimeError):transform(original,meta,{'mean':[0,0,0],'scale':[1,1,1]})

    def test_reference_causality_telemetry_invariance_and_removal(self):
        torch.manual_seed(101);a=np.random.default_rng(44).normal(size=(120,5)).astype(np.float32)
        x=gather(a,74,77);self.assertTrue(np.array_equal(x[0,:-1,1:],a[10:74,1:]));self.assertTrue(np.all(x[:,-1,1:]==0))
        b=a.copy();b[77:]+=500;self.assertTrue(np.array_equal(gather(a,74,77),gather(b,74,77)))
        for role in ['graph','removed','fusion']:
            model=CommandReferenceGCN(5,2,role).eval();e=torch.tensor([0,1,0]);t=torch.from_numpy(x)
            with torch.no_grad():ref,logit=model(t,e)
            changed=t.clone();changed[:,:,0]+=100
            with torch.no_grad():new_ref,new_logit=model(changed,e)
            self.assertTrue(torch.equal(ref,new_ref));self.assertFalse(torch.equal(logit,new_logit))
            commands=t.clone();commands[:,:-1,1:]+=np.float32(4)
            with torch.no_grad():command_ref,_=model(commands,e)
            if role=='removed':self.assertTrue(torch.equal(ref,command_ref))
            else:self.assertFalse(torch.equal(ref,command_ref))
            model.active_commands.zero_()
            with torch.no_grad():null_ref,null_logit=model(t,e)
            self.assertTrue(torch.isfinite(null_ref).all());self.assertTrue(torch.isfinite(null_logit).all())

    def test_actual_full_native_CPU_shapes_and_restoration(self):
        for nf,entities in [(55,27),(25,54)]:
            torch.manual_seed(123);model=CommandReferenceGCN(nf,entities,'graph').eval()
            saved=rng();before=state_hash(model);state={k:v.clone() for k,v in model.state_dict().items()};x=torch.zeros(256,65,nf);e=torch.arange(256)%entities
            beg=time.monotonic()
            with torch.no_grad():first=tuple(v.clone() for v in model(x,e))
            seconds=time.monotonic()-beg;model.load_state_dict(state,strict=True);restore_rng(saved)
            with torch.no_grad():second=model(x,e)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(first,second)));self.assertEqual(state_hash(model),before);self.assertTrue(rng_equal(saved,rng()))
            self.measurements.append({'kind':'full_native_shape_synthetic_CPU','shape':[256,65,nf],'entities':entities,'seconds_first_forward':seconds,'GPU_validation':False,'real_checkpoint':False})

    def test_native_history_before_common_crop_is_preserved(self):
        v=np.linspace(-2,3,600,dtype=np.float32);all_smooth=ewma(v,8);reset=ewma(v[186:],8)
        self.assertNotEqual(all_smooth[186],reset[0]);self.assertTrue(np.array_equal(ewma(v,1),v.astype(np.float64)))

    def test_independent_event_point_and_late_issue_oracle(self):
        ix=np.arange(250,300,dtype=np.int64);flags=np.zeros(50,bool)
        for a,b in [(251,254),(262,267),(276,279),(282,284)]:flags[(ix>=a)&(ix<b)]=True
        issue=np.where(flags,ix,-1);issue[(ix>=262)&(ix<267)]=290;issue[(ix>=282)&(ix<284)]=290
        r,events,out=metrics.channel([(260,270),(280,285),(300,302)],ix,flags,issue,flags.astype(float),flags.astype(float))
        for k,v in {'H':2,'M':0,'U':2,'FPt':6,'N0':35,'TP':7,'FP':6,'FN':8,'TN':29,'late_notifications':2,'events_fully_outside_support':1}.items():self.assertEqual(r[k],v,k)
        self.assertAlmostEqual(r['event_precision'],.5);self.assertAlmostEqual(r['event_recall'],1.)
        pc=.5*(1-6/35);expected=1.25*pc/(.25*pc+1)
        self.assertAlmostEqual(r['corrected_event_f0_5'],expected);self.assertAlmostEqual(r['point_f1'],.5)
        self.assertAlmostEqual(r['point_precision'],7/13);self.assertAlmostEqual(r['point_recall'],7/15)
        self.assertAlmostEqual(r['alarm_starts_per_1000_nominal'],2000/35);self.assertEqual(events[0]['first_notification'],290)
        with self.assertRaises(ValueError):metrics.channel([(260,270)],ix,flags,np.where(flags,ix-1,-1),flags,flags)

    def test_grouped_seed_bootstrap_uses_one_GDN_seed(self):
        a=np.array([[[2,1,2,3,100],[1,2,3,4,100],[2,1,2,2,100]],[[1,1,1,2,80],[1,1,2,3,80],[2,0,1,2,80]]],float)
        b=np.array([[[1,2,2,4,100]],[[1,1,1,2,80]]],float)
        result=metrics.bootstrap(a,b,100,20261005)
        af=np.mean([metrics.pooled(a[:,s])['corrected_event_f0_5'] for s in range(3)]);bf=metrics.pooled(b[:,0])['corrected_event_f0_5']
        self.assertAlmostEqual(result['difference'],af-bf);self.assertEqual(result['repetitions'],100)
        self.assertEqual(result,metrics.bootstrap(a,b,100,20261005))

class EngineeringTests(unittest.TestCase):
    miniature={}

    def test_checkpoint_hash_wrapper_and_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'selected.pt';value={'model':{'x':torch.arange(4)},'identity':{'seed':17}}
            save_torch(p,value);self.assertTrue(torch.equal(load_torch(p)['model']['x'],value['model']['x']))
            b=p.read_bytes();p.write_bytes(b[:-1]+bytes([b[-1]^1]))
            with self.assertRaisesRegex(RuntimeError,'payload corruption'):load_torch(p)

    def test_atomic_commit_stale_corrupt_and_uncommitted_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            work=Path(d);identity={'source':'immutable'};p=commit(work,'unit',identity,{'payload.txt':b'original'})
            self.assertEqual(committed(work,'unit',identity),p)
            with self.assertRaisesRegex(RuntimeError,'Stale'):committed(work,'unit',{'source':'other'})
            staging=work/'units/.tmp-interrupted';staging.mkdir();(staging/'payload').write_bytes(b'partial');discard_uncommitted(work);self.assertFalse(staging.exists())
            (p/'payload.txt').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError,'Manifest'):committed(work,'unit',identity)

    def test_unsafe_duplicate_and_traversal_archives(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../escape',b'x')
            with self.assertRaisesRegex(RuntimeError,'Unsafe'):Source(p)
            with zipfile.ZipFile(p,'w') as z:z.writestr('a',b'1');z.writestr('a',b'2')
            with self.assertRaisesRegex(RuntimeError,'Duplicate'):Source(p)

    def test_annotation_access_gate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'labels.csv';p.write_text('fictional');g=LabelGuard(p);g.install()
            try:
                with self.assertRaises(PermissionError):p.read_text()
                g.allowed=True;self.assertEqual(p.read_text(),'fictional')
            finally:g.active=False

    def test_two_generation_backup_corrupt_fallback_and_runtime_identity(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);work=d/'live';work.mkdir();env={'GPU':'CPU_FIXTURE'};identity={'fixture':True,'source':'same'}
            write(work/'RUN_FREEZE.json',{'environment':env});(work/'value').write_text('first')
            staging=work/'units/.tmp-incomplete';staging.mkdir(parents=True);(staging/'payload').write_text('partial')
            backup=Backup(work,d/'drive',d/'cache',identity);first=backup.save('first')
            self.assertFalse(any('.tmp-' in n for n in first['files']))
            shutil.rmtree(staging);(work/'value').write_text('second');second=backup.save('second')
            (d/'drive/objects'/second['objects']['state_archive']).write_bytes(b'corrupt')
            shutil.rmtree(work);self.assertTrue(backup.restore(env));self.assertEqual((work/'value').read_text(),'first')
            shutil.rmtree(work)
            with self.assertRaises(EnvironmentMismatch):backup.restore({'GPU':'OTHER'})

    def test_missing_originals_are_exactly_enumerated_without_substitution(self):
        from run import execute
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'missing_FINAL_RESULTS.zip'
            with self.assertRaises(FileNotFoundError):execute(ROOT,source,d/'work',d/'drive',d/'cache','cuda',False)
            diag=read(d/'cache/MISSING_OR_INVALID_INPUTS.json');self.assertEqual(len(diag['constituents']),2260)
            self.assertEqual(diag['exact_requested_file'],str(source));self.assertTrue(diag['no_retraining_or_substitution'])

    def test_inherited_interactive_backend_is_overridden_in_worker(self):
        env=dict(os.environ,MPLBACKEND='module://ipykernel.pylab.backend_inline',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(ROOT/'src')+os.pathsep+str(ROOT))
        command="import run; from integrity import configure; configure('cpu',True); import matplotlib; import matplotlib.pyplot as plt; f=plt.figure(); print(matplotlib.get_backend()); plt.close(f)"
        p=subprocess.run([sys.executable,'-c',command],env=env,text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout.strip().lower(),'agg')

    def test_miniature_interruption_restore_and_uninterrupted_equivalence(self):
        from fixture import make_fixture
        from run import execute
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);start=time.monotonic();root,source=make_fixture(d)
            before=sha(source);inputs_before=files(root);work=d/'resumed_work';dest=d/'resumed_drive';cache=d/'resumed_cache'
            with self.assertRaises(InterruptedError):execute(root,source,work,dest,cache,'cpu',True,interrupt_after_units=1)
            self.assertFalse((work/'SCORE_SEAL.json').exists());self.assertNotIn('ANNOTATION_CSV_PARSED',(work/'PROGRESS.jsonl').read_text())
            first_path=next((work/'units').glob('*/scores.npz'));first_job=first_path.parent.name;first_score=sha(first_path);shutil.rmtree(work)
            resumed=execute(root,source,work,dest,cache,'cpu',True)
            self.assertEqual(resumed['status'],'CPU_SYNTHETIC_FIXTURE_COMPLETE');self.assertFalse(resumed['real_NASA_comparison'])
            units=sorted((work/'units').iterdir());self.assertEqual(len(units),18);self.assertEqual(sha(work/'units'/first_job/'scores.npz'),first_score)
            clean=d/'clean_work';clean_summary=execute(root,source,clean,d/'clean_drive',d/'clean_cache','cpu',True)
            for p in units:
                a=arrays(p/'scores.npz');b=arrays(clean/'units'/p.name/'scores.npz')
                for k in a:self.assertTrue(np.array_equal(a[k],b[k]),p.name+'/'+k)
            for name in ['mission_metrics.csv','mission_seed_metrics.csv','event_notifications.csv','paired_contrasts.csv']:
                self.assertEqual((work/'reports'/name).read_bytes(),(clean/'reports'/name).read_bytes(),name)
            self.assertEqual(sha(source),before);self.assertEqual(files(root),inputs_before)
            latest=read(dest/'LATEST_EXPORT.json');archive=dest/latest['archive'];self.assertEqual(sha(archive),latest['sha256'])
            with zipfile.ZipFile(archive) as z:
                snap=__import__('json').loads(z.read('SNAPSHOT_MANIFEST.json'))
                self.assertEqual(set(z.namelist()),set(snap['files'])|{'SNAPSHOT_MANIFEST.json'})
                for n,h in snap['files'].items():self.assertEqual(__import__('hashlib').sha256(z.read(n)).hexdigest(),h)
            audit=read(work/'FINAL_AUDIT.json');self.assertFalse(audit['A100_validation']);self.assertTrue(audit['original_content_rechecked_after_execution']);self.assertFalse(audit['originals_modified'])
            log=(work/'PROGRESS.jsonl').read_text();self.assertLess(log.index('SCORE_SEAL_VERIFIED_BEFORE_ANNOTATION_PARSE'),log.index('ANNOTATION_CSV_PARSED'))
            # Completed resume must not invoke the model again.
            import inference
            original=inference.predict
            def prohibited(*args,**kwargs):raise AssertionError('Completed stream was recomputed')
            inference.predict=prohibited
            try:execute(root,source,work,dest,cache,'cpu',True)
            finally:inference.predict=original
            self.miniature.update({'status':'PASS','streams':18,'full_shape_gates':18,'interruption_after_units':1,
                'verified_restore':True,'resumed_uninterrupted_arrays_and_tables_equal':True,'completed_resume_skips_inference':True,
                'source_unchanged':True,'final_export_hash_roster_verified':True,'seconds':time.monotonic()-start,'real_NASA':False,'actual_A100':False})
            artifacts=os.environ.get('EXP015_TEST_ARTIFACTS')
            if artifacts:
                out=Path(artifacts)/'CPU_MINIATURE';shutil.copytree(work/'reports',out,dirs_exist_ok=True)
