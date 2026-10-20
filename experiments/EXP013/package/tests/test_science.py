"""Synthetic fixtures only. No NASA labels, no NASA TEST float arrays, no research fits."""
import sys,unittest,tempfile,copy,io,zipfile,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np, torch
from unittest.mock import patch
from core import *
from data import Windows,mutate,cases,inject_batch
from architectures import CommandReferenceGCN
from scoring import causal_mean,quantile,gdn_native,scores
from historical_metrics import ap,channel,fbeta
import pipeline,train,upstream

class ScienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure('cpu');seeds(17)
    def model(self,role='graph'):
        seeds(17);m=CommandReferenceGCN(6,2,role);m.eval();return m
    def fixture(self,T=360):
        gen=np.random.default_rng(13);a=gen.normal(size=(T,6)).astype(np.float32);m={'mission':'SMAP','rows':T,'columns':6,'fit_end':int(.6*T),'stop_end':int(.8*T),'cal_end':int(.9*T)}
        return {'one':a,'two':a*.9},{'one':m,'two':m.copy()}
    def test_reference_ignores_all_telemetry(self):
        m=self.model();x=torch.randn(2,65,6);e=torch.tensor([0,1]);b=x.clone();b[:,:,0]+=40
        self.assertTrue(torch.equal(m(x,e)[0],m(b,e)[0]));self.assertFalse(torch.equal(m(x,e)[1],m(b,e)[1]))
    def test_removal_cuts_every_command_path(self):
        m=self.model('removed');x=torch.randn(2,65,6);b=x.clone();b[:,:,1:]+=77;e=torch.tensor([0,1])
        for a,c in zip(m(x,e),m(b,e)):self.assertTrue(torch.equal(a,c))
    def test_graph_and_removed_initial_states_match(self):
        a=self.model('graph');b=self.model('removed');self.assertEqual(state_hash(a),state_hash(b))
    def test_current_command_is_not_used(self):
        m=self.model();x=torch.randn(2,65,6);b=x.clone();b[:,-1,1:]+=100;e=torch.tensor([0,1])
        for a,c in zip(m(x,e),m(b,e)):self.assertTrue(torch.equal(a,c))
    def test_constant_FIT_commands_use_finite_null_message(self):
        m=self.model();m.active_commands.zero_();x=torch.randn(2,65,6);b=x.clone();b[:,:,1:]+=100;e=torch.tensor([0,1])
        for a,c in zip(m(x,e),m(b,e)):self.assertTrue(torch.isfinite(a).all());self.assertTrue(torch.equal(a,c))
    def test_future_does_not_change_scores(self):
        m=self.model();a=np.random.default_rng(0).normal(size=(290,6)).astype(np.float32);b=a.copy();b[200:]+=99
        s1=scores(m,'graph',a,0,64,290,'cpu',{}, {},13);s2=scores(m,'graph',b,0,64,290,'cpu',{}, {},13)
        self.assertTrue(np.array_equal(s1[:200-64],s2[:200-64]))
    def test_no_cross_entity_windows(self):
        r,m=self.fixture();r['one'][:]=1;r['two'][:]=2;ds=Windows(r,m,['one','two'],'graph')
        for i in [0,len(ds)-1]:
            x,e,y=ds[i];self.assertTrue(torch.all(x[:-1]==e+1));self.assertTrue(torch.all(x[-1,1:]==0))
    def test_fit_stop_boundaries(self):
        r,m=self.fixture()
        for split in ['fit','stop']:
            ds=Windows(r,m,['one','two'],'gdn',split)
            for ch,t,e in ds.rows:self.assertLess(t,m[ch]['fit_end'] if split=='fit' else m[ch]['stop_end']);self.assertGreaterEqual(t,5 if split=='fit' else m[ch]['fit_end'])
    def test_bank_quarantine_and_duration_receipts(self):
        r,m=self.fixture()
        for phase in ['select','confirm']:
            for c,b in cases(r['one'],m['one'],'one',phase):
                self.assertGreaterEqual(c['onset'],m['one']['stop_end'] if phase=='select' else m['one']['cal_end']);self.assertLessEqual(c['stop'],m['one']['cal_end'] if phase=='select' else m['one']['rows']);self.assertTrue(np.array_equal(b[:c['onset']],r['one'][:c['onset']]))
    def test_commands_are_never_corrupted(self):
        a=torch.randn(32,65,6);b,valid=inject_batch(a);self.assertTrue(torch.equal(a[:,:,1:],b[:,:,1:]));self.assertTrue(torch.all(valid==(a[:,:,0]-b[:,:,0]).abs().amax(1).gt(1e-7)))
    def test_unchanged_stuck_is_invalid(self):
        a=np.ones((40,6),np.float32);b,e,valid=mutate(a,12,20,'stuck',3,9);self.assertFalse(valid)
    def test_ewma_prefix_is_causal(self):
        a=np.arange(20.);b=a.copy();b[10:]=999;self.assertTrue(np.array_equal(causal_mean(a,8)[:10],causal_mean(b,8)[:10]))
    def test_strict_tied_quantile(self):
        v=np.ones(31);t=quantile(v,.995);self.assertEqual(t,1.);self.assertEqual((v>t).sum(),0)
    def test_gdn_uses_frozen_train_stats(self):
        v=np.arange(24.).reshape(8,3);st={'median':[1,2,3],'iqr':[2,3,4]};n=gdn_native(v,st);b=v.copy();b[-1]+=999;self.assertTrue(np.array_equal(n[:-1],gdn_native(b,st)[:-1]))
    def test_tied_AP_and_no_positive_support(self):
        self.assertEqual(ap([1,0],[1,1]),.5);self.assertTrue(np.isnan(ap([0,0],[1,2])))
    def test_false_alarm_penalty_and_true_issue_time(self):
        ix=np.arange(250,260);f=np.array([1,1,0,0,0,0,0,0,0,0],bool);r,ev,_=channel([(250,252)],ix,f,np.where(f,ix+9,-1),np.arange(10),np.arange(10))
        self.assertEqual(ev[0]['delay'],9);self.assertTrue(ev[0]['late_post_event_notification']);self.assertEqual(r['within_event_notifications'],0)
    def test_empty_support_undefined_AP(self):
        r,ev,out=channel([(0,10)],np.array([],int),np.array([],bool),np.array([],int),np.array([]),np.array([]));self.assertTrue(np.isnan(r['AP_raw']));self.assertEqual(len(out),1)
    def test_paired_point_bootstrap_uses_pooled_confusion_counts(self):
        a=np.array([[[1,0,1,1,10,2,1,1]],[[0,1,0,0,5,0,0,2]]],float)
        b=np.array([[[1,0,0,0,10,1,0,2]],[[1,0,1,2,5,2,2,0]]],float)
        r=pipeline.paired_bootstrap(a,b)['point_and_alarm_contrasts']
        self.assertAlmostEqual(r['point_f1']['difference'],.5-.6)
        self.assertAlmostEqual(r['point_recall']['difference'],.4-.6)
        self.assertAlmostEqual(r['nominal_occupancy']['difference'],1/15-2/15)
        identical=pipeline.paired_bootstrap(a,a)
        self.assertEqual(identical['percentile_interval'],[0.,0.])
        self.assertEqual(identical['point_and_alarm_contrasts']['point_f1']['percentile_interval'],[0.,0.])
    def test_checkpoint_is_transactional_and_safe(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'state.pt';s={'rng':rng(),'model':self.model().state_dict()};save_torch(p,s);v=load_torch(p);self.assertTrue(torch.equal(v['rng']['torch'],s['rng']['torch']));b=bytearray(p.read_bytes());b[-10]^=1;p.write_bytes(b)
            with self.assertRaises(RuntimeError):load_torch(p)
    def test_payload_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);write(p/'a.json',{'x':1});seal(p,'m.json');manifest(p,'m.json');write(p/'a.json',{'x':2})
            with self.assertRaises(RuntimeError):manifest(p,'m.json')
    def test_zip_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.zip'
            with zipfile.ZipFile(p,'w') as z:z.writestr('../escape','no')
            with self.assertRaises(RuntimeError):safe_extract(p,Path(td)/'safe')
    def test_labels_are_not_opened_when_seal_fails(self):
        with tempfile.TemporaryDirectory() as td,patch.object(pipeline,'verify_export',side_effect=RuntimeError('unsealed')),patch.object(pipeline,'read_labels') as parser:
            with self.assertRaises(RuntimeError):pipeline.evaluate(Path(td),Path(td)/'labels.csv')
            parser.assert_not_called()
    def test_snapshot_restore_verifies_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);a=p/'work';a.mkdir();write(a/'a.json',{'v':2});pipeline.snapshot(a,p/'snap.zip');pipeline.restore_snapshot(p/'snap.zip',p/'restored');self.assertEqual(read(p/'restored/a.json'),{'v':2})
    def test_new_model_epoch_resume_is_exact(self):
        raw,meta=self.fixture();real=train.model_for
        def small(*args):
            m,c=real(*args);c.update(epochs=2,patience=20,batch_size=32);return m,c
        with tempfile.TemporaryDirectory() as td,patch.object(train,'model_for',small):
            w1=Path(td)/'whole';w2=Path(td)/'resume';identity={'fixture_only':True}
            a=train.train_job(w1,raw,meta,'SMAP','graph',17,{},'cpu',identity)
            train.train_job(w2,raw,meta,'SMAP','graph',17,{},'cpu',identity,epoch_limit=1)
            b=train.train_job(w2,raw,meta,'SMAP','graph',17,{},'cpu',identity)
            self.assertEqual(a['state_sha256'],b['state_sha256']);self.assertEqual(a['epoch'],b['epoch'])
            old=sha(w2/'models/SMAP__graph__17/selected.pt');train.train_job(w2,raw,meta,'SMAP','graph',17,{},'cpu',identity);self.assertEqual(sha(w2/'models/SMAP__graph__17/selected.pt'),old)

if __name__=='__main__':unittest.main(verbosity=2)
