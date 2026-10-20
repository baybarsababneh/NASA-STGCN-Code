import sys, unittest, tempfile, json, hashlib
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(CODE/'src'),str(CODE)]
import numpy as np
import torch
from integrity import rng,rng_equal,restore_rng,state_hash,read,sha
from vendor.frozen.hundman_detector import detect,prune,choose_threshold,dilate
from vendor.frozen import metrics
from gdn_inference import factory,gather,transform,inverse_target,native,ewma
from checkpoint import save_torch,load_torch

class ScientificTests(unittest.TestCase):
    numerical_diagnostics=[]
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

    def test_frozen_detector_and_evaluator_bytes(self):
        c=read(CODE/'INPUT_CONTRACT.json')
        self.assertEqual(sha(CODE/'vendor/frozen/hundman_detector.py'),c['frozen_detector_sha256'])
        self.assertEqual(sha(CODE/'vendor/frozen/metrics.py'),c['frozen_evaluator_sha256'])

    def test_residual_double_inverse_with_FP32_intermediate(self):
        a=np.array([[.49999997,1.],[.90000003,0.]],np.float32);meta={'mean':.32,'scale':.001};scaler={'mean':[.05,.5],'scale':[2.3,.7]}
        z=transform(a,meta,scaler)
        first=np.array([(float(v)-.32)/.001 for v in a[:,0]],np.float32)
        reference=np.array([(float(v)-.05)/2.3 for v in first],np.float32)
        self.assertTrue(np.array_equal(z[:,0],reference))
        p=np.array([.1234567,1.25],np.float32)
        original=inverse_target(p,meta,scaler)
        expected=np.array([((float(v)*2.3+.05)*.001+.32) for v in p])
        self.assertTrue(np.array_equal(original,expected))
        wrong=p.astype(float)*meta['scale']+meta['mean'];self.assertGreater(np.max(np.abs(original-wrong)),1e-3)
        self.assertTrue(np.array_equal(np.abs(a[:,0].astype(float)-original),np.array([abs(float(v)-q) for v,q in zip(a[:,0],expected)])))

    def test_causal_forecast_alignment_and_future_invariance(self):
        a=np.arange(120*55,dtype=np.float32).reshape(120,55)/1000
        x=gather(a,40,72);self.assertEqual(x.shape,(32,55,5));self.assertTrue(np.array_equal(x[0].T,a[35:40]))
        changed=a.copy();changed[72:]+=10000
        self.assertTrue(np.array_equal(x,gather(changed,40,72)))
        # A t=40 forecast also excludes observed t=40.
        changed=a.copy();changed[40:]+=10000
        self.assertTrue(np.array_equal(gather(a,40,41),gather(changed,40,41)))
        m=factory(55,'cpu').eval()
        with torch.no_grad():
            p=m(torch.from_numpy(gather(a,40,41)),None);q=m(torch.from_numpy(gather(changed,40,41)),None)
        self.assertTrue(torch.equal(p,q))

    def test_native_smoothing_precedes_common_support_crop(self):
        e=np.linspace(0,1,600*25,dtype=np.float32).reshape(600,25)
        s=native(e,{'median':[.1]*25,'iqr':[.2]*25});f=ewma(s,8)
        self.assertTrue(np.array_equal(s[:3],np.zeros(3)))
        self.assertFalse(np.array_equal(ewma(s[245:],8),f[245:]))
        expected=(e[245-3:245+1].astype(float)-.1)/.21
        self.assertAlmostEqual(s[245],expected.mean(0).max(),places=14)

    def test_pruning_buffers_ties_and_short_support(self):
        self.assertTrue(np.array_equal(dilate([0,0,1,0,0],1),[0,1,1,1,0]))
        self.assertFalse(prune(np.array([0.,9.,9.,0.,8.5,8.5,0.]),[0,1,1,0,1,1,0],8.4,.13).any())
        keep=prune(np.array([0.,9.,9.,0.,2.,2.,0.]),[0,1,1,0,1,1,0],1.8,.13)
        self.assertTrue(np.array_equal(keep,[0,1,1,0,0,0,0]))
        a=detect(np.zeros(69),np.ones(69),np.arange(250,319))
        self.assertEqual(a['status'],'NO_OUTPUT_SHORT_STREAM');self.assertFalse(a['flags'].any())
        for n in [70,71,499,500,1799,1800,2499,2500,2601]:
            y=np.sin(np.arange(n)/13);p=y.copy();p[n//2:n//2+20]+=4
            d=detect(y,p,np.arange(250,250+n));self.assertEqual(d['windows'][-1]['end_exclusive'],250+n)
            issue=d['issue_index'];f=d['flags'];self.assertTrue(np.all(issue[f]>=np.arange(250,250+n)[f]))
            self.assertEqual(d['startup_averaging_lookahead'],min(n,500))

    def test_event_count_timing_and_no_point_adjustment(self):
        ix=np.arange(250,260);f=np.array([1,0,1,0,0,1,0,0,0,1],bool);issue=np.where(f,259,-1)
        r,e,_=metrics.channel([(252,256)],ix,f,issue,np.arange(10),np.arange(10))
        self.assertEqual((r['H'],r['M'],r['U'],r['TP'],r['FP'],r['FN'],r['TN']),(1,0,2,2,2,2,4))
        self.assertEqual(e[0]['delay'],7);self.assertTrue(e[0]['late_post_event_notification'])
        self.assertAlmostEqual(r['point_f1'],.5)
        self.assertAlmostEqual(r['corrected_event_f0_5'],1.25*((1/3)*(1-2/6))/(.25*((1/3)*(1-2/6))+1))

    def test_aggregation_grouped_seeds_and_unequal_initialisations(self):
        from report import auxiliary_bootstrap
        a=np.array([[[1,0,2,4,20],[0,1,3,5,20],[1,0,1,1,20]],[[1,0,0,0,40],[1,0,1,2,40],[0,1,2,8,40]]],float)
        b=np.array([[[1,0,1,2,20]],[[0,1,1,3,40]]],float)
        r=metrics.bootstrap(a,b,repetitions=100,seed=20261005)
        expected=np.mean([metrics.pooled(a[:,s,:])['corrected_event_f0_5'] for s in range(3)])-metrics.pooled(b[:,0,:])['corrected_event_f0_5']
        self.assertEqual(r['difference'],expected);self.assertEqual(r,metrics.bootstrap(a,b,100,20261005))
        counts=np.zeros((2,1,10));counts[:,:,4]=20
        aux=auxiliary_bootstrap(counts,counts,20,20261005)
        self.assertEqual(aux['point_f1']['undefined_replicates'],20)

    def test_production_cannot_accept_CPU_validation(self):
        from integrity import configure
        if str(torch.__version__).endswith('+cpu'):
            with self.assertRaises(RuntimeError):configure('cuda',False)
        else:
            with self.assertRaises(RuntimeError):configure('cpu',False)

    def test_native_GDN_forward_gradient_equations_both_missions(self):
        def dense(layer,x,embedding,edges):
            n=len(x);edge=edges[:,edges[0]!=edges[1]];edge=torch.cat([edge,torch.arange(n).repeat(2,1)],1)
            src,dst=edge;z=layer.lin(x);zi=z[dst][:,None,:];zj=z[src][:,None,:]
            ei=embedding[dst][:,None,:];ej=embedding[src][:,None,:]
            ai=torch.cat([layer.att_i,layer.att_em_i],-1);aj=torch.cat([layer.att_j,layer.att_em_j],-1)
            score=torch.nn.functional.leaky_relu((torch.cat([zi,ei],-1)*ai).sum(-1)+(torch.cat([zj,ej],-1)*aj).sum(-1),.2)
            result=[]
            for i in range(n):
                select=dst==i;weight=score[select].softmax(0);result.append((zj[select]*weight[:,:,None]).sum(0).mean(0))
            return torch.stack(result)+layer.bias
        for nf in [55,25]:
            torch.manual_seed(17);m=factory(nf,'cpu');layer=m.gnn_layers[0].gnn
            with torch.no_grad():layer.att_em_i.uniform_(-.2,.2);layer.att_em_j.uniform_(-.2,.2)
            x=torch.randn(nf,5,requires_grad=True);emb=m.embedding.weight.detach().clone().requires_grad_(True)
            y=layer(x,m.edge_index_sets[0],emb);y.square().sum().backward();gx=x.grad.clone();ge=emb.grad.clone();gp={k:p.grad.clone() for k,p in layer.named_parameters()}
            layer.zero_grad();xx=x.detach().clone().requires_grad_(True);ee=emb.detach().clone().requires_grad_(True)
            ref=dense(layer,xx,ee,m.edge_index_sets[0]);ref.square().sum().backward()
            torch.testing.assert_close(y,ref,atol=3e-6,rtol=1e-5);torch.testing.assert_close(gx,xx.grad,atol=3e-6,rtol=1e-5)
            torch.testing.assert_close(ge,ee.grad,atol=5e-6,rtol=3e-5)
            for k,p in layer.named_parameters():torch.testing.assert_close(gp[k],p.grad,atol=5e-6,rtol=3e-5)
            self.numerical_diagnostics.append({'nodes':nf,'forward_max_abs':float((y-ref).abs().max().detach()),'input_gradient_max_abs':float((gx-xx.grad).abs().max()),'embedding_gradient_max_abs':float((ge-ee.grad).abs().max()),'parameter_gradient_max_abs':max(float((gp[k]-p.grad).abs().max()) for k,p in layer.named_parameters()),'original_tolerances_unchanged':True})

    def test_full_native_shapes_exact_checkpoint_and_rng(self):
        for nf in [55,25]:
            m=factory(nf,'cpu').eval();x=torch.randn(32,nf,5);s=rng();before=state_hash(m)
            with torch.no_grad(): first=m(x,None).clone()
            self.assertTrue(rng_equal(s,rng()));self.assertEqual(first.shape,(32,nf))
            with tempfile.TemporaryDirectory() as td:
                p=Path(td)/'selected.pt';save_torch(p,{'model':m.state_dict()});ck=load_torch(p)
                m.load_state_dict(ck['model'],strict=True);restore_rng(s)
                with torch.no_grad(): second=m(x,None)
                self.assertTrue(torch.equal(first,second));self.assertEqual(before,state_hash(m));self.assertTrue(rng_equal(s,rng()))
                b=p.read_bytes();p.write_bytes(b[:-1]+bytes([b[-1]^1]))
                with self.assertRaises(RuntimeError):load_torch(p)

    def test_saved_frozen_EXP012_detector_regression(self):
        from inputs import verify_static
        verify_static(CODE)
        # Both missions, GCN/LSTM, both long and shorter support, with frozen
        # outputs as independent references. No historical model inference.
        for arm,ch in [('gcn','A-2'),('lstm','A-2'),('gcn','C-1'),('lstm','C-1'),('gcn','D-12'),('lstm','D-12')]:
            p=CODE/'inputs/exp012_scores/units'/f'{arm}__{ch}__17'
            with np.load(p/'scores.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
            d=detect(a['target'],a['prediction'],a['target_index'],read(CODE/'config.json')['detector'])
            self.assertTrue(np.array_equal(d['smoothed'],a['scores_ewma']))
            self.assertTrue(np.array_equal(d['flags'],a['adaptive_flags']))
            self.assertTrue(np.array_equal(d['issue_index'],a['adaptive_issue_index']))
            self.assertEqual(d['windows'],read(p/'adaptive_windows.json'))

    def test_all_frozen_EXP012_counts_and_mission_aggregation(self):
        from inputs import verify_static
        from report import labels,check_historical_rows,mission_tables
        import pandas as pd
        verify_static(CODE)
        # These historical scores/results are already sealed. This is a
        # retrospective arithmetic check, not new inference or TEST selection.
        events,audit=labels(CODE,True)
        rows,_,_=check_historical_rows(CODE,events)
        self.assertEqual((len(rows),len(events),audit['original_events']),(486,81,104))
        _,means=mission_tables(rows)
        frozen=pd.read_csv(CODE/'inputs/exp012_reports/mission_metrics.csv')
        for row in means:
            old=frozen[(frozen.mission==row['mission'])&(frozen.arm==row['arm'])&(frozen.detector=='hundman_adapt_v1')]
            self.assertEqual(len(old),1)
            self.assertAlmostEqual(row['corrected_event_f0_5'],float(old.iloc[0]['corrected_event_f0_5']),places=12)

if __name__=='__main__':unittest.main()
