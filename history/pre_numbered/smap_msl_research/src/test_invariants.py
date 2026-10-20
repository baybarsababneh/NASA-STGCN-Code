"""Scientific correctness checks: time alignment, label boundaries and isolation."""
import unittest
import numpy as np
import torch
from common import intervals,event_metrics,labels,prf
from models import STGCN,adjacency
from train import prepare

class Invariants(unittest.TestCase):
    def test_endpoints_and_events(self):
        self.assertEqual(intervals([1,1,0,1]),[(0,1),(3,3)])
        self.assertEqual(event_metrics([(1,3),(7,9)],[(3,7)])['tp'],2)
        self.assertEqual(event_metrics([(1,3)],[(1,1),(3,3),(5,5)])['fp'],1)
        self.assertAlmostEqual(prf(59,10,10)['f05'],59/69)
    def test_duplicate_policy(self):
        ls=labels();self.assertEqual(len(ls),81)
        self.assertEqual(next(x for x in ls if x['channel']=='P-2')['events'],[(5300,6575)])
        self.assertNotIn('T-10',[x['channel'] for x in ls])
    def test_target_alignment(self):
        x,X,y,t,m=prepare('P-1',64)
        np.testing.assert_array_equal(X[0].numpy(),x[:64].T)
        self.assertEqual(float(y[0]),float(x[64,0]))
        self.assertEqual(t[0],64)
        fit=t[t<m['fit_end']];val=t[(t>=m['fit_end'])&(t<m['validation_end'])];cal=t[t>=m['validation_end']]
        self.assertLess(fit[-1],val[0]);self.assertLess(val[-1],cal[0])
    def test_no_edges_blocks_commands(self):
        torch.manual_seed(2);net=STGCN(3,8,4,np.zeros((3,3),np.float32))
        torch.nn.init.normal_(net.head.weight)
        x=torch.randn(4,3,8);z=x.clone();z[:,1:]+=50
        torch.testing.assert_close(net(x),net(z))
    def test_graph_edges_carry_commands(self):
        torch.manual_seed(2);a=np.zeros((3,3),np.float32);a[0,1:]=.5
        net=STGCN(3,8,4,a);torch.nn.init.normal_(net.head.weight)
        x=torch.randn(4,3,8);z=x.clone();z[:,1:]+=50
        self.assertGreater(float((net(x)-net(z)).abs().max().detach()),1e-5)
    def test_affine_scaling_cancels(self):
        rng=np.random.default_rng(3);x=rng.normal(size=200);y=5*x-7
        np.testing.assert_allclose((x-x[:100].mean())/x[:100].std(),(y-y[:100].mean())/y[:100].std(),atol=1e-12)

if __name__=='__main__': unittest.main()
