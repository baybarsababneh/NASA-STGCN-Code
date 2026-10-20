import unittest
import numpy as np
import torch
from torch.nn import functional as F
from round2_models import supplied,Temporal,TCN
from round2_train import prepare

class Audit(unittest.TestCase):
    def test_legacy_checkpoint_and_outputs(self):
        a=np.zeros((4,4),np.float32);a[0,1:]=1/3
        torch.manual_seed(17);old=supplied.TCNSTGCN(4,64,16,a)
        torch.nn.init.normal_(old.head.weight)
        torch.manual_seed(17);new=TCN(4,64,16,a)
        new.load_state_dict(old.state_dict(),strict=True)
        x=torch.randn(3,4,64)
        torch.testing.assert_close(new(x),old(x),rtol=2e-5,atol=2e-6)
    def test_effective_seven_steps(self):
        net=Temporal(8);x=torch.randn(3,4,64);z=x.clone();z[:,:,:-7]+=1000
        torch.testing.assert_close(net(x),net(z),rtol=0,atol=0)
    def test_optimized_against_causal_reference(self):
        for L,ds in [(64,[1,2]),(64,[1,2,4,8,16]),(128,[1,2,4,8,16,32]),(250,[1,2,4,8,16,32,64]),(4,[1,2]),(1,[1,2])]:
            net=Temporal(8,3,ds);x=torch.randn(2,3,L);z=x.reshape(6,1,L)
            for i,d in enumerate(ds):z=torch.relu(getattr(net,f'conv{i+1}')(F.pad(z,(2*d,0))))
            torch.testing.assert_close(net(x),z[:,:,-1].reshape(2,3,8),rtol=2e-5,atol=2e-6)
            if L==250:
                actual_grads=torch.autograd.grad(net(x).square().sum(),tuple(net.parameters()))
                reference_grads=torch.autograd.grad(z[:,:,-1].square().sum(),tuple(net.parameters()))
                for ga,gr in zip(actual_grads,reference_grads):torch.testing.assert_close(ga,gr,rtol=2e-4,atol=1e-5)
    def test_orientation_and_no_cross_node_mixing(self):
        net=Temporal(8);x=torch.randn(2,3,64)
        for b in range(2):
            for n in range(3):torch.testing.assert_close(net(x)[b,n],net(x[b:b+1,n:n+1])[0,0])
    def test_persistence_initialization(self):
        x=torch.randn(3,4,64);net=TCN(4,64,16,np.zeros((4,4),np.float32))
        torch.testing.assert_close(net(x),x[:,0,-1],rtol=0,atol=0)
    def test_target_windows_and_sealed_calibration(self):
        for L in [64,128,250]:
            x,X,y,t,m=prepare('P-1',L)
            for i in [0,100,len(t)-1]:
                ti=t[i];expected=np.pad(x[max(0,ti-L):ti],((max(0,L-ti),0),(0,0))).T
                np.testing.assert_array_equal(X[i],expected);self.assertEqual(y[i],x[ti,0])
            self.assertEqual(t[0],64);self.assertLess(t[-1],m['validation_end']);self.assertFalse(m['calibration_accessed'])
    def test_short_channel_not_excluded(self):
        x,X,y,t,m=prepare('D-12',250)
        self.assertGreater(sum(t<m['fit_end']),0)

if __name__=='__main__':unittest.main()
