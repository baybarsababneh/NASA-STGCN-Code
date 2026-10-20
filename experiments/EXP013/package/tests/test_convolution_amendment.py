import copy,hashlib,sys,unittest,json,io,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from convolution_regression import module,compare,compare_step,step,cpu_copy
FROZEN_ARCHITECTURE_SHA='de8c5697f92edb0f1bc35e875bcd29a0f9b0fe1dd02d2f3225dc86057b4e54e4'

old=module(ROOT/'frozen_originals/architectures.py','native_test')
new=module(ROOT/'src/architectures.py','gemm_test')

def rng():
    import random
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':(n[0],n[1].tolist(),n[2],n[3],n[4]),'torch':torch.get_rng_state()}

def restore(r):
    import random
    random.setstate(r['python']);n=r['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]));torch.set_rng_state(r['torch'])

def inject_batch(x):
    # The actual data.py function is loaded from --package in Colab. This local
    # copy is checked byte-for-byte against that function during release build.
    y=x.clone();B,T,_=y.shape;dev=x.device
    k=torch.randint(0,5,(B,),device=dev);amp=torch.empty(B,device=dev).uniform_(.5,4.)
    sign=2*torch.randint(0,2,(B,),device=dev)-1
    length=torch.where(k==0,1,torch.where(torch.randint(0,2,(B,),device=dev)==0,8,32))
    index=torch.arange(T,device=dev)[None];mask=index>=T-length[:,None]
    fraction=(index-(T-length[:,None])+1).float()/length[:,None]
    delta=(amp*sign)[:,None]*torch.where(k[:,None]==2,.1+.9*fraction,torch.ones_like(fraction))
    stuck=x[torch.arange(B,device=dev),T-length-1,0][:,None].expand(-1,T)
    donor=x.roll(1,0)[:,:,0]
    changed=torch.where(k[:,None]==4,donor,torch.where(k[:,None]==3,stuck,x[:,:,0]+delta))
    y[:,:,0]=torch.where(mask,changed,x[:,:,0])
    valid=(y[:,:,0]-x[:,:,0]).abs().amax(1)>1e-7
    return y,valid.float()

class CorrectionTests(unittest.TestCase):
    results=[]
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    @classmethod
    def tearDownClass(cls):
        summary={'actual_A100_validation':False,'torch':str(torch.__version__),'fixture_batch_size':8,
                 'cases':cls.results,'pass':all(v['pass'] for v in cls.results)}
        destination=os.environ.get('EXP013_REGRESSION_REPORT')
        if destination:Path(destination).write_text(json.dumps(summary,indent=2))
    def test_reference_is_original(self):
        self.assertEqual(hashlib.sha256((ROOT/'frozen_originals/architectures.py').read_bytes()).hexdigest(),FROZEN_ARCHITECTURE_SHA)
    def test_frozen_forward_and_initialisation(self):
        import ast
        a=ast.parse((ROOT/'frozen_originals/architectures.py').read_text())
        b=ast.parse((ROOT/'src/architectures.py').read_text())
        method=lambda tree:next(n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='CommandReferenceGCN' for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        self.assertEqual(ast.dump(method(a),include_attributes=False),ast.dump(method(b),include_attributes=False))
    def test_causal_kernel_tap_order_and_gradients(self):
        for T in [1,7,64,65]:
            for d in [2,4,8,16]:
                torch.manual_seed(17);c=torch.nn.Conv1d(3,5,3,dilation=d).double()
                x=torch.randn(2,3,T,dtype=torch.float64,requires_grad=True)
                p=F.pad(x,(2*d,0));a=c(p);b=new.dilated_conv1d_fp32(p,c)
                torch.testing.assert_close(a,b,atol=1e-14,rtol=1e-13)
                target=torch.randn_like(a)
                ga=torch.autograd.grad((a*target).sum(),(x,c.weight,c.bias),retain_graph=True)
                gb=torch.autograd.grad((b*target).sum(),(x,c.weight,c.bias))
                for aa,bb in zip(ga,gb):torch.testing.assert_close(aa,bb,atol=1e-13,rtol=1e-12)
    def test_double_gradcheck(self):
        torch.manual_seed(29);c=torch.nn.Conv1d(2,2,3,dilation=4).double()
        x=torch.randn(1,2,8,dtype=torch.float64,requires_grad=True)
        # Functional form includes trainable weight and bias in gradcheck.
        def f(x,w,b):
            p=F.pad(x,(8,0));windows=p.unfold(2,9,1)[...,::4]
            columns=windows.permute(0,2,1,3).reshape(8,6)
            return F.linear(columns,w.flatten(1),b)
        self.assertTrue(torch.autograd.gradcheck(f,(x,c.weight,c.bias)))
    def test_two_adam_updates_all_roles_missions_seeds(self):
        for nf in [55,25]:
            for role in ['graph','removed','fusion']:
                for seed in [17,29,43]:
                    with self.subTest(nf=nf,role=role,seed=seed):
                        torch.manual_seed(seed);a=old.CommandReferenceGCN(nf,3,role);initial=cpu_copy(a.state_dict());init_rng=rng()
                        torch.manual_seed(seed);b=new.CommandReferenceGCN(nf,3,role)
                        self.assertTrue(compare(initial,b.state_dict())['pass']);self.assertTrue(compare(init_rng,rng())['pass'])
                        # Non-contiguous command histories and telemetry view;
                        # zero-command entity tests null-only graph behaviour.
                        a.active_commands[0]=False;a.active_commands[1,::2]=False;b.load_state_dict(a.state_dict())
                        torch.manual_seed(1017);x=torch.randn(8,65,nf);x[:,-1,1:]=0
                        item=(x,torch.arange(8)%3,x[:,-1,0].clone());start=rng()
                        oa=torch.optim.Adam(a.parameters(),lr=.001,foreach=False,fused=False)
                        ob=torch.optim.Adam(b.parameters(),lr=.001,foreach=False,fused=False)
                        old_trace=[step(a,oa,item,inject_batch,rng) for _ in range(2)]
                        restore(start);new_trace=[step(b,ob,item,inject_batch,rng) for _ in range(2)]
                        for i,(aa,bb) in enumerate(zip(old_trace,new_trace)):
                            report=compare_step(aa,bb)
                            self.results.append({'features':nf,'role':role,'seed':seed,'update':i+1,
                                'pass':report['pass'],'bitwise_equal':report['bitwise_equal'],
                                'Adam_FP64_equation_errors':[aa['independent_Adam_FP64_equation_max_absolute_error'],bb['independent_Adam_FP64_equation_max_absolute_error']],
                                'comparisons':{k:{n:v[n] for n in ['pass','bitwise_equal','max_absolute_error','max_relative_L2_error','failures']} for k,v in report['comparisons'].items()}})
                            self.assertTrue(report['pass'],str(report))
    def test_candidate_exact_replay_and_warm_checkpoint_all_roles(self):
        for nf in [55,25]:
            for role in ['graph','removed','fusion']:
                torch.manual_seed(43);m=new.CommandReferenceGCN(nf,3,role)
                op=torch.optim.Adam(m.parameters(),lr=.001,foreach=False,fused=False)
                x=torch.randn(8,65,nf);item=(x,torch.arange(8)%3,x[:,-1,0].clone())
                step(m,op,item,inject_batch,rng)
                payload=io.BytesIO();torch.save({'model':m.state_dict(),'adam':op.state_dict(),'rng':rng()},payload)
                payload.seek(0);saved=torch.load(payload,weights_only=True)
                expected=step(m,op,item,inject_batch,rng)
                restored=new.CommandReferenceGCN(nf,3,role);restored.load_state_dict(saved['model'])
                opt=torch.optim.Adam(restored.parameters(),lr=.001,foreach=False,fused=False);opt.load_state_dict(saved['adam'])
                restore(saved['rng']);actual=step(restored,opt,item,inject_batch,rng)
                self.assertTrue(compare_step(expected,actual,exact=True)['pass'])
    def test_native_warm_adam_comparison_all_roles(self):
        # Private equation comparison, never production checkpoint migration.
        for role in ['graph','removed','fusion']:
            torch.manual_seed(17);a=old.CommandReferenceGCN(55,3,role)
            x=torch.randn(8,65,55);item=(x,torch.arange(8)%3,x[:,-1,0].clone())
            op=torch.optim.Adam(a.parameters(),lr=.001,foreach=False,fused=False)
            step(a,op,item,inject_batch,rng)
            saved=cpu_copy({'model':a.state_dict(),'adam':op.state_dict(),'rng':rng()})
            b=new.CommandReferenceGCN(55,3,role);b.load_state_dict(saved['model'],strict=True)
            ob=torch.optim.Adam(b.parameters(),lr=.001,foreach=False,fused=False);ob.load_state_dict(saved['adam'])
            restore(saved['rng']);aa=step(a,op,item,inject_batch,rng)
            restore(saved['rng']);bb=step(b,ob,item,inject_batch,rng)
            self.assertTrue(compare_step(aa,bb)['pass'])
    def test_eval_output_and_input_gradient_all_roles(self):
        for nf in [55,25]:
            for role in ['graph','removed','fusion']:
                torch.manual_seed(17);a=old.CommandReferenceGCN(nf,3,role).eval();b=new.CommandReferenceGCN(nf,3,role).eval();b.load_state_dict(a.state_dict())
                x=torch.randn(4,65,nf,requires_grad=True);xx=x.detach().clone().requires_grad_();e=torch.arange(4)%3
                ra,la=a(x,e);rb,lb=b(xx,e)
                self.assertTrue(compare((ra,la),(rb,lb),'outputs')['pass'])
                (ra.sum()+la.sum()).backward();(rb.sum()+lb.sum()).backward()
                self.assertTrue(compare(x.grad,xx.grad,'gradients')['pass'])
                self.assertTrue(torch.equal(xx.grad[:,-1,1:],torch.zeros_like(xx.grad[:,-1,1:])))
    def test_gate_rejects_changes(self):
        self.assertFalse(compare(torch.tensor([1.]),torch.tensor([1.01]),'outputs')['pass'])
        self.assertFalse(compare({'step':torch.tensor(1.)},{'step':torch.tensor(2.)},'adam')['pass'])
        self.assertFalse(compare(torch.tensor([1.]),torch.tensor([1.000001]),'exact')['pass'])

if __name__=='__main__':unittest.main()
