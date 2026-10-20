"""CPU-only tests of the engineering amendment, never A100 validation."""
import copy, importlib.util, sys, unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'package/src'))
import convolution_regression as cr
spec=importlib.util.spec_from_file_location('p01',ROOT/'EXP013_E03_P01_PARAMETER_GATE.py')
p01=importlib.util.module_from_spec(spec);spec.loader.exec_module(p01)

class Holder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.discriminator=torch.nn.Sequential(torch.nn.Linear(80,32))

def synthetic_pair():
    # A mathematical sensitivity example, NOT the actual failing GPU elements.
    torch.manual_seed(17);a=Holder();b=copy.deepcopy(a)
    oa=torch.optim.Adam(a.parameters(),lr=.001,foreach=False,fused=False)
    ob=torch.optim.Adam(b.parameters(),lr=.001,foreach=False,fused=False)
    ga=1e-9;eps=1e-8;wanted=6.02286309e-6
    fraction=ga/(ga+eps)+wanted/.001;gb=eps*fraction/(1-fraction)
    traces=[[],[]]
    for update in [1,2]:
        for model,op,gradient,trace in [(a,oa,ga,traces[0]),(b,ob,gb,traces[1])]:
            before=p01.snapshot(cr,model,op);op.zero_grad(set_to_none=True)
            for p in model.parameters():p.grad=torch.full_like(p,.001)
            if update==1:model.discriminator[0].weight.grad[0,0]=gradient
            gradients=cr.cpu_copy({n:p.grad for n,p in model.named_parameters()});op.step()
            trace.append({'outputs':torch.tensor([1.]),'injection':torch.tensor([0.]),
                'gradients':gradients,'parameters':cr.cpu_copy(model.state_dict()),
                'adam':cr.cpu_copy(op.state_dict()),'rng':torch.get_rng_state().clone(),
                '_p01_before':before})
    return list(zip(*traces))

class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    def test_Adam_epsilon_reproduces_narrow_stable_failure(self):
        for a,b in synthetic_pair():
            old=cr.compare_step(a,b);self.assertFalse(old['pass'])
            self.assertEqual(old['comparisons']['parameters']['failures'],['root.'+p01.TARGET])
            new,d=p01.amend_comparison(cr,cr.compare_step,a,b)
            self.assertTrue(new['pass']);self.assertTrue(d['amended_gate_used'])
            for category in ['outputs','injection','gradients','adam','rng']:
                self.assertEqual(new['comparisons'][category],old['comparisons'][category])
            self.assertTrue(d['Adam_equations_explain_difference'])
            self.assertLess(d['maximum_absolute_difference'],7.5e-6)
            self.assertGreater(d['maximum_absolute_difference'],5e-6)
            self.assertEqual(d['original_failed_element_count'],1)
    def test_strict_replay_and_RNG_unchanged(self):
        a,b=synthetic_pair()[0]
        self.assertTrue(p01.amend_comparison(cr,cr.compare_step,b,copy.deepcopy(b),True)[0]['pass'])
        bad=copy.deepcopy(b);bad['rng'][0]^=1
        self.assertFalse(p01.amend_comparison(cr,cr.compare_step,b,bad,True)[0]['pass'])
    def test_all_other_model_roles_use_original_validation(self):
        from types import SimpleNamespace
        calls=[]
        def original(*args):calls.append(args);return 'UNCHANGED'
        fake=SimpleNamespace(validate_role=original,step=cr.step,compare_step=cr.compare_step)
        p01.install(fake,None,Path('/never_touched'))
        for mission,role in [('MSL','fusion'),('SMAP','graph'),('SMAP','removed'),('SMAP','gdn'),('SMAP','catch')]:
            self.assertEqual(fake.validate_role(None,None,None,mission,role,'cpu'),'UNCHANGED')
        self.assertEqual(len(calls),5)
    def test_other_parameter_failure_not_waived(self):
        a,b=synthetic_pair()[0];b['parameters']['discriminator.0.bias'][0]+=.001
        self.assertFalse(p01.amend_comparison(cr,cr.compare_step,a,b)[0]['pass'])
    def test_other_category_failure_not_waived(self):
        for category in ['outputs','injection','gradients','adam','rng']:
            a,b=synthetic_pair()[0]
            if category=='outputs':b[category][0]+=.01
            elif category=='injection':b[category][0]+=1
            elif category=='gradients':b[category]['discriminator.0.bias'][0]+=.01
            elif category=='adam':
                pid=b['_p01_before']['name_ids']['discriminator.0.bias'];b[category]['state'][pid]['exp_avg'][0]+=.01
            else:b[category][0]^=1
            self.assertFalse(p01.amend_comparison(cr,cr.compare_step,a,b)[0]['pass'],category)
    def test_unexplained_parameter_error_rejected(self):
        a,b=synthetic_pair()[0];b['parameters'][p01.TARGET][0,0]+=7e-6
        with self.assertRaises(AssertionError):p01.amend_comparison(cr,cr.compare_step,a,b)
    def test_different_cold_initialisation_rejected(self):
        a,b=synthetic_pair()[0];b['_p01_before']['parameters'][p01.TARGET][0,0]+=5e-8
        with self.assertRaises((RuntimeError,AssertionError)):p01.amend_comparison(cr,cr.compare_step,a,b)
    def test_allowance_is_fixed_and_not_unbounded(self):
        a,b=synthetic_pair()[0]
        # Build a larger, correctly explained Adam discrepancy. It must still
        # fail the fixed 7.5e-6 ceiling, even though the equation is valid.
        grad=1.2e-9;pid=b['_p01_before']['name_ids'][p01.TARGET]
        b['gradients'][p01.TARGET][0,0]=grad
        b['adam']['state'][pid]['exp_avg'][0,0]=.1*grad
        b['adam']['state'][pid]['exp_avg_sq'][0,0]=.001*grad*grad
        b['parameters'][p01.TARGET][0,0]=b['_p01_before']['parameters'][p01.TARGET][0,0]-.001*grad/(grad+1e-8)
        self.assertFalse(p01.amend_comparison(cr,cr.compare_step,a,b)[0]['pass'])
    def test_real_small_SMAP_fusion_two_updates_and_exact_replay(self):
        native=cr.module(ROOT/'package/frozen_originals/architectures.py','old')
        candidate=cr.module(ROOT/'package/src/architectures.py','new')
        # Actual frozen synthetic injection function, extracted unchanged.
        import ast
        tree=ast.parse((ROOT/'package/src/data.py').read_text())
        function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='inject_batch')
        ns={'torch':torch};exec(compile(ast.Module(body=[function],type_ignores=[]),'frozen_injection','exec'),ns)
        torch.manual_seed(17);model=native.CommandReferenceGCN(25,3,'fusion');initial=cr.cpu_copy(model.state_dict())
        x=torch.randn(8,65,25);item=(x,torch.arange(8)%3,x[:,-1,0].clone());start=torch.get_rng_state()
        traces=[]
        for module in [native,candidate,candidate]:
            m=module.CommandReferenceGCN(25,3,'fusion');m.load_state_dict(initial)
            op=torch.optim.Adam(m.parameters(),lr=.001,foreach=False,fused=False);torch.set_rng_state(start)
            trace=[]
            for _ in range(2):
                before=p01.snapshot(cr,m,op)
                t=cr.step(m,op,item,ns['inject_batch'],lambda:torch.get_rng_state().clone());t['_p01_before']=before;trace.append(t)
            traces.append(trace)
        for a,b in zip(traces[0],traces[1]):self.assertTrue(p01.amend_comparison(cr,cr.compare_step,a,b)[0]['pass'])
        for a,b in zip(traces[1],traces[2]):self.assertTrue(p01.amend_comparison(cr,cr.compare_step,a,b,True)[0]['pass'])
        self.assertEqual(cr.TOLERANCES['parameters'],(5e-6,2e-5))

if __name__=='__main__':unittest.main(verbosity=2)
