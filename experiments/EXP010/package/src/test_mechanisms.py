"""CPU synthetic tests of telemetry interventions and causal accounting. No project scoring."""
import unittest,json
import numpy as np
from mechanisms import cases,perturb,response,nominal_context
from json_e01 import dumps_e01

class Tests(unittest.TestCase):
    def test_roster(self):
        c=cases(1000);self.assertEqual(len(c),26);self.assertEqual(sum(x['family']=='stuck32'for x in c),2)
    def test_support_and_commands(self):
        raw=np.arange(3000,dtype=np.float32).reshape(1000,3)
        for c in cases(1000):
            p=perturb(raw,c);np.testing.assert_array_equal(raw[:,1:],p[:,1:]);np.testing.assert_array_equal(raw[:c['onset']],p[:c['onset']]);np.testing.assert_array_equal(raw[c['end']:],p[c['end']:])
    def test_persistence_offset_adaptation(self):
        raw=np.zeros((1000,3),dtype=np.float32);c=next(v for v in cases(1000)if v['family']=='offset32')
        x=perturb(raw,c);t=np.arange(600,800)
        def arrays(r):return dict(target_index=t,target=r[t,0],prediction=r[t-1,0],persistence=r[t-1,0])
        stats,pol=response(arrays(raw),arrays(x),c,1000,'persistence')
        self.assertAlmostEqual(stats['response_ratio'],1/32)
        self.assertEqual(stats['post_event_residual_change_l1'],.5)
        json.loads(dumps_e01(dict(stats=stats,pol=pol)))
    def test_null_stuck(self):
        raw=np.zeros((1000,3),np.float32);c=cases(1000)[-1];x=perturb(raw,c);t=np.arange(600,800);a=dict(target_index=t,target=raw[t,0],prediction=raw[t,0],persistence=raw[t-1,0]);s,p=response(a,a,c,1000,'gcn');self.assertEqual(s['status'],'NULL_PERTURBATION');self.assertEqual(p,[])
    def test_causality_violation_rejected(self):
        raw=np.zeros((1000,3),np.float32);c=cases(1000)[0];x=perturb(raw,c);t=np.arange(600,800);base=dict(target_index=t,target=raw[t,0],prediction=raw[t,0],persistence=raw[t-1,0]);changed=dict(target_index=t,target=x[t,0],prediction=np.ones(200,np.float32),persistence=x[t-1,0])
        with self.assertRaises(AssertionError):response(base,changed,c,1000,'gcn')
    def test_causal_context(self):
        raw=np.zeros((1000,3),np.float32);raw[700:,1]=1;t=np.arange(600,800);a=dict(target_index=t,target=raw[t,0],prediction=raw[t,0],persistence=raw[t-1,0]);meta=dict(job=dict(role='fixed',mission='MSL',channel='synthetic',seed=17),reference_result=dict(num_values=1000,graph_sources=[1]))
        rows=nominal_context(a,raw,meta);r=[v for v in rows if v['phase']=='assessment'and v['model']=='gcn'and v['context_lag']==1 and v['command_active']][0];self.assertEqual(r['samples'],1)
if __name__=='__main__':unittest.main(verbosity=2)
