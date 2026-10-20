import unittest,sys
import numpy as np
from detectors import robust_scale,score,calibrate,apply,burden,stress
from run_exp009 import choose,table_summary,candidates,evaluate

class Checks(unittest.TestCase):
    def test_causality(self):
        rng=np.random.default_rng(9);x=rng.normal(size=200);y=x.copy();y[100:]+=100
        for f in ('absolute','ewma10','multiscale'):np.testing.assert_array_equal(score(x,0,1,f)[:100],score(y,0,1,f)[:100])
    def test_multiscale_shift(self):
        x=np.r_[np.zeros(64),np.ones(32)];self.assertAlmostEqual(score(x,0,1,'multiscale')[-1],np.sqrt(32))
    def test_reference_only(self):
        x=np.sin(np.arange(200));y=x.copy();y[100:]+=100;ref=np.arange(200)<100
        for f in ('absolute','ewma10','multiscale','historical_raw99','historical_ewma995'):
            c=dict(family=f,h=4);self.assertEqual(calibrate(x,ref,c),calibrate(y,ref,c))
    def test_strict_ties(self):
        s,f=apply(np.ones(10),dict(family='absolute',location=0,scale=1,threshold=1));self.assertFalse(f.any())
    def test_boundary_start(self):
        f=np.array([0,1,1,0,1],bool);m=np.array([0,0,1,1,1],bool)
        self.assertEqual(burden(f,m)['alarm_starts'],1)
    def test_zero_scale(self):
        self.assertEqual(robust_scale(np.zeros(40)),(0.,1e-6))
    def test_stress_no_inplace_mutation(self):
        e=np.zeros(160);ref=np.arange(160)<80;mask=np.arange(160)>=100
        cal=calibrate(e,ref,dict(family='absolute',h=2));rows=stress(e,mask,cal,1.,True)
        self.assertEqual(len(rows),72);self.assertTrue((e==0).all());self.assertTrue(all(r['incremental_hit'] for r in rows if r['family']!='variance32'))
    def test_false_alarm_not_rewarded(self):
        e=np.ones(160)*10;mask=np.arange(160)>=100
        cal=dict(family='absolute',location=0,scale=1,threshold=1)
        self.assertTrue(all(not r['incremental_hit']for r in stress(e,mask,cal,1)))
    def test_no_feasible_candidate(self):
        cs=[dict(id='x')];r=dict(mission='MSL',candidate='x',nominal_ok=False,sensitivity_ok=True)
        self.assertEqual(choose([r],cs),dict(MSL=None,SMAP=None))
    def test_deterministic_choice(self):
        cs=[dict(id='a'),dict(id='b')];base=dict(mission='MSL',nominal_ok=True,sensitivity_ok=True,worst_family_hit=.5,mean_family_hit=.7,mean_starts_per_1000=1,mean_occupancy=.01)
        self.assertEqual(choose([dict(**base,candidate='b'),dict(**base,candidate='a')],cs)['MSL']['id'],'a')
    def test_reserved_audit_not_used_by_selection(self):
        p=dict(job=dict(mission='MSL',channel='synthetic',seed=17,fold='canonical'),reference_result=dict(num_values=1000,fit_end=600,stop_end=800))
        a=dict(target_index=np.arange(600,800),target=np.sin(np.arange(200)),prediction=np.zeros(200),persistence=np.zeros(200))
        c=dict(id='absolute_h4',family='absolute',h=4)
        first=evaluate(p,a,'gcn',c,'selection',1.)
        a['target'][150:]+=10000
        self.assertEqual(first,evaluate(p,a,'gcn',c,'selection',1.))

    def test_known_delay(self):
        e=np.zeros(160);mask=np.arange(160)>=100;cal=dict(family='absolute',location=0,scale=1,threshold=1)
        r=stress(e,mask,cal,1)[0];self.assertTrue(r['incremental_hit']);self.assertEqual(r['delay_censored'],0)

if __name__=='__main__':unittest.main(verbosity=2)
