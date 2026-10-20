"""Synthetic engineering tests only. --sklearn optionally checks the trusted API."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
from detector_metrics_v2 import *

def tests(compare_sklearn=False):
    checks=[]
    for y in ([1,0],[0,1]):assert average_precision(y,[.5,.5])==.5
    checks.append('AP ties and permutation invariant')
    assert average_precision([1,1,0],[1,1,1])==2/3
    assert np.isnan(average_precision([0,0],[1,2]))
    assert average_precision([1,0,1],[3,1,2])==1
    checks.append('AP constant, perfect, no-positive conventions')
    rng=np.random.default_rng(8008)
    for n in [2,10,99,100,101,199,200,201]:
        x=rng.normal(size=n)
        for q in [.99,.995]:
            d=higher_threshold(x,q);assert d['threshold']==sorted(x)[int(np.ceil((n-1)*q))]
        for span in [1,10]:
            a=ewma(x,span);b=x.copy();b[n//2:]+=10;assert np.array_equal(a[:n//2],ewma(b,span)[:n//2])
    checks.append('Higher quantile order statistics and causal EWMA')
    assert np.array_equal(ewma([1,2,3],1),[1,2,3])
    assert np.allclose(ewma([0,11,0],10),[0,2,18/11])
    assert burden([1,1,0,1],True)['alarm_starts']==1
    assert burden([0,0])['longest_run']==0
    assert burden([1,1,1])['occupancy']==1
    d=higher_threshold([0]*35,.995);assert d['at_sample_max'] and d['calibration_exceedances']==0
    assert not (np.array([0])>d['threshold']).any()
    checks.append('Strict exceedance, constant scores and boundary run counting')
    y=[0,1,1,0,0,1,0];perfect=event_metrics(y,y);assert perfect['corrected_event_f0_5']==1
    on=event_metrics(y,[1]*len(y));assert on['event_recall']==1 and on['corrected_event_f0_5']==0
    off=event_metrics(y,[0]*len(y));assert off['corrected_event_f0_5']==0
    no_truth=event_metrics([0,0],[1,0]);assert np.isnan(no_truth['corrected_event_f0_5'])
    fragmented=event_metrics([0,1,1,1,0],[0,1,0,1,0]);assert fragmented['H']==1 and fragmented['U']==0
    assert np.isnan(event_metrics([1,1],[0,0])['corrected_event_f0_5'])
    assert np.isnan(event_metrics([1,1],[1,1])['corrected_event_f0_5'])
    checks.append('Event overlap, occupancy correction and explicit undefined cases')
    if compare_sklearn:
        from sklearn.metrics import average_precision_score
        for _ in range(200):
            y=rng.integers(0,2,50);s=rng.integers(0,5,50)
            assert np.isclose(average_precision(y,s),average_precision_score(y,s),atol=1e-14,rtol=0)
        checks.append('200 tied-score fixtures match scikit-learn')
    return {'status':'PASS','metric_version':VERSION,'checks':checks,'NASA_data_used':False}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sklearn',action='store_true');a=p.parse_args();print(json.dumps(tests(a.sklearn),indent=2))
