import numpy as np
from pilot import features,fit,evaluate,choose,summarise

def main():
 a={'target':np.arange(60,dtype=float)/10,'prediction':np.arange(60,dtype=float)/11,'persistence':np.r_[0,np.arange(59)/10]}
 for arm in ('residual','fusion','removed','persistence'):
  x=features(a,arm);b={k:v.copy() for k,v in a.items()}
  for k in b:b[k][40:]+=100
  assert np.array_equal(x[:40],features(b,arm)[:40])
  assert np.isfinite(features({k:np.zeros_like(v) for k,v in a.items()},arm)).all()
 rng=np.random.default_rng(9);x=np.column_stack([np.ones(100),np.r_[rng.normal(-2,.1,50),rng.normal(2,.1,50)]]);y=np.r_[np.zeros(50),np.ones(50)];w=np.full(100,.01)
 b,diag=fit(x,y,w);assert np.all((x@b>0)==y);b2,_=fit(x,y,w);assert np.array_equal(b,b2)
 rows=[]
 for ch in ('a','b','c'):
  for seed in (17,29,43):
   es=[dict(family=f,amplitude=2.,incremental=True) for f in ('spike','offset32','ramp32') for i in range(4)]
   rows.append(dict(channel=ch,seed=seed,occupancy=.001,starts=1.,events=es))
 assert summarise(rows)['feasible'] and choose({0.:rows,2.:rows})==2.
 for r in rows:r['occupancy']=.2
 assert choose({0.:rows}) is None
 # Zero nominal alarms must not rescue zero sensitivity.
 for r in rows:
  r['occupancy']=0.;r['starts']=0.
  for e in r['events']:e['incremental']=False
 assert not summarise(rows)['feasible']
 print('PASS: causal features, zero-scale safety, deterministic convex fit, ranking, no-fallback and no-alarm degeneracy checks')
if __name__=='__main__':main()
