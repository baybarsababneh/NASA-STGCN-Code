"""Execution amendment E01: exact memoization and zero-SD no-op elimination.

No detector outputs are changed. The native threshold search cannot update
epsilon when its source-computed residual SD is zero: every candidate score
has a non-finite SD reduction or fails the coverage constraint. Thus retain
its already-initialized fallback unchanged. Normal cases execute source code.
"""
from pathlib import Path
import hashlib,json,time
from functools import lru_cache
import run_extension as e
import numpy as np
R=Path(__file__).resolve().parent;O=R/'outputs';O.mkdir(exist_ok=True)
original=e.native.ErrorWindow.find_epsilon
stats={'native_threshold_source_evaluations':0,'exact_threshold_cache_hits':0,'zero_SD_no_op_eliminations':0,'whole_prediction_cache_hits':0}
threshold_cache={};forecast_cache={};forecast_channel=[None]
def key_for(w,inverse):
 a=w.e_s_inv if inverse else w.e_s
 return (inverse,a.dtype.str,a.shape,hashlib.sha256(a.tobytes()).digest(),float(w.mean_e_s),float(w.sd_e_s),w.config.error_buffer)
def fast_find(w,inverse=False):
 if w.sd_e_s==0:
  stats['zero_SD_no_op_eliminations']+=1;return
 k=key_for(w,inverse)
 eps_name='epsilon_inv' if inverse else 'epsilon';sd_name='sd_threshold_inv' if inverse else 'sd_threshold'
 if k in threshold_cache:
  stats['exact_threshold_cache_hits']+=1;eps,sd=threshold_cache[k];setattr(w,eps_name,eps);setattr(w,sd_name,sd);return
 original(w,inverse=inverse);stats['native_threshold_source_evaluations']+=1
 if len(threshold_cache)>100000:threshold_cache.clear()
 threshold_cache[k]=(getattr(w,eps_name),getattr(w,sd_name))

def synthetic_window(a):
 w=object.__new__(e.native.ErrorWindow);w.e_s=np.asarray(a,float);w.mean_e_s=np.mean(w.e_s);w.sd_e_s=np.std(w.e_s)
 w.e_s_inv=2*w.mean_e_s-w.e_s;w.epsilon=w.mean_e_s+12*w.sd_e_s;w.epsilon_inv=w.epsilon
 w.sd_threshold=12.;w.sd_threshold_inv=12.;w.sd_lim=12.;w.config=e.types.SimpleNamespace(error_buffer=100)
 return w

checks=[]
import warnings
for name,a in [('zero',np.zeros(2100)),('constant_positive',np.ones(2100)*.125),
 ('constant_negative',np.ones(2100)*-.125),('near_underflow',np.linspace(0,1e-200,2100)),
 ('ramp',np.linspace(0,1,2100)),('single_spike',np.r_[np.zeros(1000),1.,np.zeros(1099)]),
 ('smooth',.1+.05*np.sin(np.arange(2100)/30))]:
 for inverse in [False,True]:
  a1=synthetic_window(a);a2=synthetic_window(a)
  with warnings.catch_warnings():warnings.simplefilter('ignore');original(a1,inverse);fast_find(a2,inverse)
  attr='epsilon_inv' if inverse else 'epsilon';sd='sd_threshold_inv' if inverse else 'sd_threshold'
  assert getattr(a1,attr)==getattr(a2,attr) and getattr(a1,sd)==getattr(a2,sd),(name,inverse)
  checks.append({'case':name,'inverse':inverse,'exact_threshold_and_grid_selection':True})

e.native.ErrorWindow.find_epsilon=fast_find
original_detect=e.native_detect
def forecast_memo(c,te,pred):
 if c!=forecast_channel[0]:forecast_cache.clear();forecast_channel[0]=c
 k=(c,pred.dtype.str,pred.shape,hashlib.sha256(pred.tobytes()).digest())
 if k in forecast_cache:stats['whole_prediction_cache_hits']+=1;return forecast_cache[k]
 d=original_detect(c,te,pred);forecast_cache[k]=d;return d
e.native_detect=forecast_memo
note={'amendment':'E01_EXACT_EXECUTION_CACHE','analysis_source_remains_unchanged':True,
 'original_analysis_freeze_sha256':e.sha(O/'ANALYSIS_FREEZE.json'),
 'wrapper_sha256':e.sha(Path(__file__)),'statistical_parameters_changed':False,
 'reason':'Initial direct replay was slow on zero-residual-variance windows; stopped after two project streams, without completing or choosing new metric results.',
 'execution_changes':['Memoize exact signal-byte-identical threshold searches and identical forecast inputs.',
  'Skip zero-source-SD threshold searches that provably cannot replace the initialized fallback.'],
 'reference_checks':checks,'conditional_original_process_batches_control_flow':'Retained; threshold computations use the verified equivalent execution wrapper.',
 'no_tolerance_relaxation':True}
(O/'EXECUTION_AMENDMENT_E01.json').write_text(json.dumps(note,indent=2))
if __name__=='__main__':
 t=time.monotonic();e.main();stats['elapsed_seconds']=time.monotonic()-t
 (O/'EXECUTION_CACHE_STATS.json').write_text(json.dumps(stats,indent=2))
 print(json.dumps(stats,indent=2))
