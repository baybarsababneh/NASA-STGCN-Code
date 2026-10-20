"""CPU/synthetic regression receipts; never an A100 or new NASA result."""
import os,sys
os.environ['MPLBACKEND']='Agg';os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT),str(ROOT/'tests')]
import argparse,time,unittest,json,platform
from integrity import write,sha,configure

class Tee:
    def __init__(self,a,b):self.a,self.b=a,b
    def write(self,s):self.a.write(s);self.b.write(s);self.flush()
    def flush(self):self.a.flush();self.b.flush()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);ap.add_argument('--log',type=Path)
    a=ap.parse_args();original_out,original_err=sys.stdout,sys.stderr;log=None
    if a.log:
        a.log.parent.mkdir(parents=True,exist_ok=True);log=a.log.open('w');sys.stdout=Tee(original_out,log);sys.stderr=Tee(original_err,log)
    env=configure('cpu',True);start=time.monotonic()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_*.py');result=unittest.TextTestRunner(verbosity=2).run(suite)
    from test_exp015 import ScientificTests,EngineeringTests
    report={'status':'PASS_CPU_REGRESSION' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'seconds':time.monotonic()-start,
            'environment':env,'actual_A100_execution':False,'real_EXP013_checkpoint_replay':False,
            'new_NASA_scientific_results':False,'fictional_model_initialisation_is_training':False,
            'saved_frozen_detector_regression_streams':8,'full_native_CPU_shapes':[[256,65,55],[256,65,25]],
            'numerical_measurements':ScientificTests.measurements,'miniature_experiment':EngineeringTests.miniature,
            'source_sha256':{str(p.relative_to(ROOT)):sha(p) for folder in ['src','tests','vendor'] for p in (ROOT/folder).rglob('*.py')},
            'unexecuted_production_requirements':['18 real EXP013 checkpoint loads/state matches','729 real native score/flag/issue bitwise replays','real A100 40/80 GB full-shape memory/restoration/throughput','729 new NASA residual/Hundman streams and post-seal evaluation']}
    write(a.report,report)
    if log:sys.stdout,sys.stderr=original_out,original_err;log.close()
    print(json.dumps({k:report[k] for k in ['status','tests_run','failures','errors','seconds','actual_A100_execution']},sort_keys=True))
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
