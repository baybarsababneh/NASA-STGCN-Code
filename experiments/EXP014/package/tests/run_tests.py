"""One bounded regression command. Test outputs are never production results."""
import os,sys
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1');sys.dont_write_bytecode=True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(CODE/'src'),str(CODE),str(CODE/'tests')]
import argparse,unittest,time,platform,json,hashlib
from integrity import write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path,required=True);ap.add_argument('--log',type=Path);a=ap.parse_args()
    if a.log:
        a.log.parent.mkdir(parents=True,exist_ok=True);log=a.log.open('w');sys.stdout=log;sys.stderr=log
    suite=unittest.defaultTestLoader.discover(str(CODE/'tests'),pattern='test_*.py')
    t=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
    import torch,numpy,pandas,torch_geometric
    sources={str(p.relative_to(CODE)):hashlib.sha256(p.read_bytes()).hexdigest() for d in ['src','tests','vendor'] for p in (CODE/d).rglob('*.py')}
    report={'status':'PASS_CPU_REGRESSION' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'seconds':time.monotonic()-t,'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':numpy.__version__,'pandas':pandas.__version__,'torch_geometric':torch_geometric.__version__},'computation_device':'cpu','actual_A100_execution':False,'real_frozen_GDN_NASA_checkpoint_replay':False,'new_NASA_GDN_Hundman_results_generated':False,'CPU_full_native_shapes':[[32,55,5],[32,25,5]],'saved_EXP012_detector_reference_streams_checked':6,'original_GDN_dense_oracle_tolerances':{'forward_input_gradient':{'atol':3e-6,'rtol':1e-5},'embedding_parameter_gradient':{'atol':5e-6,'rtol':3e-5}},'source_sha256':sources,'scope':['exact frozen detector/evaluator bytes','six saved historical detector output/window/issue references','residual construction and two-stage inverse scaling','causal forecast support/future invariance','native history before support cropping','pruning/startup/buffer/tail/short-stream rules','event/count/strict-point/timing definitions','native GDN full shapes/dense forward/gradient parity','exact checkpoint/output/RNG restoration and corrupt checkpoint detection','atomic unit publication and stale/corrupt output rejection','duplicate/traversal archive rejection','annotation-open/parse gate','two-generation verified backup/fallback/runtime mismatch rejection','complete four-channel synthetic workflow','interruption after commit, local runtime loss, verified restore/skip','resumed versus uninterrupted numerical equality','complete report/figure/final archive integrity'],'unexecuted_requirements':['real A10040 memory and throughput','both real frozen GDN checkpoint restorations','all81 bitwise native-reference replays','all81 new NASA target-residual detector streams']}
    from test_science import ScientificTests
    report['GDN_dense_oracle_measured_CPU_differences']=ScientificTests.numerical_diagnostics
    report['historical_frozen_EXP012_count_rows_checked']=486
    report['scope'].append('all 486 frozen EXP012 count rows and four mission-level event results')
    write(a.report,report)
    if a.log:log.close();sys.stdout=sys.__stdout__;sys.stderr=sys.__stderr__
    print(json.dumps({k:report[k] for k in ['status','tests_run','failures','errors','seconds','actual_A100_execution']},sort_keys=True))
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
