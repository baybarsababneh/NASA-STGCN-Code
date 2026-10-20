"""Retained original fictional E2E, now under actual E01 backup/cache hooks."""
import sys,time,importlib.util,contextlib
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pipeline,scoring,execution_runtime,upstream
from recovery import Recovery,LocalTransport
from execution_scoring import ScoreMemoizer

spec=importlib.util.spec_from_file_location('retained_pipeline_fixture',Path(__file__).with_name('test_pipeline_fixture.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
original_scores=scoring.scores

def managed(fn):
    def invoke(work,*args,**kwargs):
        work=Path(work);backend=LocalTransport(work.parent/'FAKE_DRIVE_ONLY')
        recovery=Recovery(backend,work,work.parent/'FAKE_BACKUP_CACHE',{'fictional_fixture_only':True})
        class Active:
            def __init__(self):self.last=time.monotonic()
            def checkpoint_due(self):return time.monotonic()-self.last>=5
            def progress(self,*a,**kw):pass
            def backup(self,reason,force=False):
                if force or self.checkpoint_due():recovery.backup(reason);self.last=time.monotonic()
        active=Active();cache=ScoreMemoizer(work,original_scores);previous=execution_runtime.ACTIVE
        try:
            execution_runtime.ACTIVE=active
            with patch.object(scoring,'scores',cache),patch.object(pipeline,'scores',cache):
                result=fn(work,*args,**kwargs)
            recovery.backup('verified_stage_complete')
            return result
        finally:execution_runtime.ACTIVE=previous
    return invoke

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--official-work',type=Path,required=True);args=parser.parse_args();paths=upstream.prepare(args.official_work)
    with patch.object(pipeline,'fit_and_select',managed(pipeline.fit_and_select)),patch.object(pipeline,'confirm_all',managed(pipeline.confirm_all)),patch.object(pipeline,'export_test',managed(pipeline.export_test)):
        fixture.main(paths)
    print('PASS MANAGED E01 E2E: original fictional 22-job pipeline, verified incremental recovery callbacks, score-call caching, study/CONF/TEST/label ordering and unchanged evaluator.')
