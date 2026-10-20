"""Retained 22-fit fictional integration, under the E02 backup and score hooks."""
import importlib.util, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pipeline, upstream
from simple_run import Runner

spec=importlib.util.spec_from_file_location('original_fixture',Path(__file__).with_name('test_pipeline_fixture.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)

def managed(fn):
    def call(work,*args,**kwargs):
        work=Path(work);r=Runner(work,work.parent/'FAKE_DRIVE_E02',work.parent/'E02_CACHE','cpu')
        with r.active(),r.scientific_hooks():
            result=fn(work,*args,**kwargs);r.backup('verified_stage_complete',force=True)
        return result
    return call

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--official-work',type=Path,required=True);a=p.parse_args()
    paths=upstream.prepare(a.official_work)
    with patch.object(pipeline,'fit_and_select',managed(pipeline.fit_and_select)),patch.object(pipeline,'confirm_all',managed(pipeline.confirm_all)),patch.object(pipeline,'export_test',managed(pipeline.export_test)):
        fixture.main(paths)
    print('PASS E02 fictional 22-fit integration with automatic durable backup and unchanged final scientific ordering.')
