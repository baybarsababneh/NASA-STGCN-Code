"""Synthetic workflow/commit tests. No CUDA and no NASA inference."""
import json,tempfile,sys
from pathlib import Path
import numpy as np
from replay_io import commit,read_commit
from nominal_study import records_for_trace,summarise
ROOT=Path(__file__).resolve().parents[1]
class LeaseStub:
    def check(self):pass

def main():
    plan=json.loads((ROOT/'plan.json').read_text());assert len(plan)==534
    assert sum(j['job']['stage']=='robust' for j in plan)==48
    assert sum(j['job']['stage']=='ablation' for j in plan)==243
    assert sum(j['job']['stage']=='canonical' for j in plan)==243
    rows=[];seen=set();rng=np.random.default_rng(8008)
    for meta in plan:
        if meta['job']['stage']!='robust':continue
        r=meta['reference_result'];t=np.arange(r['fit_end'],r['assess_end']);y=rng.normal(size=len(t)).astype(np.float32)
        a={'target_index':t,'target':y,'prediction':np.zeros_like(y),'persistence':np.r_[0,y[:-1]]}
        rows+=records_for_trace(meta,a,'gcn');key=(meta['job']['channel'],meta['job']['fold'])
        if key not in seen:rows+=records_for_trace(meta,a,'persistence');seen.add(key)
    assert len(rows)==288 and len(seen)==24
    with tempfile.TemporaryDirectory() as temp:
        p=Path(temp);summarise(rows,p/'tables')
        # Save/resume is tested on synthetic arrays, never parent predictions.
        meta=plan[0];r={'job_id':meta['job_id'],'job':meta['job'],'parent_archive_sha256':meta['archive_sha256']}
        a={'target_index':np.arange(3),'target':np.zeros(3),'prediction':np.ones(3),'persistence':np.zeros(3)}
        assert read_commit(p,meta,'synthetic') is None
        commit(p,meta,'synthetic',r,a,LeaseStub());rr,aa=read_commit(p,meta,'synthetic');assert rr==r
        assert all(np.array_equal(aa[k],a[k]) for k in a)
        try:read_commit(p,meta,'wrong_hash')
        except AssertionError:pass
        else:raise AssertionError('Wrong run accepted')
        archive=p/'replays'/f"{meta['job_id']}.zip";archive.write_bytes(archive.read_bytes()+b'corruption')
        try:read_commit(p,meta,'synthetic')
        except AssertionError:pass
        else:raise AssertionError('Corrupt commit accepted')
    print(json.dumps({'status':'PASS','plan_jobs':534,'synthetic_nominal_rows':288,'commit_resume_corruption_tests':True,'uses_NASA_prediction_values':False,'uses_NASA_labels':False,'CUDA_runtime_tested':False},indent=2))
if __name__=='__main__':main()
