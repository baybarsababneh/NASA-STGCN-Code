"""Synthetic workflow integration: genuine stage/commit/rank/resume logic; mocked detector outputs.
Does not select a detector on project residuals. Independent numerical tests cover the detector.
"""
import contextlib,io,json,tempfile,types,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_exp009 as r


def main():
    # Only split/roster metadata is used. Every target/prediction below is synthetic zero.
    with zipfile.ZipFile(r.ROOT/'data/EXP008_results.zip') as z:plan=json.loads(z.read('PLAN.json'))
    streams=[]
    for p in plan:
        rr=p['reference_result'];t=np.arange(rr['fit_end'],rr['stop_end'] if p['job']['fold']=='canonical' else rr['assess_end'])
        streams.append((p,dict(target_index=t,target=np.zeros(len(t)),prediction=np.zeros(len(t)),persistence=np.zeros(len(t)))))
    def fake(p,a,model,c,stage,scale):
        ref,assess=r.masks(p,a,stage)
        meta=dict(mission=p['job']['mission'],channel=p['job']['channel'],fold=p['job']['fold'],seed=0 if model=='persistence' else p['job']['seed'],model=model,candidate=c['id'],stage=stage)
        nom=dict(**meta,occupancy=0.,starts_per_1000=0.)
        rows=[dict(**meta,family=f,amplitude=amp,incremental_hit=True)for f in ('spike','step8','step32','ramp32')for amp in (2,4,8)]
        return dict(nominal=nom,probes=rows)
    with tempfile.TemporaryDirectory(prefix='EXP009_SYNTHETIC_') as d:
        args=types.SimpleNamespace(input=r.ROOT/'data/EXP008_results.zip',output=d)
        with patch.object(r,'load_parent',return_value=streams),patch.object(r,'evaluate',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):r.run(args)
        close=json.loads((Path(d)/'STUDY_CLOSE.json').read_text());assert close['units']==4833;assert close['audit_stream_rows']==1917
        from verify_results import verify
        with contextlib.redirect_stdout(io.StringIO()):verify(d)
        chosen=json.loads((Path(d)/'DETECTOR_CANDIDATE_FREEZE.json').read_text());assert all(c['id']=='absolute_h2'for c in chosen['selected'].values())
        with patch.object(r,'load_parent',return_value=streams),patch.object(r,'evaluate',side_effect=AssertionError('Completed unit was recomputed')),contextlib.redirect_stdout(io.StringIO()):r.run(args)
        victim=next((Path(d)/'units').glob('*.json'));victim.write_text('{}')
        try:
            with patch.object(r,'load_parent',return_value=streams),contextlib.redirect_stdout(io.StringIO()):r.run(args)
        except (AssertionError,KeyError):pass
        else:raise AssertionError('Corrupt committed unit accepted')
    print(json.dumps(dict(status='PASS',synthetic_only=True,full_budget_units=4833,resume_without_recomputation=True,corruption_rejected=True,actual_detector_selection_run=False),indent=2))

if __name__=='__main__':main()
