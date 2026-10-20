"""Read-only full-unit integrity verifier. No model or anomaly evaluation."""
import argparse,json,zipfile,io
from pathlib import Path
import numpy as np
from common import digest,sha
from mechanisms import cases
ROOT=Path(__file__).resolve().parents[1]

def verify(out):
    out=Path(out);m=json.loads((out/'OUTPUT_MANIFEST.json').read_text())
    for f,h in m.items():assert sha(out/f)==h,f
    freeze=json.loads((out/'RUN_FREEZE.json').read_text());rh=digest(freeze);plan=json.loads((ROOT/'plan.json').read_text());expected=set();nom=0;probes=0
    for p in plan:
        for c in [dict(id='nominal')]+cases(p['reference_result']['num_values']):
            ident=dict(runhash=rh,parent_job=p['job_id'],case=c['id']);uid=digest(ident)[:24];expected.add(uid)
            archive=out/'units'/f'{uid}.zip';assert json.loads(archive.with_suffix('.done.json').read_text())==dict(identity=ident,sha256=sha(archive))
            with zipfile.ZipFile(archive)as z:
                r=json.loads(z.read('record.json'))
                with np.load(io.BytesIO(z.read('arrays.npz')),allow_pickle=False)as a:assert all(np.isfinite(a[k]).all()for k in a.files)
            if c['id']=='nominal':
                assert r['kind']=='nominal';ref=r['parent_replay'];assert ref['job_id']==p['job_id'];assert all(v['exact_match']and v['repeat_prediction_exact']for v in ref['metrics'].values());nom+=1
            else:assert r['case']==c and r['parent_archive_sha256']==p['archive_sha256'];probes+=1
    assert expected=={p.name.removesuffix('.done.json')for p in (out/'units').glob('*.done.json')}
    assert nom==114 and probes==2964
    close=json.loads((out/'STUDY_CLOSE.json').read_text());assert close['status']=='COMPLETE_DIAGNOSTIC_NO_SELECTION'and close['units']==3078
    print(json.dumps(dict(status='PASS',nominal=nom,perturbed=probes,scope='unit integrity and exact nominal replay assertions; response interpretation requires scientific review'),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();verify(a.output)
