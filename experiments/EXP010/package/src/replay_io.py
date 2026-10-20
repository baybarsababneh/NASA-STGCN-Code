"""Transactional replay records; no model execution."""
import hashlib,io,json,zipfile
import numpy as np
from common import atomic_bytes,sha,write_json

def commit(out,meta,run_hash,record,arrays,lease):
    jid=meta['job_id'];buffer=io.BytesIO();np.savez_compressed(buffer,**arrays)
    raw=io.BytesIO()
    with zipfile.ZipFile(raw,'w',zipfile.ZIP_STORED) as z:z.writestr('record.json',json.dumps(record,sort_keys=True,allow_nan=False));z.writestr('predictions.npz',buffer.getvalue())
    final=out/'replays'/f'{jid}.zip';lease.check();atomic_bytes(final,raw.getvalue());h=sha(final)
    assert h==hashlib.sha256(raw.getvalue()).hexdigest();lease.check()
    write_json(final.with_suffix('.done.json'),dict(job_id=jid,run_hash=run_hash,parent_archive_sha256=meta['archive_sha256'],archive_sha256=h))

def read_commit(out,meta,run_hash):
    p=out/'replays'/f"{meta['job_id']}.zip";marker=p.with_suffix('.done.json')
    if not marker.exists():return None
    m=json.loads(marker.read_text());assert m==dict(job_id=meta['job_id'],run_hash=run_hash,parent_archive_sha256=meta['archive_sha256'],archive_sha256=sha(p)),'Resume hash/identity mismatch'
    with zipfile.ZipFile(p) as z:
        r=json.loads(z.read('record.json'));assert r['job_id']==meta['job_id'] and r['job']==meta['job']
        with np.load(io.BytesIO(z.read('predictions.npz')),allow_pickle=False) as n:a={k:n[k] for k in n.files}
    assert all(np.isfinite(v).all() for v in a.values())
    return r,a

