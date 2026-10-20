"""Transactional inference records, usable without PyTorch."""
import io,json,zipfile
import numpy as np
from common import sha,atomic_bytes,write_json
from json_e01 import dumps_e01

def commit(out,uid,identity,record,arrays,lease):
    buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);b=io.BytesIO()
    with zipfile.ZipFile(b,'w',zipfile.ZIP_STORED)as z:z.writestr('record.json',dumps_e01(record));z.writestr('arrays.npz',buffer.getvalue())
    lease.check();p=out/'units'/f'{uid}.zip';atomic_bytes(p,b.getvalue());write_json(p.with_suffix('.done.json'),dict(identity=identity,sha256=sha(p)))

def read(out,uid,identity):
    p=out/'units'/f'{uid}.zip';m=p.with_suffix('.done.json')
    if not m.exists():return None
    assert json.loads(m.read_text())==dict(identity=identity,sha256=sha(p)),'Resume integrity mismatch'
    with zipfile.ZipFile(p)as z:
        record=json.loads(z.read('record.json'))
        with np.load(io.BytesIO(z.read('arrays.npz')),allow_pickle=False)as n:arrays={k:n[k].copy()for k in n.files}
    assert all(np.isfinite(v).all()for v in arrays.values());return record,arrays

