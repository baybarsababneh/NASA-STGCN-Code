import tempfile,json
from pathlib import Path
import numpy as np
from trace_io import commit,read
class Lease:
    def check(self):pass
with tempfile.TemporaryDirectory()as d:
    out=Path(d);ident=dict(case='synthetic',runhash='synthetic',parent_job='synthetic');arr=dict(prediction=np.array([1.,2.],np.float32));record=dict(value=np.int64(32))
    commit(out,'synthetic',ident,record,arr,Lease());r,a=read(out,'synthetic',ident);assert r=={'value':32};assert np.array_equal(a['prediction'],arr['prediction'])
    original=(out/'units/synthetic.zip').read_bytes();read(out,'synthetic',ident);assert (out/'units/synthetic.zip').read_bytes()==original
    (out/'units/synthetic.zip').write_bytes(b'corrupt')
    try:read(out,'synthetic',ident)
    except AssertionError:pass
    else:raise AssertionError('Corruption not rejected')
print('PASS: NumPy scalar commit, exact resume, unchanged bytes and corruption rejection')
