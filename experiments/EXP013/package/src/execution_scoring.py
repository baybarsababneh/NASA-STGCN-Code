"""Memoise complete, verified score calls without changing frozen score arithmetic."""
import hashlib,time
from pathlib import Path
import numpy as np
from core import ROOT,sha,digest,read,write,npz,arrays,seal,manifest
import execution_runtime

class ScoreMemoizer:
    def __init__(self,work,original):self.work=Path(work);self.original=original
    def __call__(self,model,kind,a,entity,start,end,device,paths,cfg,seed=17):
        selected=getattr(model,'_execution_selected_identity',None)
        if selected is None:
            # Original isolated science tests may supply an uncommitted model;
            # no cache can be authorised without a verified selected identity.
            return self.original(model,kind,a,entity,start,end,device,paths,cfg,seed)
        look=64 if kind in ['graph','removed','fusion'] else 5 if kind=='gdn' else 191
        if start<look or end>len(a) or start>=end:raise ValueError('Invalid causal score support')
        # Hash ONLY bytes actually visible to this score call. In particular,
        # selection does not inspect quarantined CONF or later TEST values.
        visible=np.ascontiguousarray(a[start-look:end])
        h=hashlib.sha256();h.update(str(visible.dtype).encode());h.update(str(visible.shape).encode());h.update(memoryview(visible).cast('B'))
        identity={'selected':selected,'package_manifest_sha256':sha(ROOT/'PACKAGE_MANIFEST.json'),
            'DATA_FREEZE_sha256':sha(self.work/'DATA_FREEZE.json'),'kind':kind,'entity':entity,
            'start':start,'end':end,'device':device,'configuration':cfg,'seed':seed,'visible_input_sha256':h.hexdigest()}
        folder=self.work/'score_call_cache'/digest(identity)
        if (folder/'COMMIT.json').exists():
            receipt=manifest(folder,'COMMIT.json')
            if receipt['identity']!=identity:raise RuntimeError('Score-call cache identity differs')
            values=arrays(folder/'values.npz')
            if set(values)!={'scores'}:raise RuntimeError('Score-call cache schema differs')
            result=values['scores'];expected=(end-start,a.shape[1]) if kind=='gdn' else (end-start,)
            if result.shape!=expected or not np.isfinite(result).all():raise RuntimeError('Score-call cache support/nonfinite error')
            model.eval();return result
        active=execution_runtime.ACTIVE;beg=time.monotonic()
        if active is not None and hasattr(active,'log_record'):
            active.log_record({'phase':'SCORE_CALL_START','job':selected['job'],'kind':kind,'samples':end-start,'seed':seed,'completed_fits':active.completed()})
        result=self.original(model,kind,a,entity,start,end,device,paths,cfg,seed)
        npz(folder/'values.npz',scores=result);write(folder/'identity.json',identity)
        seal(folder,'COMMIT.json',{'identity':identity})
        if active is not None and hasattr(active,'log_record'):
            active.log_record({'phase':'SCORE_CALL_COMMITTED','job':selected['job'],'samples':end-start,'seconds':round(time.monotonic()-beg,3),'completed_fits':active.completed()})
        execution_runtime.checkpoint(self.work,'completed_score_call',force=False)
        return result
