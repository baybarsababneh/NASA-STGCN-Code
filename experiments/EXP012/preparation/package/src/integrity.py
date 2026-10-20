"""Atomic commits, strict JSON, allowlists and single-writer execution."""
import hashlib, json, os, time, uuid, platform, subprocess, importlib.metadata
from pathlib import Path
import numpy as np

def normal(value):
    if isinstance(value, np.ndarray): return normal(value.tolist())
    if isinstance(value, np.generic): return normal(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): normal(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [normal(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        raise ValueError('Nonfinite JSON value must be explicitly classified, not silently saved')
    return value

def canonical(value): return json.dumps(normal(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value): return hashlib.sha256(canonical(value)).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def atomic(path, blob):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with open(tmp,'wb') as f: f.write(blob); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def write_json(path,value): atomic(path,json.dumps(normal(value),indent=2,sort_keys=True,allow_nan=False).encode())
def freeze(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=normal(value): raise RuntimeError('Frozen identity differs: '+str(path))
    else: write_json(path,value)

class Lease:
    def __init__(self,out): self.path=Path(out)/'ACTIVE_RUN.json'; self.token=uuid.uuid4().hex
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists(): raise RuntimeError('ACTIVE_RUN exists. Confirm the old runtime is terminated, then use the explicit notebook stale-lock release. Never run concurrent notebooks.')
        with open(self.path,'x') as f: json.dump({'token':self.token,'pid':os.getpid(),'utc':time.time()},f)
        return self
    def check(self):
        if json.loads(self.path.read_text())['token']!=self.token: raise RuntimeError('Lost single-writer lock')
    def __exit__(self,*args):
        if self.path.exists() and json.loads(self.path.read_text()).get('token')==self.token: self.path.unlink()

def commit(out,jid,identity,files):
    """Commit data first; the last marker is authoritative. Orphan files are never complete."""
    base=Path(out)/'units'/jid; base.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,blob in files.items():
        if Path(name).name!=name: raise ValueError(name)
        atomic(base/name,blob); hashes[name]=sha(base/name)
    write_json(base/'COMMIT.json',{'identity':identity,'files':hashes})
    return base
def read_commit(out,jid,identity):
    base=Path(out)/'units'/jid; marker=base/'COMMIT.json'
    if not marker.exists(): return None
    m=json.loads(marker.read_text())
    if m['identity']!=normal(identity): raise RuntimeError('Commit identity mismatch '+jid)
    actual={p.name for p in base.iterdir() if p.is_file() and '.tmp-' not in p.name and p.name!='COMMIT.json'}
    if actual!=set(m['files']): raise RuntimeError('Committed file roster mismatch '+jid)
    for n,h in m['files'].items():
        if sha(base/n)!=h: raise RuntimeError('Committed bytes corrupted '+jid+'/'+n)
    return base

def configure_cuda():
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no scientific CPU fallback')
    if platform.python_version()!='3.13.15' or torch.__version__!='2.11.0+cu130' or np.__version__!='2.1.3':
        raise RuntimeError('Pinned scientific Python/Torch/NumPy do not match')
    import pandas as pd
    if pd.__version__!='2.2.3' or torch.backends.cudnn.version()!=91900: raise RuntimeError('Pinned pandas/cuDNN do not match')
    props=torch.cuda.get_device_properties(0)
    if 'A100' not in props.name: raise RuntimeError('This run specification requires A100. Do not mix accelerator families.')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8': raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG before importing Torch')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True; torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.set_float32_matmul_precision('highest')
    return {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
            'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':props.name,
            'capability':list(torch.cuda.get_device_capability()),'gpu_bytes':props.total_memory,
            'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
            'libc':list(platform.libc_ver()),'torch_build':torch.__config__.show(),
            'packages':{d.metadata['Name'].lower():d.version for d in importlib.metadata.distributions()},
            'deterministic':True,'tf32':False,'amp':False,'compile':False,'threads':1}

class DataGuard:
    def __init__(self,root,out): self.root=Path(root).resolve();self.out=Path(out);self.phase='DEVELOPMENT';self.accesses=[]
    def hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)): return
        p=Path(os.fsdecode(args[0])).resolve(); n=p.name.lower()
        if 'labeled_anomal' in n or 'labelled_anomal' in n or 'test' in [x.lower() for x in p.parts]:
            raise PermissionError('TEST/label access prohibited in EXP012')
        if p.suffix=='.npy':
            allowed=p.parent==self.root/'development' or (self.phase=='CALIBRATION' and p.parent==self.root/'calibration')
            if not allowed: raise PermissionError('Array outside phase-specific allowlist')
            record={'path':str(p.relative_to(self.root)),'phase':self.phase}
            self.accesses.append(record)
            with open(self.out/'DATA_ACCESS_EVENTS.jsonl','a') as f:
                f.write(json.dumps(record,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    def install(self):
        import sys
        sys.addaudithook(self.hook)
    def enable_calibration(self):
        if not (self.out/'MODEL_FREEZE.json').exists(): raise RuntimeError('Model freeze required before calibration')
        self.phase='CALIBRATION'
