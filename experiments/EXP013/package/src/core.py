"""Local-only, transactional experiment state. No training code reads NASA labels."""
import os, json, hashlib, io, zipfile, uuid, random, platform, time
from pathlib import Path
import numpy as np
import torch

VERSION = 'EXP013_COMMAND_REFERENCE_V1'
ROOT = Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def norm(v):
    if isinstance(v,dict): return {str(k):norm(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)): return [norm(x) for x in v]
    if isinstance(v,np.ndarray): return norm(v.tolist())
    if isinstance(v,np.generic): return norm(v.item())
    if isinstance(v,Path): return str(v)
    if isinstance(v,float) and not np.isfinite(v): return None
    return v
def blob(v): return json.dumps(norm(v),sort_keys=True,indent=2,allow_nan=False).encode()
def digest(v): return hashlib.sha256(blob(v)).hexdigest()
def atomic(p,b):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
    with t.open('wb') as f: f.write(b);f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def write(p,v): atomic(p,blob(v))
def read(p): return json.loads(Path(p).read_text())
def freeze(p,v):
    if Path(p).exists():
        if read(p)!=norm(v): raise RuntimeError('Frozen identity changed: '+str(p))
    else: write(p,v)
def npz(p,**v):
    b=io.BytesIO();np.savez_compressed(b,**v);atomic(p,b.getvalue())
def arrays(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def checked(p,expected):
    if not Path(p).is_file(): raise FileNotFoundError(str(p))
    if sha(p)!=expected: raise RuntimeError('SHA256 differs: '+str(p))
def manifest(root,name='MANIFEST.json'):
    root=Path(root);m=read(root/name);actual={f.relative_to(root).as_posix() for f in root.rglob('*') if f.is_file() and f.name!=name and '__pycache__' not in f.parts and '.tmp-' not in f.name}
    if actual!=set(m['files']): raise RuntimeError('Manifest roster differs: '+str(root))
    for n,h in m['files'].items(): checked(root/n,h)
    return m
def seal(root,name,extra=None):
    root=Path(root);files={f.relative_to(root).as_posix():sha(f) for f in root.rglob('*') if f.is_file() and f.name!=name and '.tmp-' not in f.name and '__pycache__' not in f.parts}
    freeze(root/name,{'files':files,**(extra or {})});return sha(root/name)
def safe_extract(p,dst):
    dst=Path(dst);dst.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(p) as z:
        if len(z.namelist())!=len(set(z.namelist())): raise RuntimeError('Duplicate ZIP names')
        for a in z.infolist():
            q=(dst/a.filename).resolve()
            if dst.resolve() not in q.parents or (a.external_attr>>16)&0o170000==0o120000: raise RuntimeError('Unsafe ZIP member')
            if not a.is_dir(): q.parent.mkdir(parents=True,exist_ok=True);atomic(q,z.read(a))
def event(work,action,**kw):
    p=Path(work)/'ACCESS.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f: f.write(json.dumps(norm({'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'action':action,**kw}))+'\n')
def seeds(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def rng():
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':(n[0],n[1].tolist(),n[2],n[3],n[4]),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def restore_rng(r):
    random.setstate(r['python']);n=r['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]));torch.set_rng_state(r['torch'])
    if r['cuda']: torch.cuda.set_rng_state_all(r['cuda'])
def state_hash(model):
    h=hashlib.sha256()
    for k,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()
def save_torch(p,v):
    b=io.BytesIO();torch.save(v,b);payload=b.getvalue()
    # Hash and payload are one atomic file, so interruptions cannot split a checkpoint/hash pair.
    atomic(p,b'EXP013_CHECKPOINT_V1\n'+hashlib.sha256(payload).hexdigest().encode()+b'\n'+payload)
def load_torch(p):
    tag,h,payload=Path(p).read_bytes().split(b'\n',2)
    if tag!=b'EXP013_CHECKPOINT_V1' or h.decode()!=hashlib.sha256(payload).hexdigest():raise RuntimeError('Checkpoint payload corruption')
    return torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
def environment(device):
    import pandas,scipy,sklearn,einops,torch_geometric,importlib.metadata,subprocess
    optional={}
    for name in ['torch-scatter','torch-sparse','pyg-lib','torch-cluster','torch-spline-conv']:
        try:optional[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:optional[name]=None
    driver=None
    if device=='cuda':driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()
    return {'python':platform.python_version(),'torch':str(torch.__version__),'torch_build':torch.__config__.show(),'numpy':np.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'einops':einops.__version__,'torch_geometric':torch_geometric.__version__,'optional_PyG_extensions':optional,'device':device,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'GPU':torch.cuda.get_device_name(0) if device=='cuda' else None,'GPU_total_bytes':torch.cuda.get_device_properties(0).total_memory if device=='cuda' else None,'GPU_driver':driver,'deterministic':True,'AMP':False,'TF32':False,'threads':1}
def configure(device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('Select a GPU runtime. A100 40GB is supported; 80GB is not required.')
    return environment(device)
class Lock:
    def __init__(self,p): self.p=Path(p);self.fd=None
    def __enter__(self):
        import fcntl
        self.p.parent.mkdir(parents=True,exist_ok=True);self.fd=self.p.open('a');fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return self
    def __exit__(self,*a): self.fd.close()
