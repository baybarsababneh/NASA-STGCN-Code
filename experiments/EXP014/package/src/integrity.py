"""Small file/commit helpers; no fitting or detector selection."""
import os, json, hashlib, uuid, zipfile, shutil, io, random, sys, platform, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'EXP014_LITERATURE_COMPARISON_V1'

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def norm(v):
    if isinstance(v, dict): return {str(k):norm(x) for k,x in v.items()}
    if isinstance(v, (tuple,list)): return [norm(x) for x in v]
    if isinstance(v, np.ndarray): return norm(v.tolist())
    if isinstance(v, np.generic): return norm(v.item())
    if isinstance(v, Path): return str(v)
    if isinstance(v, float) and not np.isfinite(v): return None
    return v

def blob(v): return json.dumps(norm(v), sort_keys=True, indent=2, allow_nan=False).encode()
def digest(v): return hashlib.sha256(blob(v)).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def atomic(p,b):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
    try:
        with t.open('wb') as f: f.write(b); f.flush(); os.fsync(f.fileno())
        os.replace(t,p)
    finally: t.unlink(missing_ok=True)
def write(p,v): atomic(p,blob(v))
def freeze(p,v):
    if Path(p).exists():
        if read(p)!=norm(v): raise RuntimeError('Frozen identity changed: '+str(p))
    else: write(p,v)
def checked(p,h):
    if not Path(p).is_file(): raise FileNotFoundError(str(p))
    if sha(p)!=h: raise RuntimeError('SHA256 differs: '+str(p))
def safe_rel(n):
    q=Path(n)
    if q.is_absolute() or '..' in q.parts or '\\' in n or not n: raise RuntimeError('Unsafe relative path: '+n)
    return q
def safe_extract(p,dst):
    dst=Path(dst); dst.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(p) as z:
        if len(z.namelist())!=len(set(z.namelist())): raise RuntimeError('Duplicate ZIP paths')
        for a in z.infolist():
            rel=safe_rel(a.filename)
            if (a.external_attr>>16)&0o170000==0o120000: raise RuntimeError('ZIP symlink prohibited')
            if not a.is_dir():
                q=dst/rel; q.parent.mkdir(parents=True,exist_ok=True)
                with z.open(a) as f: atomic(q,f.read())
def files(root, exclude=()):
    root=Path(root)
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.relative_to(root).as_posix() not in exclude
            and '__pycache__' not in p.relative_to(root).parts and not any('.tmp-' in x or '.partial-' in x for x in p.relative_to(root).parts) and not p.name.endswith('.lock')}
def manifest(root,name='MANIFEST.json'):
    root=Path(root);m=read(root/name)
    if files(root,(name,))!=m['files']: raise RuntimeError('Manifest files/hashes differ: '+str(root))
    return m
def seal(root,name,extra=None):
    freeze(Path(root)/name,{'files':files(root,(name,)),**(extra or {})})
    return sha(Path(root)/name)
def arrays(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def npz_bytes(**a):
    f=io.BytesIO(); np.savez_compressed(f,**a);return f.getvalue()
def event(work,action,**kw):
    p=Path(work)/'PROGRESS.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        f.write(json.dumps(norm({'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'action':action,**kw}))+'\n'); f.flush();os.fsync(f.fileno())

def committed(work,jid,identity):
    p=Path(work)/'units'/safe_rel(jid)
    if not p.exists(): return None
    m=manifest(p,'COMMIT.json')
    if m['identity']!=norm(identity): raise RuntimeError('Stale scientific unit: '+jid)
    return p

def commit(work,jid,identity,payload):
    existing=committed(work,jid,identity)
    if existing is not None: return existing
    base=Path(work)/'units';base.mkdir(parents=True,exist_ok=True)
    tmp=base/('.tmp-'+jid+'-'+uuid.uuid4().hex);tmp.mkdir()
    try:
        for n,b in payload.items(): atomic(tmp/safe_rel(n),b)
        seal(tmp,'COMMIT.json',{'identity':identity})
        manifest(tmp,'COMMIT.json')
        dst=base/safe_rel(jid)
        os.replace(tmp,dst)
        # The directory rename publishes payload and receipt together.
        fd=os.open(base,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
        return dst
    finally:
        if tmp.exists(): shutil.rmtree(tmp)

def discard_uncommitted(work):
    for q in Path(work).glob('.tmp-reports-*'):
        event(work,'DISCARDED_UNCOMMITTED_REPORT_STAGING',directory=q.name)
        shutil.rmtree(q)
    p=Path(work)/'units'
    if not p.exists(): return
    for q in p.iterdir():
        if q.name.startswith('.tmp-'):
            event(work,'DISCARDED_UNCOMMITTED_STAGING',directory=q.name)
            shutil.rmtree(q)

def rng():
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def restore_rng(r):
    import torch
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    if r['cuda']:torch.cuda.set_rng_state_all(r['cuda'])
def rng_equal(a,b):
    import torch
    return a['python']==b['python'] and a['numpy'][0]==b['numpy'][0] and np.array_equal(a['numpy'][1],b['numpy'][1]) and a['numpy'][2:]==b['numpy'][2:] and torch.equal(a['torch'],b['torch']) and len(a['cuda'])==len(b['cuda']) and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda']))
def state_hash(model):
    h=hashlib.sha256()
    for k,t in sorted(model.state_dict().items()):
        a=t.detach().cpu().contiguous();h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()

class LabelGuard:
    def __init__(self,path): self.path=Path(path).resolve();self.allowed=False;self.active=True
    def hook(self,event,args):
        if self.active and event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)) and Path(os.fsdecode(args[0])).resolve()==self.path and not self.allowed:
            raise PermissionError('Annotation CSV may only be opened after the complete score seal is verified')
    def install(self): sys.addaudithook(self.hook)

class Lock:
    def __init__(self,p): self.p=Path(p)
    def __enter__(self):
        import fcntl
        self.p.parent.mkdir(parents=True,exist_ok=True);self.fd=self.p.open('a');fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB);return self
    def __exit__(self,*args): self.fd.close()

def configure(device,fixture=False):
    import torch, pandas, scipy, torch_geometric, sklearn, einops, matplotlib, importlib.metadata
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch_version=str(torch.__version__)
    if fixture and torch_version not in ['2.6.0+cpu','2.6.0+cu126']: raise RuntimeError('CPU fixtures require the frozen PyTorch 2.6 build')
    expected={'torch':torch_version if fixture else '2.6.0+cu126','numpy':'2.1.3','pandas':'2.2.3','scipy':'1.15.2','torch_geometric':'2.6.1','sklearn':'1.6.1','einops':'0.8.1','matplotlib':'3.10.1'}
    actual={'torch':torch_version,'numpy':np.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'torch_geometric':torch_geometric.__version__,'sklearn':sklearn.__version__,'einops':einops.__version__,'matplotlib':matplotlib.__version__}
    if actual!=expected: raise RuntimeError('Pinned numerical dependencies differ: '+str(actual))
    if sys.version_info[:2] not in [(3,12),(3,13)]: raise RuntimeError('Use supported Python 3.12 or 3.13; no automatic scientific migration')
    extensions={}
    for n in ['torch-scatter','torch-sparse','pyg-lib','torch-cluster','torch-spline-conv']:
        try: extensions[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError: extensions[n]=None
    if any(v is not None for v in extensions.values()): raise RuntimeError('Use the original native PyG path without optional compiled extensions')
    env={**actual,'python':platform.python_version(),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'device':device,'fixture':fixture,'AMP':False,'TF32':False,'deterministic':True,'threads':1,'optional_PyG_extensions':extensions,'GPU':None,'GPU_total_bytes':None}
    if fixture:
        if device!='cpu': raise RuntimeError('Fixtures are CPU-only and cannot pass production GPU requirements')
    else:
        if device!='cuda' or not torch.cuda.is_available(): raise RuntimeError('An A100 40 GB runtime is required for production inference')
        p=torch.cuda.get_device_properties(0)
        import subprocess
        driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()
        env.update(GPU=p.name,GPU_total_bytes=p.total_memory,GPU_driver=driver)
        if 'A100' not in p.name or not 35_000_000_000<=p.total_memory<=45_000_000_000: raise RuntimeError('Select A100 40 GB, rather than an 80 GB or different GPU')
        if torch.version.cuda!='12.6' or torch.backends.cudnn.version()!=90501: raise RuntimeError('Frozen CUDA 12.6 / CuDNN 9.5.1 required')
    return env
