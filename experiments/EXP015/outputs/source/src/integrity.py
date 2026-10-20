"""Small file/commit helpers; no fitting or detector selection."""
import os, json, hashlib, uuid, zipfile, shutil, io, random, sys, platform, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'EXP015_EXP013_GCN_HUNDMAN_SUPPLEMENT_V1'

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
    import torch, pandas, scipy, sklearn, matplotlib, subprocess
    os.environ['MPLBACKEND']='Agg'
    matplotlib.use('Agg',force=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    actual={'torch':str(torch.__version__),'numpy':np.__version__,'pandas':pandas.__version__,
            'scipy':scipy.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__}
    expected={'torch':'2.6.0+cpu' if fixture else '2.6.0+cu126','numpy':'2.1.3','pandas':'2.2.3',
              'scipy':'1.15.2','sklearn':'1.6.1','matplotlib':'3.10.1'}
    if fixture and actual['torch']=='2.6.0+cu126':expected['torch']=actual['torch']
    if actual!=expected:raise RuntimeError('Pinned numerical dependencies differ: '+str(actual))
    if sys.version_info[:2] not in [(3,12),(3,13)]:raise RuntimeError('Python 3.12 or 3.13 required')
    env={**actual,'python':platform.python_version(),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
         'device':device,'fixture':fixture,'AMP':False,'TF32':False,'deterministic':True,'threads':1,
         'matmul_precision':torch.get_float32_matmul_precision(),'MPLBACKEND':os.environ['MPLBACKEND'],
         'GPU':None,'GPU_total_bytes':None,'torch_build':torch.__config__.show()}
    if fixture:
        if device!='cpu':raise RuntimeError('Synthetic fixture cannot satisfy GPU production checks')
    else:
        if device!='cuda' or not torch.cuda.is_available():raise RuntimeError('A100 40 GB or 80 GB required')
        p=torch.cuda.get_device_properties(0)
        capacity='40GB' if 35_000_000_000<=p.total_memory<=45_000_000_000 else '80GB' if 75_000_000_000<=p.total_memory<=90_000_000_000 else None
        if 'A100' not in p.name or capacity is None:raise RuntimeError('Actual A100 capacity is outside the 40/80 GB ranges')
        if torch.version.cuda!='12.6' or torch.backends.cudnn.version()!=90501:raise RuntimeError('CUDA 12.6 and CuDNN 9.5.1 required')
        env.update(GPU=p.name,GPU_total_bytes=p.total_memory,GPU_capacity=capacity,
                   GPU_driver=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip())
    return env

