import hashlib,json,os,uuid,zipfile,csv,io
from pathlib import Path
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def normal(v):
 if isinstance(v,np.ndarray):return normal(v.tolist())
 if isinstance(v,np.generic):return normal(v.item())
 if isinstance(v,Path):return str(v)
 if isinstance(v,dict):return {str(k):normal(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [normal(x) for x in v]
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def digest(v):return hashlib.sha256(json.dumps(normal(v),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic(path,blob):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
 with open(tmp,'wb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def write(path,v):atomic(path,json.dumps(normal(v),indent=2,sort_keys=True,allow_nan=False).encode())
def freeze(path,v):
 path=Path(path)
 if path.exists():
  if json.loads(path.read_text())!=normal(v):raise RuntimeError('Frozen identity differs: '+str(path))
 else:write(path,v)
def table(path,rows,fields=None):
 fields=fields or list(dict.fromkeys(k for r in rows for k in r));f=io.StringIO();w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(normal(rows));atomic(path,f.getvalue().encode())
def npz_bytes(v):
 f=io.BytesIO();np.savez_compressed(f,**v);return f.getvalue()
def loadnpz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def verify_manifest(root,name):
 root=Path(root);m=json.loads((root/name).read_text());files=m['files'];actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!=name}
 if actual!=set(files):raise RuntimeError('Roster differs: '+str(root))
 for n,h in files.items():
  if sha(root/n)!=h:raise RuntimeError('Changed bytes: '+n)
 return m
def safe_extract(path,dst):
 dst=Path(dst);dst.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(path) as z:
  if len(z.namelist())!=len(set(z.namelist())):raise RuntimeError('Duplicate ZIP paths')
  for item in z.infolist():
   p=(dst/item.filename).resolve()
   if dst.resolve() not in p.parents or (item.external_attr>>16)&0o170000==0o120000:raise RuntimeError('Unsafe ZIP path')
   if item.is_dir():p.mkdir(parents=True,exist_ok=True)
   else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(item.filename))
class Lease:
 def __init__(self,root):self.p=Path(root)/'ACTIVE_RUN.json';self.token=uuid.uuid4().hex
 def __enter__(self):
  self.p.parent.mkdir(parents=True,exist_ok=True)
  with self.p.open('x') as f:json.dump({'token':self.token,'pid':os.getpid()},f)
  return self
 def __exit__(self,*a):
  if self.p.exists() and json.loads(self.p.read_text()).get('token')==self.token:self.p.unlink()
def commit(root,jid,identity,files):
 b=Path(root)/'units'/jid;b.mkdir(parents=True,exist_ok=True)
 for n,blob in files.items():
  if Path(n).name!=n:raise ValueError(n)
  atomic(b/n,blob)
 write(b/'COMMIT.json',{'identity':identity,'files':{n:sha(b/n) for n in files}})
 return b
def reuse(root,jid,identity):
 b=Path(root)/'units'/jid
 if not (b/'COMMIT.json').exists():return None
 m=json.loads((b/'COMMIT.json').read_text())
 if m['identity']!=identity or {p.name for p in b.iterdir() if p.is_file() and '.tmp-' not in p.name}!={*m['files'],'COMMIT.json'}:raise RuntimeError('Changed commit: '+jid)
 for n,h in m['files'].items():
  if sha(b/n)!=h:raise RuntimeError('Corrupt commit: '+jid+'/'+n)
 return b
