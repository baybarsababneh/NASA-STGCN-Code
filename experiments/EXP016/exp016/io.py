import hashlib, io, json, os, shutil, tempfile, time, zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def save_json(path,value):atomic(path,json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode())
def read_json(path):return json.loads(Path(path).read_text())
def freeze(path,obj):
    path=Path(path)
    if path.exists():
        if read_json(path)!=obj:raise RuntimeError(f'Immutable identity changed: {path}. Use original package/data/environment to resume.')
    else:save_json(path,obj)
def save_npz(path,**arrays):
    b=io.BytesIO();np.savez_compressed(b,**arrays);atomic(path,b.getvalue())
def safe_extract(archive,dest):
    dest=Path(dest).resolve();dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for i in z.infolist():
            p=(dest/i.filename).resolve()
            if not p.is_relative_to(dest) or (i.external_attr>>16)&0o170000==0o120000:
                raise ValueError('Unsafe ZIP member: '+i.filename)
        z.extractall(dest)
def package_identity():
    m=read_json(ROOT/'PACKAGE_MANIFEST.json')
    for name,h in m.items():
        if sha(ROOT/name)!=h:raise RuntimeError('Package integrity failure: '+name)
    return digest(m)
def commit(folder,names,identity):
    folder=Path(folder);save_json(folder/'COMMIT.json',{'identity':identity,'files':{n:sha(folder/n) for n in names}})
def committed(folder,identity=None):
    folder=Path(folder)
    if not (folder/'COMMIT.json').exists():return False
    c=read_json(folder/'COMMIT.json')
    if identity is not None and c['identity']!=identity:raise RuntimeError('Commit identity mismatch: '+str(folder))
    for n,h in c['files'].items():
        if not (folder/n).exists() or sha(folder/n)!=h:raise RuntimeError('Corrupted committed file: '+str(folder/n))
    return True
class Mirror:
    """Local execution, durable copy at stage boundaries and at most five minutes apart."""
    def __init__(self,local,durable=None):
        self.local=Path(local);self.durable=Path(durable) if durable else None;self.last=0.;self.stamps={}
        if self.durable and self.local.resolve()==self.durable.resolve():raise ValueError('Local work and durable mirror must differ')
    def restore(self):
        if not self.durable or not self.durable.exists():return
        for p in sorted(self.durable.rglob('*')):
            if p.is_file() and not p.name.startswith('.'):
                q=self.local/p.relative_to(self.durable)
                if not q.exists():atomic(q,p.read_bytes())
    def sync(self,force=False):
        if not self.durable or (not force and time.monotonic()-self.last<300):return
        files=[p for p in self.local.rglob('*') if p.is_file() and not p.name.startswith('.') and p.suffix!='.zip']
        # Commit receipts are copied after their payloads.
        files.sort(key=lambda p:(p.name=='COMMIT.json',str(p)))
        for p in files:
            rel=p.relative_to(self.local);st=(p.stat().st_size,p.stat().st_mtime_ns)
            if self.stamps.get(str(rel))==st:continue
            atomic(self.durable/rel,p.read_bytes());self.stamps[str(rel)]=st
        self.last=time.monotonic()
def export_run(out):
    out=Path(out);files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and not p.name.startswith('.') and p.name!='RESULT_MANIFEST.json' and p.suffix!='.zip'}
    save_json(out/'RESULT_MANIFEST.json',files)
    path=out.parent/(out.name+'_results.zip');tmp=path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name in [*files,'RESULT_MANIFEST.json']:z.write(out/name,arcname=out.name+'/'+name)
    os.replace(tmp,path)
    with zipfile.ZipFile(path) as z:
        if z.testzip():raise RuntimeError('Final archive CRC failure')
    return path
