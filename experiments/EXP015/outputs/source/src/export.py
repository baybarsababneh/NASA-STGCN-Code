"""Hash-verified immutable final exports, retained from EXP014."""
import os,shutil,zipfile
from pathlib import Path
from integrity import *
def snapshot(work,path):
    work=Path(work);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+__import__('uuid').uuid4().hex)
    try:
        records=files(work)
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for n in sorted(records):z.write(work/n,n)
            z.writestr('SNAPSHOT_MANIFEST.json',blob({'version':VERSION,'files':records}))
        with zipfile.ZipFile(tmp) as z:
            for n,h in records.items():
                if __import__('hashlib').sha256(z.read(n)).hexdigest()!=h:raise RuntimeError('Final ZIP verification failed')
        os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)
    return sha(path)

def publish_archive(local,dest):
    h=sha(local);dest=Path(dest)
    if dest.exists():
        checked(dest,h);return
    tmp=dest.with_name(dest.name+'.partial-'+__import__('uuid').uuid4().hex)
    try:
        dest.parent.mkdir(parents=True,exist_ok=True)
        with Path(local).open('rb') as src,tmp.open('wb') as dst:
            shutil.copyfileobj(src,dst,8*2**20);dst.flush();os.fsync(dst.fileno())
        checked(tmp,h);os.replace(tmp,dest);checked(dest,h)
    finally:tmp.unlink(missing_ok=True)

