"""E02: local archives, immutable Drive blobs, two verified commit receipts.

Drive is ONLY a backup destination. A failed copy never publishes a receipt.
Checkpoint binaries are copied only when changed; other state is one ZIP.
"""
import os, shutil, time, uuid, zipfile
from pathlib import Path
from core import sha, read, write, blob, atomic, checked, safe_extract

VERSION = 'EXP013_COMMAND_REFERENCE_V1_E03_CONV'

class Backup:
    def __init__(self, work, destination, cache, identity):
        self.work, self.destination, self.cache = map(Path, (work, destination, cache))
        self.identity = identity
        w,d,c=map(lambda p:p.resolve(),(self.work,self.destination,self.cache))
        if w==d or d in w.parents or w in d.parents or w==c or w in c.parents:
            raise RuntimeError('Live work, backup destination and archive cache must be separate directories')
        for p in [self.destination/'objects', self.destination/'commits', self.cache]:
            p.mkdir(parents=True, exist_ok=True)
        self.hash_cache = {}
        self.verified_objects = {}

    def file_hash(self, p):
        st=p.stat(); key=(st.st_ino, st.st_mtime_ns, st.st_size)
        if self.hash_cache.get(p, (None,))[0] != key:
            self.hash_cache[p] = (key, sha(p))
        return self.hash_cache[p][1]

    def copy_object(self, local):
        h=self.file_hash(local); dst=self.destination/'objects'/h
        if dst.exists():
            st=dst.stat(); key=(st.st_ino,st.st_mtime_ns,st.st_size)
            if self.verified_objects.get(h)!=key:
                checked(dst,h)  # Read back through FUSE; never trust a name alone.
                self.verified_objects[h]=key
            return h
        tmp=dst.with_name(h+'.partial-'+uuid.uuid4().hex)
        try:
            with local.open('rb') as source, tmp.open('wb') as target:
                shutil.copyfileobj(source,target,8*2**20)
                target.flush(); os.fsync(target.fileno())
            checked(tmp,h); os.replace(tmp,dst); checked(dst,h)
            st=dst.stat(); self.verified_objects[h]=(st.st_ino,st.st_mtime_ns,st.st_size)
        finally:
            tmp.unlink(missing_ok=True)
        return h

    def save(self, reason):
        start=time.monotonic()
        files={p.relative_to(self.work).as_posix():self.file_hash(p)
               for p in self.work.rglob('*') if p.is_file()
               and '__pycache__' not in p.parts and '.tmp-' not in p.name
               and '.partial-' not in p.name and not p.name.endswith('.lock')}
        binaries={n:h for n,h in files.items() if n.endswith('.pt')}
        archive=self.cache/'state.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for n in sorted(set(files)-set(binaries)): z.write(self.work/n,n)
        objects={'state_archive':self.copy_object(archive),'checkpoints':{}}
        for n,h in binaries.items():
            if self.copy_object(self.work/n)!=h: raise RuntimeError('State changed during backup')
            objects['checkpoints'][n]=h
        receipt={'version':VERSION,'identity':self.identity,'reason':reason,
                 'UTC':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                 'files':files,'objects':objects}
        # Nanosecond ordering is essential: several fast commits may occur in
        # one second, and random UUID order must never discard the newest state.
        existing=[p.name.split('_')[0] for p in (self.destination/'commits').iterdir()
                  if p.is_dir() and not p.name.endswith('.partial')]
        serial=max(time.time_ns(),max([int(v) for v in existing if v.isdigit()]+[0])+1)
        generation=f'{serial:020d}_'+uuid.uuid4().hex
        pending=self.destination/'commits'/(generation+'.partial')
        pending.mkdir()
        write(pending/'manifest.json',receipt)
        checked(pending/'manifest.json',__import__('hashlib').sha256(blob(receipt)).hexdigest())
        write(pending/'COMMITTED.json',{'manifest_sha256':sha(pending/'manifest.json')})
        checked(pending/'manifest.json',read(pending/'COMMITTED.json')['manifest_sha256'])
        os.replace(pending,self.destination/'commits'/generation)
        self.verify_receipt(self.destination/'commits'/generation,verify_objects=False)
        # Retain TWO committed generations. Objects referenced by either survive.
        commits=sorted(p for p in (self.destination/'commits').iterdir()
                       if p.is_dir() and not p.name.endswith('.partial'))
        for old in commits[:-2]: shutil.rmtree(old)
        keep=set()
        for p in commits[-2:]:
            r=self.verify_receipt(p,verify_objects=False)
            keep.add(r['objects']['state_archive']); keep.update(r['objects']['checkpoints'].values())
        for p in (self.destination/'objects').iterdir():
            if p.name not in keep and '.partial-' not in p.name: p.unlink()
        print('BACKUP VERIFIED',generation,'seconds',round(time.monotonic()-start,2),reason,flush=True)
        return receipt

    def verify_receipt(self, directory, verify_objects=True):
        checked(directory/'manifest.json',read(directory/'COMMITTED.json')['manifest_sha256'])
        r=read(directory/'manifest.json')
        if r['version']!=VERSION or r['identity']!=self.identity:
            raise RuntimeError('Backup source identity differs')
        for n,h in r['files'].items():
            if Path(n).is_absolute() or '..' in Path(n).parts or len(h)!=64:
                raise RuntimeError('Unsafe recovery path/hash')
        objects=r['objects']; expected={n:h for n,h in r['files'].items() if n.endswith('.pt')}
        if objects['checkpoints']!=expected: raise RuntimeError('Checkpoint receipt roster differs')
        if verify_objects:
            for h in {objects['state_archive'],*objects['checkpoints'].values()}:
                checked(self.destination/'objects'/h,h)
        return r

    def restore(self, environment):
        if self.work.exists() and any(self.work.iterdir()):
            raise RuntimeError('Restore requires an empty local work directory')
        commits=sorted((p for p in (self.destination/'commits').iterdir()
                        if p.is_dir() and not p.name.endswith('.partial')),reverse=True)
        if not commits:
            if any((self.destination/'commits').iterdir()) or any((self.destination/'objects').iterdir()):
                raise RuntimeError('Backup objects exist without a valid commit; inspect before creating a new run')
            return False
        errors=[]
        for directory in commits:
            temp=self.cache/('restore-'+uuid.uuid4().hex)
            try:
                r=self.verify_receipt(directory)
                safe_extract(self.destination/'objects'/r['objects']['state_archive'],temp)
                for n,h in r['objects']['checkpoints'].items():
                    p=temp/n; p.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(self.destination/'objects'/h,p)
                actual={p.relative_to(temp).as_posix() for p in temp.rglob('*') if p.is_file()}
                if actual!=set(r['files']): raise RuntimeError('Recovered file roster differs')
                for n,h in r['files'].items(): checked(temp/n,h)
                if read(temp/'RUN_FREEZE.json')['environment']!=environment:
                    # Do not fall back to an older runtime or silently start new fits.
                    raise EnvironmentMismatch('GPU/software differs from the scientific run; original state preserved')
                self.work.parent.mkdir(parents=True,exist_ok=True)
                if self.work.exists(): self.work.rmdir()
                os.replace(temp,self.work)
                print('RESTORED VERIFIED',directory.name,flush=True); return True
            except EnvironmentMismatch: raise
            except Exception as exc:
                errors.append(str(exc)); print('REJECTED BACKUP',directory.name,str(exc),flush=True)
            finally:
                if temp.exists(): shutil.rmtree(temp)
        raise RuntimeError('No valid recovery generation: '+'; '.join(errors))

class EnvironmentMismatch(RuntimeError): pass
