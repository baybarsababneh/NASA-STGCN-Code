"""Incremental, immutable archive objects plus commit-last recovery receipts."""
import hashlib, json, os, shutil, tempfile, time, uuid, zipfile
from pathlib import Path
from drive_transport import AMENDMENT,file_sha

SCHEMA='EXP013_E01_RECOVERY_V1'

def encoded(v):return json.dumps(v,sort_keys=True,indent=2,allow_nan=False).encode()
def atomic(path,content):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with tmp.open('wb') as f:f.write(content);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def usable(r):return r.get('properties',{}).get('status')=='complete'
def safe_name(name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts or not name or '\\' in name:raise RuntimeError('Unsafe recovery member')

class IncompatibleRecovery(RuntimeError):pass

class Recovery:
    def __init__(self,transport,work,cache,identity,keep=2):
        if keep<2:raise ValueError('At least two verified recovery generations must be retained')
        self.transport=transport;self.work=Path(work);self.cache=Path(cache)
        self.identity=identity;self.keep=keep;self.cache.mkdir(parents=True,exist_ok=True)
        self.last_verified_time=None;self.last_receipt=None

    def _read_record(self,r):
        expected=r.get('properties',{}).get('sha256');size=int(r.get('size',-1))
        if not expected or not usable(r):raise RuntimeError('Recovery receipt is incomplete')
        self.transport.verify(r['id'],expected,size)
        p=self.cache/('receipt-'+r['id']+'.json');self.transport.download(r['id'],p)
        if p.stat().st_size!=size or file_sha(p)!=expected:raise RuntimeError('Recovery receipt SHA256 mismatch')
        v=json.loads(p.read_text())
        if v.get('schema')!=SCHEMA or v.get('identity')!=self.identity:
            raise IncompatibleRecovery('Recovery source/protocol/amendment identity differs')
        if set(v.get('files',{}))!={n for obj in v['objects'] for n in obj['members']}:
            raise RuntimeError('Recovery file roster mismatch')
        for obj in v['objects']:
            self.transport.verify(obj['remote_id'],obj['sha256'],obj['bytes'])
            if any(v['files'].get(n)!=h for n,h in obj['members'].items()):raise RuntimeError('Object/member manifest differs')
        return v

    def candidates(self):
        return sorted([r for r in self.transport.list('commit') if usable(r)],key=lambda r:r['createdTime'],reverse=True)

    def latest(self,environment=None):
        candidates=self.candidates();errors=[]
        for r in candidates:
            try:
                v=self._read_record(r)
                if environment is not None and v.get('environment') is not None and v['environment']!=environment:
                    raise IncompatibleRecovery('Recovery runtime differs from the exact frozen execution environment')
                return r,v
            except IncompatibleRecovery:raise
            except Exception as ex:
                print('REJECTED RECOVERY',r['name'],type(ex).__name__,str(ex),flush=True);errors.append(str(ex))
        if candidates:raise RuntimeError('No compatible, intact completed recovery generation: '+'; '.join(errors))
        # Pending archives/receipts are not successful backups. Announce these;
        # never claim they can recover an interrupted first-ever upload.
        pending=[r for r in self.transport.list() if not usable(r)]
        if pending:print('No committed recovery state. Ignoring',len(pending),'incomplete uploads.',flush=True)
        return None,None

    def _archive(self,members):
        token=uuid.uuid4().hex;path=self.cache/('archive-'+token+'.zip')
        # Model/optimizer snapshots are already binary; no repeated multi-GB
        # deflate buffering. Small metadata/scores are compressed together.
        large=any((self.work/n).stat().st_size>=32*2**20 for n in members)
        compression=zipfile.ZIP_STORED if large else zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(path,'w',compression=compression,compresslevel=None if large else 1,allowZip64=True) as z:
            for n in sorted(members):
                safe_name(n);z.write(self.work/n,n)
            z.writestr('ARCHIVE_MANIFEST.json',encoded({'schema':SCHEMA,'files':members}))
        self._verify_archive(path,members)
        return path

    def _verify_archive(self,path,members):
        with zipfile.ZipFile(path) as z:
            names=z.namelist()
            if len(names)!=len(set(names)) or set(names)!={*members,'ARCHIVE_MANIFEST.json'}:raise RuntimeError('Archive roster mismatch')
            if json.loads(z.read('ARCHIVE_MANIFEST.json'))!={'schema':SCHEMA,'files':members}:raise RuntimeError('Archive manifest mismatch')
            for n,h in members.items():
                safe_name(n);digest=hashlib.sha256()
                with z.open(n) as f:
                    for b in iter(lambda:f.read(2**20),b''):digest.update(b)
                if digest.hexdigest()!=h:raise RuntimeError('Archive member SHA256 mismatch: '+n)

    def backup(self,reason,progress=None):
        start=time.monotonic()
        records={p.relative_to(self.work).as_posix():file_sha(p) for p in self.work.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.tmp-' not in p.name and not p.name.endswith('.lock')}
        if not records:raise RuntimeError('No local experiment state to back up')
        large=[n for n in records if (self.work/n).stat().st_size>=32*2**20]
        groups=[{n:records[n]} for n in large]
        small={n:h for n,h in records.items() if n not in large}
        if small:groups.append(small)
        previous=self.latest()[1]
        old_objects=[] if previous is None else previous['objects']
        objects=[];uploaded_bytes=0
        for members in groups:
            match=next((v for v in old_objects if v['members']==members),None)
            if match:
                self.transport.verify(match['remote_id'],match['sha256'],match['bytes']);objects.append(match);continue
            path=self._archive(members);expected=file_sha(path);size=path.stat().st_size
            existing=next((r for r in self.transport.list('archive') if usable(r) and r.get('properties',{}).get('sha256')==expected),None)
            if existing:record=self.transport.verify(existing['id'],expected,size)
            else:
                record=self.transport.upload(path,'object-'+expected+'.zip','archive',progress)
                record=self.transport.finalize(record['id'],'object-'+expected+'.zip');uploaded_bytes+=size
            objects.append({'remote_id':record['id'],'sha256':expected,'bytes':size,'members':members})
            path.unlink(missing_ok=True)
        # Recheck the local roster and bytes at a synchronous safe boundary.
        current={p.relative_to(self.work).as_posix():file_sha(p) for p in self.work.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.tmp-' not in p.name and not p.name.endswith('.lock')}
        if records!=current:raise RuntimeError('Local state changed during snapshot creation; no commit issued')
        environment=None
        if 'RUN_FREEZE.json' in records:environment=json.loads((self.work/'RUN_FREEZE.json').read_text())['environment']
        receipt={'schema':SCHEMA,'identity':self.identity,'environment':environment,'reason':reason,'generation':uuid.uuid4().hex,'created_unix_ns':time.time_ns(),'files':records,'objects':objects,'uploaded_bytes':uploaded_bytes,'previous_commit_id':None if previous is None else self.latest()[0]['id']}
        local=self.cache/(receipt['generation']+'.complete.json');atomic(local,encoded(receipt))
        uploaded=self.transport.upload(local,'recovery-'+str(receipt['created_unix_ns'])+'-'+receipt['generation']+'.complete.json','commit',progress)
        final=self.transport.finalize(uploaded['id'],'recovery-'+str(receipt['created_unix_ns'])+'-'+receipt['generation']+'.complete.json')
        # Commit-last completion is durable only after both receipt and ALL
        # referenced objects are reverified. Older good generations still exist.
        self._read_record(final)
        self.last_verified_time=time.monotonic();self.last_receipt=final
        atomic(self.cache/'LAST_DURABLE.json',encoded({'receipt':final,'reason':reason,'backup_seconds':time.monotonic()-start,'uploaded_bytes':uploaded_bytes,'work_files':len(records),'interval_guarantee':False}))
        print('DURABLE BACKUP VERIFIED',reason,'receipt',final['name'],'uploaded_MiB',round(uploaded_bytes/2**20,2),'seconds',round(time.monotonic()-start,2),flush=True)
        self._retain()
        return final,receipt

    def restore(self,environment=None):
        if self.work.exists() and any(self.work.iterdir()):raise RuntimeError('Automatic restore requires an empty local work directory; existing state is never overwritten')
        remote,receipt=self.latest(environment)
        if receipt is None:return False
        parent=self.work.parent;parent.mkdir(parents=True,exist_ok=True)
        staging=parent/(self.work.name+'.restore-'+uuid.uuid4().hex);staging.mkdir()
        try:
            for obj in receipt['objects']:
                path=self.cache/('download-'+obj['sha256']+'.zip')
                self.transport.download(obj['remote_id'],path)
                if file_sha(path)!=obj['sha256'] or path.stat().st_size!=obj['bytes']:raise RuntimeError('Downloaded recovery archive checksum differs')
                self._verify_archive(path,obj['members'])
                with zipfile.ZipFile(path) as z:
                    for n in obj['members']:
                        safe_name(n);info=z.getinfo(n)
                        if (info.external_attr>>16)&0o170000==0o120000:raise RuntimeError('Recovery symlink rejected')
                        target=staging/n;target.parent.mkdir(parents=True,exist_ok=True)
                        with z.open(n) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,2**20);dst.flush();os.fsync(dst.fileno())
                path.unlink(missing_ok=True)
            actual={p.relative_to(staging).as_posix():file_sha(p) for p in staging.rglob('*') if p.is_file()}
            if actual!=receipt['files']:raise RuntimeError('Restored file manifest differs')
            if self.work.exists():self.work.rmdir()
            os.replace(staging,self.work)
            self.last_receipt=remote;self.last_verified_time=time.monotonic()
            print('RESTORED VERIFIED RECOVERY',remote['name'],'files',len(actual),flush=True)
            return True
        finally:shutil.rmtree(staging,ignore_errors=True)

    def _retain(self):
        candidates=self.candidates();good=[]
        for r in candidates:
            try:good.append((r,self._read_record(r)))
            except IncompatibleRecovery:raise
            except Exception:continue
            if len(good)>=self.keep:break
        if len(good)<self.keep:return
        kept={r['id'] for r,v in good};objects={o['remote_id'] for r,v in good for o in v['objects']}
        # Delete superseded generations only AFTER two complete, verified states.
        # Pending upload remnants are ignored, never used as recovery evidence.
        for r in candidates:
            if r['id'] not in kept:self.transport.delete(r['id'])
        for r in self.transport.list('archive'):
            if usable(r) and r['id'] not in objects:self.transport.delete(r['id'])

class LocalTransport:
    """Faithful local transfer fixture, including partial uploads and corruption."""
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.fail_after=None;self.fail_kind=None
    def metadata(self,i):return json.loads((self.root/(i+'.meta.json')).read_text())
    def list(self,kind=None):
        rows=[json.loads(p.read_text()) for p in self.root.glob('*.meta.json')]
        return [r for r in rows if kind is None or r['properties']['kind']==kind]
    def upload(self,path,name,kind,progress=None):
        identifier=uuid.uuid4().hex;dst=self.root/(identifier+'.bin');size=0
        meta={'id':identifier,'name':name+'.partial','size':'0','createdTime':str(time.time_ns()),'properties':{'amendment':AMENDMENT,'kind':kind,'status':'pending','sha256':file_sha(path)}}
        atomic(self.root/(identifier+'.meta.json'),encoded(meta))
        with Path(path).open('rb') as src,dst.open('wb') as out:
            for b in iter(lambda:src.read(4096),b''):
                out.write(b);size+=len(b)
                if self.fail_after is not None and size>=self.fail_after and (self.fail_kind is None or self.fail_kind==kind):raise OSError('Simulated interrupted upload')
        meta.update(size=str(size),sha256Checksum=file_sha(dst));atomic(self.root/(identifier+'.meta.json'),encoded(meta))
        return self.verify(identifier,file_sha(path),Path(path).stat().st_size)
    def verify(self,i,expected,size):
        r=self.metadata(i);p=self.root/(i+'.bin')
        if p.stat().st_size!=size or file_sha(p)!=expected:raise RuntimeError('Remote SHA256/size differs')
        return r
    def finalize(self,i,name):
        r=self.metadata(i);r['name']=name;r['properties']['status']='complete';atomic(self.root/(i+'.meta.json'),encoded(r));return r
    def download(self,i,path):Path(path).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(self.root/(i+'.bin'),path)
    def delete(self,i):(self.root/(i+'.bin')).unlink();(self.root/(i+'.meta.json')).unlink()
