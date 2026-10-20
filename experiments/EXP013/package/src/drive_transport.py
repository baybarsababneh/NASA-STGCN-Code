"""Drive API transport: binary archives only, no Drive FUSE live I/O."""
import hashlib, json, os, tempfile, time, uuid
from pathlib import Path

AMENDMENT = 'EXP013_COMMAND_REFERENCE_V1_E01'
FOLDER_NAME = AMENDMENT + '_recovery'
FIELDS = 'id,name,size,sha256Checksum,md5Checksum,properties,parents,createdTime'

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

class DriveTransport:
    def __init__(self, service, folder_id): self.service,self.folder_id=service,folder_id

    @classmethod
    def connect(cls, folder_id=None):
        # Official Colab authentication provides Drive-enabled application
        # credentials. Authentication is initial setup, never per checkpoint.
        from google.colab import auth
        import google.auth, httplib2, google_auth_httplib2
        from googleapiclient.discovery import build
        auth.authenticate_user()
        credentials,_=google.auth.default()
        http=google_auth_httplib2.AuthorizedHttp(credentials,http=httplib2.Http(timeout=60))
        service=build('drive','v3',http=http,cache_discovery=False)
        service.about().get(fields='storageQuota').execute(num_retries=3)
        if folder_id:
            r=service.files().get(fileId=folder_id,fields='id,mimeType,trashed').execute(num_retries=3)
            if r.get('trashed') or r.get('mimeType')!='application/vnd.google-apps.folder':
                raise RuntimeError('The configured Drive backup folder is invalid')
        else:
            q="trashed=false and 'root' in parents and mimeType='application/vnd.google-apps.folder' and name='"+FOLDER_NAME+"'"
            folders=service.files().list(q=q,fields='files(id,name)',pageSize=100).execute(num_retries=3)['files']
            if len(folders)>1:raise RuntimeError('Duplicate EXP013 E01 backup folders; set DRIVE_FOLDER_ID to the intended folder ID')
            if folders:folder_id=folders[0]['id']
            else:folder_id=service.files().create(body={'name':FOLDER_NAME,'mimeType':'application/vnd.google-apps.folder'},fields='id').execute(num_retries=3)['id']
        print('Google Drive recovery connected:',FOLDER_NAME,'folder ID',folder_id,flush=True)
        return cls(service,folder_id)

    def list(self,kind=None):
        q="trashed=false and '"+self.folder_id+"' in parents and properties has {key='amendment' and value='"+AMENDMENT+"'}"
        if kind:q+=" and properties has {key='kind' and value='"+kind+"'}"
        items=[];token=None
        while True:
            r=self.service.files().list(q=q,fields='nextPageToken,files('+FIELDS+')',pageSize=1000,pageToken=token,orderBy='createdTime desc').execute(num_retries=3)
            items.extend(r.get('files',[]));token=r.get('nextPageToken')
            if not token:return items

    def metadata(self,identifier):
        return self.service.files().get(fileId=identifier,fields=FIELDS).execute(num_retries=3)

    def storage_quota(self):
        return self.service.about().get(fields='storageQuota').execute(num_retries=3).get('storageQuota',{})

    @classmethod
    def from_existing_authorization(cls,folder_id):
        # The notebook authorizes once. Workers reuse Colab's ADC credentials;
        # they never try to display a second authentication prompt.
        import google.auth,httplib2,google_auth_httplib2
        from googleapiclient.discovery import build
        credentials,_=google.auth.default()
        http=google_auth_httplib2.AuthorizedHttp(credentials,http=httplib2.Http(timeout=60))
        service=build('drive','v3',http=http,cache_discovery=False)
        folder=service.files().get(fileId=folder_id,fields='id,mimeType,trashed').execute(num_retries=3)
        if folder.get('trashed') or folder.get('mimeType')!='application/vnd.google-apps.folder':raise RuntimeError('Authorized recovery folder is unavailable')
        return cls(service,folder_id)

    def upload(self,path,name,kind,progress=None):
        from googleapiclient.http import MediaFileUpload
        path=Path(path);expected=file_sha(path)
        body={'name':name+'.partial-'+uuid.uuid4().hex,'parents':[self.folder_id],
              'properties':{'amendment':AMENDMENT,'kind':kind,'status':'pending','sha256':expected}}
        request=self.service.files().create(body=body,media_body=MediaFileUpload(str(path),mimetype='application/octet-stream',chunksize=8*2**20,resumable=True),fields=FIELDS)
        response=None;start=time.monotonic();last=0.
        while response is None:
            status,response=request.next_chunk(num_retries=3)
            if progress and status and time.monotonic()-last>=30:
                progress(status.resumable_progress,path.stat().st_size,time.monotonic()-start);last=time.monotonic()
        return self.verify(response['id'],expected,path.stat().st_size)

    def download(self,identifier,path):
        from googleapiclient.http import MediaIoBaseDownload
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+'.partial-'+uuid.uuid4().hex)
        try:
            with tmp.open('wb') as f:
                request=self.service.files().get_media(fileId=identifier)
                reader=MediaIoBaseDownload(f,request,chunksize=8*2**20);done=False
                while not done:_,done=reader.next_chunk(num_retries=3)
                f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:tmp.unlink(missing_ok=True)

    def verify(self,identifier,expected,size):
        r=self.metadata(identifier)
        if int(r.get('size',-1))!=size:raise RuntimeError('Drive byte count mismatch')
        server=r.get('sha256Checksum')
        if server is not None:
            if server!=expected:raise RuntimeError('Drive SHA256 mismatch')
        else:
            # Some Drive binary objects have no reported SHA256. Full API
            # readback is mandatory then; never substitute MD5-only acceptance.
            with tempfile.TemporaryDirectory(prefix='EXP013_DRIVE_VERIFY_') as td:
                p=Path(td)/'readback';self.download(identifier,p)
                if file_sha(p)!=expected:raise RuntimeError('Drive SHA256 readback mismatch')
        return r

    def finalize(self,identifier,name):
        r=self.metadata(identifier);props=r.get('properties',{})
        if props.get('amendment')!=AMENDMENT or self.folder_id not in r.get('parents',[]):raise RuntimeError('Refusing to mutate unrelated Drive data')
        props={**props,'status':'complete'}
        return self.service.files().update(fileId=identifier,body={'name':name,'properties':props},fields=FIELDS).execute(num_retries=3)

    def delete(self,identifier):
        r=self.metadata(identifier)
        if r.get('properties',{}).get('amendment')!=AMENDMENT or self.folder_id not in r.get('parents',[]):raise RuntimeError('Refusing retention cleanup outside this amendment folder')
        self.service.files().delete(fileId=identifier).execute(num_retries=3)

def cache_release(transport,path,expected):
    matches=[r for r in transport.list('release') if r.get('properties',{}).get('status')=='complete' and r.get('properties',{}).get('sha256')==expected]
    if matches:
        transport.verify(matches[0]['id'],expected,Path(path).stat().st_size)
        return matches[0]
    record=transport.upload(path,Path(path).name,'release')
    return transport.finalize(record['id'],Path(path).name)
