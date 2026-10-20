"""Fictional weights/data for recovery/E2E tests. No fitting or NASA evaluation."""
import io,json,csv,shutil,zipfile
from pathlib import Path
import numpy as np
import torch
from integrity import ROOT,read,write,sha,files,seal,blob,npz_bytes,state_hash
from checkpoint import save_torch
from inference import transform,gather,ewma
from vendor.exp013.architectures import CommandReferenceGCN
from vendor.frozen import metrics
from report_common import write_csv,mission_tables

def inventory(root):
    present={n:h for n,h in files(root).items() if n.startswith(('inputs/','evidence/'))}
    write(root/'INPUT_INVENTORY.json',{'version':'CPU_FICTIONAL_FIXTURE','present':present,'missing_originals':[]})

def fake_history_row(ch,mission,seed,arm,events,n,variant=0):
    ix=np.arange(250,n,dtype=np.int64);f=np.zeros(len(ix),bool)
    if variant%3!=2:f[(ix>=365)&(ix<380)]=True
    if variant%3==1:f[(ix>=310)&(ix<315)]=True
    issue=np.where(f,ix,-1);raw=f.astype(float)
    r,er,out=metrics.channel(events,ix,f,issue,raw,raw)
    meta={'job_id':f'{arm}__{ch}__{seed}','channel':ch,'mission':mission,'seed':seed,'arm':arm,
          'panel':'common_detector','detector':'hundman_adapt_v1','source_experiment':'CPU_FICTIONAL_HISTORY'}
    return {**meta,**r,'_delays':[v['delay'] for v in er if v['delay'] is not None]},[{**meta,'event_number':i,**v} for i,v in enumerate(er)]

def make_fixture(base,n=600):
    base=Path(base);root=base/'fixture_inputs';source=base/'fictional_EXP013_FINAL_RESULTS.zip'
    root.mkdir(parents=True);(root/'inputs').mkdir();(root/'evidence').mkdir()
    cfg=read(ROOT/'config.json');cfg['fixture']=True;cfg['bootstrap_repetitions']=100
    write(root/'config.json',cfg)
    for name in ['EXP012_CONFIG.json']:shutil.copy2(ROOT/'inputs'/name,root/'inputs'/name)
    rows={};meta={};native_rows=[];native_events=[];common=[];common_events=[];originals={}
    fake_events=[(360,400),(500,525)]
    for mission,ch,nf in [('MSL','C-1',55),('SMAP','A-1',25)]:
        t=np.arange(n);a=np.zeros((n,nf),np.float64)
        a[:,0]=.35+.07*np.sin(t/11);a[375:395,0]+=.5
        for j in range(1,nf):a[:,j]=(t//(7+j)%2).astype(float)
        f=io.BytesIO();np.save(f,a,allow_pickle=False);b=f.getvalue();originals[ch]=b
        rows[ch]={'mission':mission,'rows':n,'columns':nf,'sha256':__import__('hashlib').sha256(b).hexdigest()}
        meta[ch]={'mission':mission,'rows':n,'columns':nf,'mean':.35,'scale':.2,'fit_end':360,'stop_end':480,'cal_end':540,
                  'FIT_varying_command_slots':list(range(1,nf))}
        for arm in ['gcn','lstm','gdn']:
            for seed in ([17] if arm=='gdn' else cfg['seeds']):
                r,er=fake_history_row(ch,mission,seed,arm,fake_events,n,variant=seed+(arm=='lstm'))
                common.append(r);common_events+=er
    with zipfile.ZipFile(root/'inputs/NASA_arrays_only.zip','w',zipfile.ZIP_DEFLATED) as z:
        for ch,b in originals.items():z.writestr('data/data/test/'+ch+'.npy',b)
    write(root/'inputs/TEST_ROSTER.json',rows);(root/'inputs/freeze').mkdir()
    write(root/'inputs/freeze/DATA_FREEZE.json',{'version':'CPU_FICTIONAL_DATA','channels':meta})
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=['chan_id','spacecraft','anomaly_sequences','num_values']);writer.writeheader()
    for ch,r in rows.items():writer.writerow({'chan_id':ch,'spacecraft':r['mission'],'anomaly_sequences':str([(a,b-1) for a,b in fake_events]),'num_values':n})
    (root/'inputs/labeled_anomalies.csv').write_text(out.getvalue())
    selected={};source_tree=base/'fictional_source_tree';source_tree.mkdir()
    for mission in cfg['missions']:
        channels=sorted(ch for ch,r in rows.items() if r['mission']==mission);nf=rows[channels[0]]['columns']
        for role in cfg['roles']:
            for seed in cfg['seeds']:
                job=f'{mission}__{role}__{seed}';d=source_tree/'models'/job;d.mkdir(parents=True)
                torch.manual_seed(seed);model=CommandReferenceGCN(nf,len(channels),role,h=32).eval()
                modelcfg={'lookback':64,'batch_size':256,'loss':'fictional initial weights, no optimizer or fitting'}
                identity={'mission':mission,'role':role,'seed':seed,'configuration':modelcfg,'scaler':None,'channels':channels,'n_features':nf,'run':'CPU_FICTIONAL_NO_TRAINING'}
                ck={'identity':identity,'model':model.state_dict(),'channels':channels,'n_features':nf,'configuration':modelcfg,'scaler':None}
                save_torch(d/'selected.pt',ck);write(d/'IDENTITY.json',identity)
                st={'status':'COMPLETE','checkpoint_sha256':sha(d/'selected.pt'),'state_sha256':state_hash(model),'configuration':modelcfg}
                write(d/'COMPLETE.json',st)
                th={'mission':mission,'role':role,'seed':seed,'model_state_sha256':st['state_sha256'],'channels':{}}
                for ch in channels:th['channels'][ch]={'threshold':0.,'span':8,'native_stats':None}
                write(source_tree/'calibration'/job/'thresholds.json',th)
                selected[job]=(model,ck,st,th)
    study_files=files(source_tree);write(root/'inputs/freeze/STUDY_FREEZE.json',{'files':study_files,'NASA_labels_opened':False,'version':'CPU_FICTIONAL_STUDY'})
    study_hash=sha(root/'inputs/freeze/STUDY_FREEZE.json')
    for job,(model,ck,st,th) in selected.items():
        mission,role,s=job.split('__');seed=int(s)
        for e,ch in enumerate(ck['channels']):
            a=np.load(io.BytesIO(originals[ch]));z=transform(a,meta[ch]);logits=[]
            # Independent native-export loop, preserving the original saved representation.
            with torch.no_grad():
                for lo in range(64,n,256):
                    hi=min(n,lo+256);x=np.stack([z[t-64:t+1] for t in range(lo,hi)]);x[:,-1,1:]=0
                    entities=torch.full((len(x),),e,dtype=torch.long);_,l=model(torch.from_numpy(x),entities);logits.append(l.numpy())
            v=np.concatenate(logits).astype(np.float64);smooth=ewma(v,8);ix=np.arange(250,n,dtype=np.int64)
            native=v[186:];smooth=smooth[186:];flags=smooth>0
            uid={'study':study_hash,'model_state_sha256':st['state_sha256'],'TEST_array_sha256':rows[ch]['sha256'],
                 'channel':ch,'mission':mission,'role':role,'seed':seed}
            jid=job+'__'+ch;u=source_tree/'scores/units'/jid;u.mkdir(parents=True)
            (u/'scores.npz').write_bytes(npz_bytes(target_index=ix,native_score=native,detector_score=smooth,flags=flags,issue_index=np.where(flags,ix,-1)))
            write(u/'record.json',{'job_id':jid,'identity':uid,'status':'FINITE_SCORES','threshold':0.,'detector_span':8})
            seal(u,'COMMIT.json',{'identity':uid})
            r,er,out=metrics.channel(fake_events,ix,flags,np.where(flags,ix,-1),native,smooth)
            common_meta={'job_id':jid,'channel':ch,'mission':mission,'role':role,'arm':role,'seed':seed,'panel':'original_pipeline','detector':'CPU_FICTIONAL_NATIVE','source_experiment':'CPU_FICTIONAL_HISTORY'}
            native_rows.append({**common_meta,**r,'_delays':[v['delay'] for v in er if v['delay'] is not None]})
            native_events.extend({**common_meta,'event_number':i,**v} for i,v in enumerate(er))
    seal(source_tree/'scores','SCORE_SEAL.json',{'study_freeze_sha256':study_hash,'NASA_labels_opened':False})
    seal_hash=sha(source_tree/'scores/SCORE_SEAL.json');write(root/'inputs/freeze/FINAL_AUDIT.json',{'study_freeze_sha256':study_hash,'score_seal_sha256':seal_hash})
    contract=read(ROOT/'INPUT_CONTRACT.json');contract['known_required_files']={**study_files,'scores/SCORE_SEAL.json':seal_hash};contract['study_freeze_sha256']=study_hash
    write(root/'INPUT_CONTRACT.json',contract)
    for name,records,ers in [('exp014_reports',common,common_events),('exp013_reports',native_rows,native_events),('exp012_reports',[r for r in common if r['arm']!='gdn'],[r for r in common_events if r['arm']!='gdn'])]:
        d=root/'inputs'/name;d.mkdir();per,means=mission_tables(records)
        if name=='exp013_reports':
            for r in means:r['role']=r['arm']
        write_csv(d/('common_detector_channels.csv' if name=='exp014_reports' else 'channel_metrics.csv'),records)
        write_csv(d/('common_detector_event_notifications.csv' if name=='exp014_reports' else 'event_notifications.csv'),ers)
        write_csv(d/'mission_metrics.csv',means);seal(d,'LABEL_EVALUATION_COMPLETE.json' if name=='exp012_reports' else 'COMPLETE.json')
    write(root/'evidence/EXP014_FINAL_AUDIT.json',{'new_score_streams':len(rows),'fixture':True})
    with zipfile.ZipFile(source,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(source_tree.rglob('*')):
            if p.is_file():archive.write(p,p.relative_to(source_tree).as_posix())
    inventory(root);return root,source
