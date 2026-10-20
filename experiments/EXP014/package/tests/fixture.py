"""Synthetic input contract for the real runner. Never a NASA/GPU result."""
import sys, io, shutil, zipfile, hashlib
from pathlib import Path
CODE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(CODE/'src'),str(CODE)]
import numpy as np
import torch
from integrity import read,write,sha,blob,seal,files,state_hash,npz_bytes,atomic
from checkpoint import save_torch
from gdn_inference import factory,native,ewma
from vendor.frozen.hundman_detector import detect
from vendor.frozen import metrics
from report import write_csv,mission_tables

def create(base):
    base=Path(base);root=base/'input';source=base/'source'
    root.mkdir(parents=True);source.mkdir()
    for d in ['inputs/freeze','inputs/exp012_scores/units','inputs/exp012_reports','inputs/exp013_reports']:(root/d).mkdir(parents=True)
    cfg=read(CODE/'config.json');cfg.update(fixture=True,bootstrap_repetitions=40,backup_interval_seconds=.2)
    write(root/'config.json',cfg);write(root/'inputs/EXP012_CONFIG.json',{'detector':cfg['detector']})
    gen=np.random.default_rng(312);raw={};roster={};meta={};label_rows=[]
    for mission,nf,names in [('MSL',55,['F-MSL-0','F-MSL-1']),('SMAP',25,['F-SMAP-0','P-2'])]:
        for i,ch in enumerate(names):
            x=gen.normal(0,.1,(1040,nf)).astype(np.float64)
            x[:,0]=.2+np.sin(np.arange(len(x))/40)*.1
            x[730:755,0]+=1.5+i*.1
            f=io.BytesIO();np.save(f,x,allow_pickle=False);data=f.getvalue();raw[ch]=(x,data)
            roster[ch]={'mission':mission,'rows':len(x),'columns':nf,'sha256':hashlib.sha256(data).hexdigest()}
            meta[ch]={'mission':mission,'mean':.2,'scale':.4,'columns':nf}
            label_rows.append({'chan_id':ch,'spacecraft':mission,'anomaly_sequences':'[[730, 754]]','class':'synthetic','num_values':len(x)})
    label_rows.append({**label_rows[-1],'anomaly_sequences':'[[800, 810]]'})
    write_csv(root/'inputs/labeled_anomalies.csv',label_rows)
    write(root/'inputs/TEST_ROSTER.json',roster);write(root/'inputs/freeze/DATA_FREEZE.json',{'channels':meta})
    with zipfile.ZipFile(root/'inputs/NASA_arrays_only.zip','w',zipfile.ZIP_DEFLATED) as z:
        for ch,(x,data) in raw.items():z.writestr('data/data/test/'+ch+'.npy',data)
    known={};old_rows=[];old_events=[];native_rows=[];native_events=[]
    study={'fixture':True,'eligible_model_jobs':[['MSL','gdn',17],['SMAP','gdn',17]],'files':{}}
    # Synthetic untrained weights exercise native checkpoint loading and full
    # mission dimensions. They are never substituted into production inputs.
    for mission,nf in [('MSL',55),('SMAP',25)]:
        torch.manual_seed(17);m=factory(nf,'cpu').eval();channels=sorted(ch for ch,r in roster.items() if r['mission']==mission)
        scaler={'mean':([.05]+[0.]*(nf-1)),'scale':([1.3]+[1.]*(nf-1))}
        identity={'fixture':True,'mission':mission,'role':'gdn','seed':17,'channels':channels,'n_features':nf,'scaler':scaler}
        ck={'identity':identity,'model':m.state_dict(),'channels':channels,'n_features':nf,'scaler':scaler,'configuration':{'lookback':5,'batch_size':32}}
        directory=source/'models'/f'{mission}__gdn__17';directory.mkdir(parents=True)
        write(directory/'IDENTITY.json',identity);save_torch(directory/'selected.pt',ck)
        write(directory/'COMPLETE.json',{'status':'COMPLETE','state_sha256':state_hash(m),'checkpoint_sha256':sha(directory/'selected.pt')})
        th={'model_state_sha256':state_hash(m),'channels':{}}
        for ch in channels:
            th['channels'][ch]={'native_stats':{'median':[.05]*nf,'iqr':[.2]*nf},'span':8,'threshold':2.,'q':.995}
        write(source/'calibration'/f'{mission}__gdn__17/thresholds.json',th)
        for n in ['IDENTITY.json','selected.pt','COMPLETE.json']:
            rel=f'models/{mission}__gdn__17/{n}';known[rel]=sha(source/rel)
        rel=f'calibration/{mission}__gdn__17/thresholds.json';known[rel]=sha(source/rel)
        for ch in channels:
            x=raw[ch][0];a=x.astype(np.float32,copy=True);a[:,0]=(x[:,0].astype(np.float64)-meta[ch]['mean'])/meta[ch]['scale']
            z=np.asarray((a-np.array(scaler['mean']))/np.array(scaler['scale']),np.float32)
            err=[]
            with torch.no_grad():
                for lo in range(5,len(z),32):
                    hi=min(len(z),lo+32)
                    windows=np.ascontiguousarray(np.stack([z[t-5:t].T for t in range(lo,hi)]))
                    p=m(torch.from_numpy(windows),None).numpy();err.append(np.abs(p-z[lo:hi]))
            score_all=native(np.concatenate(err),th['channels'][ch]['native_stats']);sm_all=ewma(score_all,8)
            score=score_all[245:];sm=sm_all[245:];ix=np.arange(250,len(z),dtype=np.int64);flags=sm>2.
            jid=f'{mission}__gdn__17__{ch}';d=source/'scores/units'/jid;d.mkdir(parents=True)
            uid={'study':None,'TEST_array_sha256':roster[ch]['sha256'],'model_state_sha256':state_hash(m)}
            write(d/'record.json',{'job_id':jid,'identity':uid,'status':'FINITE_SCORES','threshold':2.,'detector_span':8})
            atomic(d/'scores.npz',npz_bytes(target_index=ix,native_score=score,detector_score=sm,flags=flags,issue_index=np.where(flags,ix,-1)))
            # Filled once study hashes are available below.
            for role,sc in [('gdn',sm),('catch',np.abs(x[250:,0].astype(np.float64)-x[249:-1,0]))]:
                f=flags if role=='gdn' else sc>.15
                r,er,_=metrics.channel([(730,755)],ix,f,np.where(f,ix,-1),sc,sc)
                identity_fields={'job_id':f'{mission}__{role}__17__{ch}','role':role,'mission':mission,'channel':ch,'seed':17}
                native_rows.append({**identity_fields,**r})
                native_events.extend({**identity_fields,**e} for e in er)
    study['files']=dict(known);write(root/'inputs/freeze/STUDY_FREEZE.json',study)
    for ch,r in roster.items():
        d=source/'scores/units'/f"{r['mission']}__gdn__17__{ch}";rec=read(d/'record.json');rec['identity']['study']=sha(root/'inputs/freeze/STUDY_FREEZE.json');write(d/'record.json',rec);seal(d,'COMMIT.json',{'identity':rec['identity']})
    seal(source/'scores','SCORE_SEAL.json',{'study_freeze_sha256':sha(root/'inputs/freeze/STUDY_FREEZE.json'),'NASA_labels_opened':False})
    known['scores/SCORE_SEAL.json']=sha(source/'scores/SCORE_SEAL.json')
    subset_files={};sub_roster=[]
    for ch,r in roster.items():
        x=raw[ch][0];ix=np.arange(250,len(x),dtype=np.int64);target=x[250:,0].astype(np.float64)
        for arm in ['gcn','lstm']:
            for seed in [17,29,43]:
                pred=x[249:-1,0].astype(np.float64)+(.002 if arm=='gcn' else .004)*seed/17
                det=detect(target,pred,ix,cfg['detector']);raw_score=np.abs(target-pred)
                j=f'{arm}__{ch}__{seed}';d=root/'inputs/exp012_scores/units'/j;d.mkdir()
                atomic(d/'scores.npz',npz_bytes(target_index=ix,target=target,prediction=pred,scores_raw=raw_score,scores_ewma=det['smoothed'],adaptive_flags=det['flags'],adaptive_issue_index=det['issue_index']))
                record={'job_id':j,'arm':arm,'channel':ch,'mission':r['mission'],'seed':seed,'original_TEST_rows':len(x),'TEST_array_sha256':r['sha256'],'target_start':250}
                write(d/'record.json',record);write(d/'adaptive_windows.json',det['windows']);seal(d,'COMMIT.json',{'identity':{'job_id':j}})
                for n,h in files(d).items():subset_files[f'units/{j}/{n}']=h
                sub_roster.append(j)
                met,er,_=metrics.channel([(730,755)],ix,det['flags'],det['issue_index'],raw_score,det['smoothed'])
                old_rows.append({**record,'detector':'hundman_adapt_v1',**met,'panel':'common_detector','source_experiment':'EXP012','_delays':[v['delay'] for v in er if v['delay'] is not None]})
                old_events.extend({**record,**e} for e in er)
    write(root/'inputs/exp012_scores/EVALUATION_RUN_FREEZE.json',{'fixture':True});write(root/'inputs/exp012_scores/TEST_INPUT_FREEZE.json',{'fixture':True})
    write(root/'inputs/exp012_scores/SCORE_EXPORT_FREEZE.json',{'files':subset_files,'roster':sub_roster,'run_freeze_sha256':sha(root/'inputs/exp012_scores/EVALUATION_RUN_FREEZE.json'),'TEST_input_freeze_sha256':sha(root/'inputs/exp012_scores/TEST_INPUT_FREEZE.json')})
    write(root/'inputs/exp012_scores/SUBSET.json',{'files':subset_files,'roster':sub_roster,'original_seal_sha256':sha(root/'inputs/exp012_scores/SCORE_EXPORT_FREEZE.json')})
    write_csv(root/'inputs/exp012_reports/channel_metrics.csv',old_rows);write_csv(root/'inputs/exp012_reports/event_notifications.csv',old_events)
    _,means=mission_tables(old_rows);write_csv(root/'inputs/exp012_reports/mission_metrics.csv',means);seal(root/'inputs/exp012_reports','LABEL_EVALUATION_COMPLETE.json')
    write_csv(root/'inputs/exp013_reports/channel_metrics.csv',native_rows);write_csv(root/'inputs/exp013_reports/event_notifications.csv',native_events);seal(root/'inputs/exp013_reports','COMPLETE.json')
    contract=read(CODE/'INPUT_CONTRACT.json');contract['known_required_files']=known;contract['fixture']=True
    write(root/'INPUT_CONTRACT.json',contract)
    write(root/'INPUT_INVENTORY.json',{'present':files(root/'inputs'), 'fixture':True})
    # Inventory uses paths relative to the input root, exactly like production.
    inv=read(root/'INPUT_INVENTORY.json');inv['present']={'inputs/'+n:h for n,h in inv['present'].items()};write(root/'INPUT_INVENTORY.json',inv)
    return root,source

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    r,s=create(a.output);print('CPU synthetic fixtures, not NASA results:',r,s)
