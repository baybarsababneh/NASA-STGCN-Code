"""One-sided scores, observed target t available at issue t. No adaptive TEST thresholds."""
import math,hashlib
from core import *
from train import NEW_ROLES,get_model,job_name
from data import transform,cases
from upstream import catch_losses
from historical_metrics import runs,channel,aggregate

def stable_seed(key): return int(hashlib.sha256(str(key).encode()).hexdigest()[:8],16)%2**31
@torch.no_grad()
def scores(model,kind,a,entity,start,end,device,paths,cfg,seed=17):
    look=64 if kind in NEW_ROLES else 5 if kind=='gdn' else 191
    if start<look or end>len(a) or start>=end: raise ValueError('Invalid causal score support')
    saved=rng();seeds(seed);model.eval();out=[];batch=256 if kind in NEW_ROLES else cfg['batch_size']
    criterion=None
    if kind=='catch': _,criterion=catch_losses(paths,cfg['config'])
    for lo in range(start,end,batch):
        ids=range(lo,min(end,lo+batch))
        if kind in NEW_ROLES:
            x=np.stack([a[t-64:t+1] for t in ids]);x[:,-1,1:]=0
            xb=torch.from_numpy(x).to(device);e=torch.full((len(x),),entity,dtype=torch.long,device=device)
            _,v=model(xb,e);v=v.cpu().numpy()
        elif kind=='gdn':
            x=np.ascontiguousarray(np.stack([a[t-5:t].T for t in ids]));xb=torch.from_numpy(x).to(device);v=np.abs(model(xb,None).cpu().numpy()-a[lo:lo+len(x)])
        else:
            x=np.stack([a[t-191:t+1] for t in ids]);xb=torch.from_numpy(x).to(device);re,_,_=model(xb)
            temp=(xb-re).square().mean(-1);freq=criterion(xb,re).mean(-1)
            v=(temp+cfg['config']['score_lambda']*freq)[:,-1].cpu().numpy()
        if not np.isfinite(v).all(): raise FloatingPointError('Nonfinite native '+kind+' score; no substitution is permitted')
        out.append(v)
    restore_rng(saved);return np.concatenate(out)

def causal_mean(v,span):
    v=np.asarray(v,np.float64)
    if span==1:return v.copy()
    if span<1:raise ValueError('Invalid span')
    # Fixed EWMA, initialized with first observed score. No future startup averaging.
    alpha=2/(span+1);y=np.empty_like(v);y[0]=v[0]
    for i in range(1,len(v)): y[i]=alpha*v[i]+(1-alpha)*y[i-1]
    return y
def gdn_native(v,stats):
    if stats is None: raise ValueError('GDN needs TRAIN-only score normalization')
    z=(v-np.asarray(stats['median']))/(np.abs(stats['iqr'])+.01)
    y=np.zeros_like(z)
    for i in range(3,len(z)):y[i]=z[i-3:i+1].mean(0)
    return y.max(1)
def native(v,role,stats): return gdn_native(v,stats) if role=='gdn' else np.asarray(v,np.float64)
def quantile(v,q):
    if len(v)<16 or not np.isfinite(v).all(): raise ValueError('Calibration support insufficient/nonfinite')
    return float(np.quantile(v,q,method='higher'))
def occupancy(f): return float(np.mean(f)) if len(f) else float('nan')
def starts_rate(f): return len(runs(f))*1000/len(f) if len(f) else float('nan')

def calibrated(work,raw,meta,paths,device,m,r,s):
    model,ck=get_model(work,m,r,s,paths,device)
    if model is None:return None,ck
    chans=ck['channels'];scaler=ck['scaler'];z=transform({ch:raw[ch] for ch in chans},scaler)
    calibration={};baseline_stats={};full={}
    cache=Path(work)/'nominal'/job_name(m,r,s);cache.mkdir(parents=True,exist_ok=True)
    identity={'model_state_sha256':status_hash(work,m,r,s),'data_freeze_sha256':sha(Path(work)/'DATA_FREEZE.json')}
    if (cache/'COMPLETE.json').exists():
        done=manifest(cache,'COMPLETE.json')
        if done['identity']!=identity:raise RuntimeError('Nominal cache identity differs')
        baseline_stats=read(cache/'stats.json')
        for ch in chans:
            a=arrays(cache/(ch+'.npz'));start=int(a['start']);full[ch]=(start,a['native']);calibration[ch]=a['native'][meta[ch]['stop_end']-start:meta[ch]['cal_end']-start]
        return (model,ck,z,calibration,baseline_stats,full),{'status':'COMPLETE'}
    for e,ch in enumerate(chans):
        row=meta[ch];start=64 if r in NEW_ROLES else 5 if r=='gdn' else 191
        v=scores(model,r,z[ch],e,start,row['cal_end'],device,paths,ck['configuration'],stable_seed((m,r,s,ch,'cal')))
        stats=None
        if r=='gdn':
            nominal=v[row['stop_end']-start:row['cal_end']-start]
            stats={'median':np.median(nominal,0).tolist(),'iqr':(np.quantile(nominal,.75,axis=0)-np.quantile(nominal,.25,axis=0)).tolist(),'normalization_source':'TRAIN80:90 only; original released get_err_scores instead normalizes on TEST and is deliberately not called'}
        n=native(v,r,stats);full[ch]=(start,n)
        calibration[ch]=n[row['stop_end']-start:row['cal_end']-start]
        baseline_stats[ch]=stats
        npz(cache/(ch+'.npz'),start=np.array(start),native=n)
        print('CALIBRATION CACHE',m,r,s,ch,flush=True)
    write(cache/'stats.json',baseline_stats);seal(cache,'COMPLETE.json',{'identity':identity})
    return (model,ck,z,calibration,baseline_stats,full),{'status':'COMPLETE'}

def select_bank(work,raw,meta,paths,device,m,r):
    out=Path(work)/'development'/f'{m}__{r}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists(): manifest(out,'COMPLETE.json');return read(out/'selection.json')
    try:state,st=calibrated(work,raw,meta,paths,device,m,r,17)
    except FloatingPointError as exc:
        if r in NEW_ROLES:raise
        result={'status':'FAILED_PUBLISHED_NATIVE_SCORE','mission':m,'role':r,'reason':str(exc)}
        write(out/'selection.json',result);seal(out,'COMPLETE.json');return result
    if state is None:
        write(out/'selection.json',{'status':st['status'],'reason':st.get('reason'),'mission':m,'role':r});seal(out,'COMPLETE.json');return read(out/'selection.json')
    model,ck,z,cal,stats,full=state;chans=ck['channels'];bank=[]
    try:
        for e,ch in enumerate(chans):
            print('SYNTHETIC SELECT',m,r,ch,flush=True)
            row=meta[ch];look=64 if r in NEW_ROLES else 5 if r=='gdn' else 191
            for case,b in cases(raw[ch],row,ch,'select'):
                if not case['valid']: bank.append({**case,'channel':ch,'valid':False});continue
                start=max(look,case['start']-max(32,look));stop=case['stop'];base=raw[ch]
                mutated=transform({ch:b},ck['scaler'])[ch]
                seq_seed=stable_seed((m,r,17,ch,case['id']))
                clean=scores(model,r,z[ch],e,start,stop,device,paths,ck['configuration'],seq_seed)
                inj=scores(model,r,mutated,e,start,stop,device,paths,ck['configuration'],seq_seed)
                bank.append({**case,'channel':ch,'valid':True,'origin':start,'clean':native(clean,r,stats[ch]),'injected':native(inj,r,stats[ch])})
        grid=[];case_records=[]
        for span in [1,8]:
            for q in [.99,.995,.999]:
                th={ch:quantile(causal_mean(full[ch][1],span)[meta[ch]['stop_end']-full[ch][0]:meta[ch]['cal_end']-full[ch][0]],q) for ch in chans}
                clean_occup=[];clean_starts=[];families={};rows=[]
                for ch in chans:
                    start,n=full[ch];smooth=causal_mean(n,span);f=smooth[meta[ch]['stop_end']-start:meta[ch]['cal_end']-start]>th[ch]
                    clean_occup.append(occupancy(f));clean_starts.append(starts_rate(f))
                for item in bank:
                    if not item['valid']:continue
                    lo=item['start']-item['origin'];hi=item['stop']-item['origin'];on=item['onset']-item['origin'];end=item['end']-item['origin']
                    bg=causal_mean(item['clean'],span)>th[item['channel']];fg=causal_mean(item['injected'],span)>th[item['channel']]
                    background=bool(bg[on:end].any());hit=bool(fg[on:end].any());incremental=hit and not background
                    truth=np.zeros(hi-lo,bool);truth[max(0,on-lo):min(hi-lo,end-lo)]=True;flags=fg[lo:hi]
                    tp=int((flags&truth).sum());fp=int((flags&~truth).sum());fn=int((~flags&truth).sum());p=tp/(tp+fp) if tp+fp else 0.;rec=tp/(tp+fn) if tp+fn else 0.
                    f05=1.25*p*rec/(.25*p+rec) if .25*p+rec else 0.
                    cr={k:v for k,v in item.items() if k not in ['clean','injected']};cr.update(span=span,q=q,background_overlap=background,unconditional_injected_hit=hit,incremental_hit=incremental,point_f0_5=f05)
                    case_records.append(cr);families.setdefault(item['family'],[]).append(cr)
                by_family={k:{'incremental_recall':np.mean([v['incremental_hit'] for v in vs]),'unconditional_recall':np.mean([v['unconditional_injected_hit'] for v in vs]),'background_overlap':np.mean([v['background_overlap'] for v in vs]),'clean_window_conditional_recall':np.mean([v['unconditional_injected_hit'] for v in vs if not v['background_overlap']]) if any(not v['background_overlap'] for v in vs) else None,'point_f0_5':np.mean([v['point_f0_5'] for v in vs]),'cases':len(vs)} for k,vs in families.items()}
                nominal=float(np.mean(clean_occup));starts=float(np.mean(clean_starts));ok=nominal<=.01 and starts<=1. and all(any(v['channel']==ch and v['valid'] for v in bank) for ch in chans)
                objective=float(np.mean([.5*v['incremental_recall']+.5*v['point_f0_5'] for v in by_family.values()])) if by_family else 0.
                grid.append({'span':span,'q':q,'objective':objective,'CAL_nominal_occupancy_macro':nominal,'CAL_starts_per_1000_macro':starts,'feasible':ok,'families':by_family})
        feasible=[v for v in grid if v['feasible']]
        winner=max(feasible,key=lambda v:(v['objective'],-v['CAL_nominal_occupancy_macro'],-v['span'],v['q'])) if feasible else next(v for v in grid if v['span']==1 and v['q']==.999)
        result={'status':'SELECTED_TRAIN_SYNTHETIC' if feasible else 'NO_FEASIBLE_CANDIDATE','mission':m,'role':r,'selection_seed':17,'winner':winner,'grid':grid,'invalid_unchanged_cases':sum(not v['valid'] for v in bank),'selection_labels':'synthetic only; NASA labels not opened','no_real_anomaly_superiority_claim':True}
        write(out/'selection.json',result);write(out/'case_diagnostics.json',case_records);seal(out,'COMPLETE.json');return result
    except FloatingPointError as exc:
        result={'status':'FAILED_PUBLISHED_NATIVE_SCORE','mission':m,'role':r,'reason':str(exc)}
        if r in NEW_ROLES:raise
        write(out/'selection.json',result);seal(out,'COMPLETE.json');return result

def threshold_all(work,raw,meta,paths,device,m,r,s,choice):
    out=Path(work)/'calibration'/job_name(m,r,s);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists(): manifest(out,'COMPLETE.json');return read(out/'thresholds.json')
    state,st=calibrated(work,raw,meta,paths,device,m,r,s)
    if state is None: raise RuntimeError('Missing selected model')
    model,ck,z,cal,stats,full=state;span=choice['winner']['span'];q=choice['winner']['q'];rows={}
    for ch,(start,n) in full.items():
        v=causal_mean(n,span)[meta[ch]['stop_end']-start:meta[ch]['cal_end']-start]
        rows[ch]={'threshold':quantile(v,q),'n_calibration':len(v),'span':span,'q':q,'native_stats':stats[ch],'nominal_occupancy':occupancy(v>quantile(v,q)),'finite':True}
    result={'mission':m,'role':r,'seed':s,'selection_status':choice['status'],'channels':rows,'model_state_sha256':status_hash(work,m,r,s)}
    write(out/'thresholds.json',result);seal(out,'COMPLETE.json');return result
def status_hash(work,m,r,s):return read(Path(work)/'models'/job_name(m,r,s)/'COMPLETE.json')['state_sha256']

def confirm(work,raw,meta,paths,device,m,r,s,thresholds):
    out=Path(work)/'confirmation'/job_name(m,r,s);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():manifest(out,'COMPLETE.json');return read(out/'results.json')
    model,ck=get_model(work,m,r,s,paths,device);z=transform({ch:raw[ch] for ch in ck['channels']},ck['scaler']);rows=[];nominal=[]
    for e,ch in enumerate(ck['channels']):
        print('SYNTHETIC CONFIRM',m,r,s,ch,flush=True)
        row=meta[ch];t=thresholds['channels'][ch];look=64 if r in NEW_ROLES else 5 if r=='gdn' else 191
        v=scores(model,r,z[ch],e,look,row['rows'],device,paths,ck['configuration'],stable_seed((m,r,s,ch,'confirm-nominal')))
        n=native(v,r,t['native_stats']);f=causal_mean(n,t['span'])[row['cal_end']-look:]>t['threshold']
        nominal.append({'channel':ch,'samples':len(f),'occupancy':occupancy(f),'starts_per_1000':starts_rate(f)})
        for case,b in cases(raw[ch],row,ch,'confirm'):
            cr={**case,'channel':ch}
            if not case['valid']:rows.append(cr);continue
            start=max(look,case['start']-max(32,look));stop=case['stop'];on=case['onset']-start;end=case['end']-start
            seed=stable_seed((m,r,s,ch,case['id']));muta=transform({ch:b},ck['scaler'])[ch]
            bg=native(scores(model,r,z[ch],e,start,stop,device,paths,ck['configuration'],seed),r,t['native_stats'])
            fg=native(scores(model,r,muta,e,start,stop,device,paths,ck['configuration'],seed),r,t['native_stats'])
            bflag=causal_mean(bg,t['span'])>t['threshold'];flag=causal_mean(fg,t['span'])>t['threshold']
            background=bool(bflag[on:end].any());hit=bool(flag[on:end].any());where=np.flatnonzero(flag[on:end]);delay=int(where[0]) if len(where) else None
            ix=np.arange(case['start'],stop);sl=slice(case['start']-start,None);ff=flag[sl]
            metric,_,_=channel([(case['onset'],case['end'])],ix,ff,np.where(ff,ix,-1),fg[sl],fg[sl])
            cr.update(background_overlap=background,unconditional_hit=hit,incremental_hit=hit and not background,delay=delay,point_f0_5=metric['point_f0_5'],point_f1=metric['point_f1'],point_precision=metric['point_precision'],point_recall=metric['point_recall']);rows.append(cr)
    result={'mission':m,'role':r,'seed':s,'thresholds_sha256':digest(thresholds),'cases':rows,'nominal':nominal,'mechanism_confirmation_only':True,'never_use_to_retune':True}
    write(out/'results.json',result);seal(out,'COMPLETE.json');return result
