"""Only new EXP013 models are fitted. Epoch-boundary resumes include RNG and mask gradients."""
import time, copy, math
from pathlib import Path
from torch.utils.data import DataLoader,WeightedRandomSampler
import torch.nn.functional as F
from core import *
from data import Windows,inject_batch,external_scaler,transform
from architectures import CommandReferenceGCN
from upstream import factory,catch_losses

NEW_ROLES=['graph','removed','fusion']
def jobs():
    return [(m,r,s) for m in ['MSL','SMAP'] for r in NEW_ROLES for s in [17,29,43]]+[(m,r,17) for m in ['MSL','SMAP'] for r in ['gdn','catch']]
def model_for(role,mission,channels,nf,paths,device):
    if role in NEW_ROLES: return CommandReferenceGCN(nf,len(channels),role),{'epochs':25,'patience':5,'batch_size':256,'lr':.001,'lookback':64,'balanced_FIT_windows_cap_per_channel':2048,'STOP_windows_cap_per_channel':32,'loss':'Huber(reference, clean target) + .5*(BCE(clean,0)+BCE(injected,actual_changed))','checkpoint_objective':'balanced synthetic BCE + .1 clean Huber; no NASA labels'}
    return factory(role,mission,nf,paths,device)

def job_name(m,r,s): return f'{m}__{r}__{s}'
def status(work,m,r,s): return read(Path(work)/'models'/job_name(m,r,s)/'COMPLETE.json')
def get_model(work,m,r,s,paths,device):
    d=Path(work)/'models'/job_name(m,r,s);st=read(d/'COMPLETE.json')
    if st['status']!='COMPLETE': return None,st
    checked(d/'selected.pt',st['checkpoint_sha256']);ck=load_torch(d/'selected.pt')
    model,cfg=model_for(r,m,ck['channels'],ck['n_features'],paths,device);model.load_state_dict(ck['model'],strict=True);model.to(device).eval()
    if state_hash(model)!=st['state_sha256']: raise RuntimeError('Selected state hash mismatch')
    return model,ck

def validate(model,loader,role,device):
    saved=rng();seeds(912013);model.eval();vals=[]
    with torch.no_grad():
        for item in loader:
            if role in NEW_ROLES:
                x,e,y=[a.to(device) for a in item];ref,l=model(x,e);inj,truth=inject_batch(x);_,li=model(inj,e)
                loss=.5*(F.binary_cross_entropy_with_logits(l,torch.zeros_like(l))+F.binary_cross_entropy_with_logits(li,truth))+.1*F.huber_loss(ref,y)
            elif role=='gdn':
                x,y=[a.to(device) for a in item];loss=F.mse_loss(model(x,None),y)
            else:
                x,_=item;x=x.to(device);out,_,_=model(x);loss=F.mse_loss(out,x)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite STOP objective')
            vals.append((float(loss),len(item[0])))
    restore_rng(saved);model.train();return sum(v*n for v,n in vals)/sum(n for v,n in vals)

def train_job(work,raw,meta,mission,role,seed,paths,device,identity,epoch_limit=None):
    d=Path(work)/'models'/job_name(mission,role,seed);d.mkdir(parents=True,exist_ok=True)
    channels=sorted(ch for ch,m in meta.items() if m['mission']==mission);nf=meta[channels[0]]['columns']
    if any(meta[ch]['columns']!=nf for ch in channels): raise RuntimeError('Mission feature count differs')
    seeds(seed);model,cfg=model_for(role,mission,channels,nf,paths,device)
    if role in NEW_ROLES:
        active=np.stack([np.ptp(raw[ch][:meta[ch]['fit_end'],1:],axis=0)>0 for ch in channels]);model.active_commands.copy_(torch.from_numpy(active))
    model.to(device)
    scaler=external_scaler(raw,meta,channels) if role in ['gdn','catch'] else None;z=transform(raw,scaler)
    jid={'run':identity,'mission':mission,'role':role,'seed':seed,'channels':channels,'n_features':nf,'configuration':cfg,'scaler':scaler}
    freeze(d/'IDENTITY.json',jid)
    if (d/'COMPLETE.json').exists():
        done=read(d/'COMPLETE.json')
        if done['status']=='COMPLETE': checked(d/'selected.pt',done['checkpoint_sha256'])
        print('REUSE',d.name,done['status'],flush=True);return done
    cap=cfg.get('balanced_FIT_windows_cap_per_channel');fit=Windows(z,meta,channels,role,'fit',cap)
    stop=Windows(z,meta,channels,role,'stop',cfg.get('STOP_windows_cap_per_channel'))
    # Match the data order and synthetic perturbations between all new controls.
    seeds(seed+1000);g=torch.Generator().manual_seed(seed)
    sampler=None
    if role in NEW_ROLES:
        from collections import Counter
        count=Counter(ch for ch,t,e in fit.rows)
        sampler=WeightedRandomSampler([1/count[ch] for ch,t,e in fit.rows],len(fit),replacement=True,generator=g)
    loader=DataLoader(fit,batch_size=cfg['batch_size'],shuffle=sampler is None,sampler=sampler,generator=g,num_workers=0,drop_last=False)
    val=DataLoader(stop,batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
    mask_optimizer=None;aux=None
    if role=='catch':
        c=cfg['config'];main=[p for n,p in model.named_parameters() if 'mask_generator' not in n]
        optimizer=torch.optim.Adam(main,lr=c['lr'],foreach=False,fused=False)
        mask_optimizer=torch.optim.Adam(model.mask_generator.parameters(),lr=c['Mlr'],foreach=False,fused=False)
        # Constructor changes initial LR, exactly as the official OneCycleLR constructors.
        torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=c['lr'],epochs=c['num_epochs'],steps_per_epoch=len(loader),pct_start=c['pct_start'])
        torch.optim.lr_scheduler.OneCycleLR(mask_optimizer,max_lr=c['Mlr'],epochs=c['num_epochs'],steps_per_epoch=len(loader),pct_start=c['pct_start'])
        aux,_=catch_losses(paths,c)
    else: optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],foreach=False,fused=False)
    start=0;best=float('inf');bad=0;history=[]
    if (d/'last.pt').exists():
        ck=load_torch(d/'last.pt')
        if ck['identity']!=jid: raise RuntimeError('Resume identity differs')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        if mask_optimizer is not None:
            mask_optimizer.load_state_dict(ck['mask_optimizer'])
            for p,grad in zip(model.mask_generator.parameters(),ck['mask_grads']): p.grad=None if grad is None else grad.to(device)
        restore_rng(ck['rng']);g.set_state(ck['loader_rng']);start=ck['epoch']+1;best=ck['best'];bad=ck['bad'];history=ck['history']
        print('RESUME',d.name,'epoch',start+1,flush=True)
    elapsed0=time.monotonic()
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(start,cfg['epochs']):
            # A STOP-completed last epoch must not be repeated on an interruption.
            if bad>=cfg['patience']: break
            model.train();sums=0.;n=0;beg=time.monotonic()
            mask_step=max(1,min(int(len(loader)/10),100)) if role=='catch' else None
            for i,item in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                if role in NEW_ROLES:
                    x,e,y=[a.to(device) for a in item];ref,l=model(x,e);inj,truth=inject_batch(x);_,li=model(inj,e)
                    loss=F.huber_loss(ref,y)+.5*(F.binary_cross_entropy_with_logits(l,torch.zeros_like(l))+F.binary_cross_entropy_with_logits(li,truth))
                elif role=='gdn':
                    x,y=[a.to(device) for a in item];loss=F.mse_loss(model(x,None),y)
                else:
                    x=item[0].to(device);out,complex_out,dcloss=model(x)
                    normalized=model.revin_layer(x,'transform')
                    loss=F.mse_loss(out,x)+c['dc_lambda']*dcloss+c['auxi_lambda']*aux(complex_out,normalized)
                    # Preserve the upstream mask optimizer's previous-gradient ordering.
                    if (i+1)%mask_step==0: mask_optimizer.step();mask_optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite FIT objective; baseline source retained')
                loss.backward();optimizer.step();sums+=float(loss.detach())*len(item[0]);n+=len(item[0])
                if role in ['gdn','catch'] and (i+1)%100==0:
                    print(d.name,'epoch',epoch+1,'batch',i+1,'of',len(loader),'elapsed_seconds',round(time.monotonic()-beg,1),flush=True)
            score=validate(model,val,role,device)
            improve=score<=best if role=='catch' else score<best
            if improve:
                best=score;bad=0
                save_torch(d/'selected.pt',{'identity':jid,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'channels':channels,'n_features':nf,'configuration':cfg,'scaler':scaler,'epoch':epoch+1,'STOP_objective':score})
            else: bad+=1
            if role=='catch':
                # Official type1 applies config.learning_rate to BOTH optimizers.
                lr=c['lr']*.5**epoch
                for op in [optimizer,mask_optimizer]:
                    for pg in op.param_groups: pg['lr']=lr
            history.append({'epoch':epoch+1,'FIT_loss':sums/n,'STOP_objective':score,'selected':improve,'seconds':time.monotonic()-beg})
            save_torch(d/'last.pt',{'identity':jid,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'mask_optimizer':None if mask_optimizer is None else mask_optimizer.state_dict(),'mask_grads':[] if mask_optimizer is None else [None if p.grad is None else p.grad.detach().cpu().clone() for p in model.mask_generator.parameters()],'rng':rng(),'loader_rng':g.get_state(),'epoch':epoch,'best':best,'bad':bad,'history':history})
            write(d/'history.json',history)
            print(d.name,'epoch',epoch+1,'STOP',round(score,6),'seconds',round(history[-1]['seconds'],1),flush=True)
            if epoch_limit and epoch+1>=epoch_limit: return {'status':'PAUSED_FOR_TEST'}
        chosen=load_torch(d/'selected.pt');model.load_state_dict(chosen['model'])
        from collections import Counter
        fit_count=Counter(ch for ch,t,e in fit.rows);stop_count=Counter(ch for ch,t,e in stop.rows)
        result={'status':'COMPLETE','checkpoint_sha256':sha(d/'selected.pt'),'state_sha256':state_hash(model),'epoch':chosen['epoch'],'fit_windows':len(fit),'stop_windows':len(stop),'FIT_windows_by_channel':{ch:fit_count[ch] for ch in channels},'STOP_windows_by_channel':{ch:stop_count[ch] for ch in channels},'zero_FIT_window_channels':[ch for ch in channels if not fit_count[ch]],'parameters':sum(p.numel() for p in model.parameters()),'effective_trainable_parameters':sum(p.numel() for name,p in model.named_parameters() if role!='removed' or not name.startswith(('command','query','key','message'))),'run_seconds_this_session':time.monotonic()-elapsed0,'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,'configuration':cfg}
        write(d/'COMPLETE.json',result)
        # Completed jobs need only their selected weights, not a duplicate optimizer snapshot.
        (d/'last.pt').unlink(missing_ok=True);Path(str(d/'last.pt')+'.sha.json').unlink(missing_ok=True)
        return result
    except FloatingPointError as exc:
        if role in NEW_ROLES: raise
        # Published numerical failure is explicit, never replaced with invented scores.
        result={'status':'FAILED_PUBLISHED_SOURCE','reason':str(exc),'configuration':cfg};write(d/'COMPLETE.json',result);print(d.name,result,flush=True);return result

def train_all(work,raw,meta,paths,device,identity,on_complete=None):
    for m,r,s in jobs():
        already_complete=(Path(work)/'models'/job_name(m,r,s)/'COMPLETE.json').exists()
        train_job(work,raw,meta,m,r,s,paths,device,identity)
        if on_complete and not already_complete:on_complete()
