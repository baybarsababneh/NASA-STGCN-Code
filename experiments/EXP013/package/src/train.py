"""Only new EXP013 models are fitted. E01 exact FIT/STOP batch resumes; scientific equations/settings preserved."""
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
    model._execution_selected_identity={'job':job_name(m,r,s),'state_sha256':st['state_sha256'],'checkpoint_sha256':st['checkpoint_sha256']}
    return model,ck

from execution_data import epoch_plan,gather,move
from execution_checkpoint import save_torch,load_torch
import execution_runtime as runtime

def validate(model, loader, role, device, *, resume=None, on_batch=None):
    state = resume if resume is not None else {}
    if 'plan' not in state:
        state.update(training_rng=rng(), cursor=0, values=[])
        seeds(912013)
        state['plan'] = epoch_plan(loader)
    model.eval()
    with torch.no_grad():
        for i in range(state['cursor'], len(state['plan'])):
            with runtime.operation('STOP_data_preparation',batch=i+1): item = gather(loader.dataset, state['plan'][i])
            if role in NEW_ROLES:
                with runtime.operation('STOP_clean_forward',batch=i+1): x,e,y = move(item,device); ref,l=model(x,e)
                with runtime.operation('STOP_injected_forward',batch=i+1): inj,truth=inject_batch(x); _,li=model(inj,e)
                loss=.5*(F.binary_cross_entropy_with_logits(l,torch.zeros_like(l))+F.binary_cross_entropy_with_logits(li,truth))+.1*F.huber_loss(ref,y)
            elif role=='gdn':
                with runtime.operation('STOP_GDN_forward',batch=i+1): x,y=move(item,device); loss=F.mse_loss(model(x,None),y)
            else:
                with runtime.operation('STOP_CATCH_forward',batch=i+1): x=item[0].to(device); out,_,_=model(x); loss=F.mse_loss(out,x)
            value=float(loss.detach())
            if not math.isfinite(value): raise FloatingPointError('Nonfinite STOP objective')
            state['values'].append((value,len(item[0]))); state['cursor']=i+1
            if on_batch: on_batch(state)
    restore_rng(state['training_rng']); model.train()
    return sum(v*n for v,n in state['values'])/sum(n for v,n in state['values'])

def train_job(work,raw,meta,mission,role,seed,paths,device,identity,
              epoch_limit=None,interrupt_after_batches=None,interrupt_phase='fit'):
    # Test interruption arguments are never exposed as scientific configuration.
    d=Path(work)/'models'/job_name(mission,role,seed);d.mkdir(parents=True,exist_ok=True)
    channels=sorted(ch for ch,m in meta.items() if m['mission']==mission);nf=meta[channels[0]]['columns']
    if any(meta[ch]['columns']!=nf for ch in channels): raise RuntimeError('Mission feature count differs')
    with runtime.operation('model_initialisation',job=d.name):
        seeds(seed);model,cfg=model_for(role,mission,channels,nf,paths,device)
    if role in NEW_ROLES:
        active=np.stack([np.ptp(raw[ch][:meta[ch]['fit_end'],1:],axis=0)>0 for ch in channels])
        model.active_commands.copy_(torch.from_numpy(active))
    with runtime.operation('model_transfer',job=d.name): model.to(device)
    scaler=external_scaler(raw,meta,channels) if role in ['gdn','catch'] else None
    # Minimum correctness repair: a 55-column MSL scaler cannot transform the
    # unrelated 25-column SMAP arrays. Scoring already uses this mission subset.
    z=transform({ch:raw[ch] for ch in channels},scaler)
    jid={'run':identity,'mission':mission,'role':role,'seed':seed,'channels':channels,'n_features':nf,'configuration':cfg,'scaler':scaler}
    freeze(d/'IDENTITY.json',jid)
    if (d/'COMPLETE.json').exists():
        done=read(d/'COMPLETE.json')
        if done['status']=='COMPLETE':
            checked(d/'selected.pt',done['checkpoint_sha256']); chosen=load_torch(d/'selected.pt')
            if chosen['identity']!=jid:raise RuntimeError('Completed model identity differs')
            model.load_state_dict(chosen['model'],strict=True)
            if state_hash(model)!=done['state_sha256']:raise RuntimeError('Completed model state differs')
        elif done['status']!='FAILED_PUBLISHED_SOURCE' or role in NEW_ROLES:
            raise RuntimeError('Invalid completed-fit status')
        print('REUSE',d.name,done['status'],flush=True);return done
    cap=cfg.get('balanced_FIT_windows_cap_per_channel');fit=Windows(z,meta,channels,role,'fit',cap)
    stop=Windows(z,meta,channels,role,'stop',cfg.get('STOP_windows_cap_per_channel'))
    seeds(seed+1000);g=torch.Generator().manual_seed(seed);sampler=None
    if role in NEW_ROLES:
        from collections import Counter
        count=Counter(ch for ch,t,e in fit.rows)
        sampler=WeightedRandomSampler([1/count[ch] for ch,t,e in fit.rows],len(fit),replacement=True,generator=g)
    loader=DataLoader(fit,batch_size=cfg['batch_size'],shuffle=sampler is None,sampler=sampler,generator=g,num_workers=0,drop_last=False)
    val=DataLoader(stop,batch_size=cfg['batch_size'],shuffle=False,num_workers=0)
    mask_optimizer=None;aux=None;schedulers=[]
    if role=='catch':
        c=cfg['config'];main=[p for n,p in model.named_parameters() if 'mask_generator' not in n]
        optimizer=torch.optim.Adam(main,lr=c['lr'],foreach=False,fused=False)
        mask_optimizer=torch.optim.Adam(model.mask_generator.parameters(),lr=c['Mlr'],foreach=False,fused=False)
        # Original constructors set initial LR, but the frozen code never steps
        # these schedulers. Retain/restore their state without introducing steps.
        schedulers=[torch.optim.lr_scheduler.OneCycleLR(op,max_lr=lr,epochs=c['num_epochs'],steps_per_epoch=len(loader),pct_start=c['pct_start'])
                    for op,lr in [(optimizer,c['lr']),(mask_optimizer,c['Mlr'])]]
        aux,_=catch_losses(paths,c)
    else: optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],foreach=False,fused=False)
    state={'checkpoint_schema':'EXP013_E01_BATCH_V1','identity':jid,'epoch':0,
           'phase':'new_epoch','best':float('inf'),'bad':0,'history':[]}
    if (d/'last.pt').exists():
        with runtime.operation('checkpoint_read',deadline=600): state=load_torch(d/'last.pt')
        if state.get('checkpoint_schema')!='EXP013_E01_BATCH_V1' or state['identity']!=jid:
            raise RuntimeError('Resume schema/scientific identity differs; no silent old-release migration')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        if mask_optimizer is not None:
            mask_optimizer.load_state_dict(state['mask_optimizer'])
            if len(state['mask_grads'])!=len(list(model.mask_generator.parameters())):raise RuntimeError('Mask gradient roster differs')
            for p,grad in zip(model.mask_generator.parameters(),state['mask_grads']):p.grad=None if grad is None else grad.to(device)
        if len(schedulers)!=len(state['schedulers']):raise RuntimeError('Scheduler roster differs')
        for scheduler,value in zip(schedulers,state['schedulers']):scheduler.load_state_dict(value)
        restore_rng(state['rng']);g.set_state(state['loader_rng'])
        # Release deserialised CPU tensor copies once live optimiser/model state
        # has been restored; capture reconstructs these fields at safe boundaries.
        for key in ['model','optimizer','mask_optimizer','mask_grads','schedulers','rng','loader_rng']:state.pop(key,None)
        print('EXACT RESUME',d.name,'epoch',state['epoch']+1,'phase',state['phase'],'next_batch',state.get('cursor',0)+1,flush=True)
    elapsed0=time.monotonic();epoch_beg=time.monotonic();session_batches=0
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    def commit(reason,force=False):
        state.update(model=model.state_dict(),optimizer=optimizer.state_dict(),
            mask_optimizer=None if mask_optimizer is None else mask_optimizer.state_dict(),
            mask_grads=[] if mask_optimizer is None else [None if p.grad is None else p.grad.detach().cpu().clone() for p in model.mask_generator.parameters()],
            schedulers=[v.state_dict() for v in schedulers],rng=rng(),loader_rng=g.get_state())
        state['elapsed_before_resume']=time.monotonic()-epoch_beg+state.get('elapsed_offset',0.)
        with runtime.operation('checkpoint_serialisation',deadline=600): save_torch(d/'last.pt',state)
        runtime.checkpoint(work,reason,force=force)
    print('FIT START',d.name,'FIT_windows',len(fit),'FIT_batches',len(loader),'STOP_windows',len(stop),'STOP_batches',len(val),'batch_size',cfg['batch_size'],'epoch_cap',cfg['epochs'],flush=True)
    try:
        while state['epoch']<cfg['epochs'] and state['bad']<cfg['patience']:
            epoch=state['epoch']
            if state['phase']=='epoch_done':
                state['epoch']+=1;state['phase']='new_epoch'
                continue
            if state['phase']=='new_epoch':
                epoch_beg=time.monotonic();state['elapsed_offset']=0.
                state.update(phase='fit',plan=epoch_plan(loader),cursor=0,sums=0.,n=0)
                state.pop('validation',None)
                if epoch==0 or runtime.due():commit('epoch_start:'+d.name,force=(epoch==0))
            else:
                epoch_beg=time.monotonic();state['elapsed_offset']=state.get('elapsed_before_resume',0.)
            model.train();mask_step=max(1,min(int(len(loader)/10),100)) if role=='catch' else None
            if state['phase']=='fit':
                for i in range(state['cursor'],len(state['plan'])):
                    batch_beg=time.monotonic()
                    with runtime.operation('data_preparation',batch=i+1): item=gather(fit,state['plan'][i])
                    data_seconds=time.monotonic()-batch_beg
                    optimizer.zero_grad(set_to_none=True)
                    if role in NEW_ROLES:
                        with runtime.operation('input_transfer',batch=i+1): x,e,y=move(item,device)
                        with runtime.operation('clean_forward',batch=i+1): ref,l=model(x,e)
                        with runtime.operation('synthetic_injection',batch=i+1): inj,truth=inject_batch(x)
                        with runtime.operation('injected_forward',batch=i+1): _,li=model(inj,e)
                        loss=F.huber_loss(ref,y)+.5*(F.binary_cross_entropy_with_logits(l,torch.zeros_like(l))+F.binary_cross_entropy_with_logits(li,truth))
                    elif role=='gdn':
                        with runtime.operation('input_transfer',batch=i+1): x,y=move(item,device)
                        with runtime.operation('forward_loss',batch=i+1): loss=F.mse_loss(model(x,None),y)
                    else:
                        with runtime.operation('input_transfer',batch=i+1): x=item[0].to(device)
                        with runtime.operation('forward',batch=i+1): out,complex_out,dcloss=model(x);normalized=model.revin_layer(x,'transform')
                        loss=F.mse_loss(out,x)+c['dc_lambda']*dcloss+c['auxi_lambda']*aux(complex_out,normalized)
                        if (i+1)%mask_step==0:mask_optimizer.step();mask_optimizer.zero_grad(set_to_none=True)
                    # One scalar transfer serves the SAME immediate finite gate
                    # and SAME Python loss accumulator; removes a duplicate sync.
                    with runtime.operation('loss_scalar_synchronisation',batch=i+1): value=float(loss.detach())
                    if not math.isfinite(value):raise FloatingPointError('Nonfinite FIT objective; baseline source retained')
                    with runtime.operation('backward',batch=i+1): loss.backward()
                    runtime.gradients(model)
                    with runtime.operation('optimizer_step',batch=i+1): optimizer.step()
                    state['sums']+=value*len(item[0]);state['n']+=len(item[0]);state['cursor']=i+1;session_batches+=1
                    runtime.progress(work,d.name,'FIT',epoch+1,i+1,len(state['plan']),epoch_beg,cfg['epochs'],epoch_elapsed=state.get('elapsed_offset',0.),data_seconds=data_seconds,batch_seconds=time.monotonic()-batch_beg)
                    if runtime.due():commit('batch:'+d.name)
                    if interrupt_phase=='fit' and interrupt_after_batches and session_batches>=interrupt_after_batches:
                        commit('fixture_pause',force=True);return {'status':'PAUSED_FOR_TEST'}
                state['phase']='stop';state['validation']={}
                if runtime.due():commit('before_STOP:'+d.name)
            if state['phase']=='stop':
                def stop_batch(v):
                    nonlocal session_batches
                    session_batches+=1
                    runtime.progress(work,d.name,'STOP',epoch+1,v['cursor'],len(v['plan']),epoch_beg,cfg['epochs'],epoch_elapsed=state.get('elapsed_offset',0.))
                    if runtime.due():commit('STOP_batch:'+d.name)
                    if interrupt_phase=='stop' and interrupt_after_batches and session_batches>=interrupt_after_batches:
                        commit('fixture_STOP_pause',force=True);raise PauseForTest()
                with runtime.operation('STOP_validation',deadline=600): score=validate(model,val,role,device,resume=state['validation'],on_batch=stop_batch)
                improve=score<=state['best'] if role=='catch' else score<state['best']
                if improve:
                    state['best']=score;state['bad']=0
                    save_torch(d/'selected.pt',{'identity':jid,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'channels':channels,'n_features':nf,'configuration':cfg,'scaler':scaler,'epoch':epoch+1,'STOP_objective':score})
                else:state['bad']+=1
                if role=='catch':
                    lr=c['lr']*.5**epoch
                    for op in [optimizer,mask_optimizer]:
                        for pg in op.param_groups:pg['lr']=lr
                seconds=time.monotonic()-epoch_beg+state.get('elapsed_offset',0.)
                state['history'].append({'epoch':epoch+1,'FIT_loss':state['sums']/state['n'],'STOP_objective':score,'selected':improve,'seconds':seconds})
                state['phase']='epoch_done';state.pop('validation',None)
                commit('epoch_complete:'+d.name)
                write(d/'history.json',state['history'])
                print('EPOCH COMPLETE',d.name,'epoch',epoch+1,'STOP',round(score,6),'FIT_loss',round(state['sums']/state['n'],6),'seconds',round(seconds,1),flush=True)
                if epoch_limit and epoch+1>=epoch_limit:return {'status':'PAUSED_FOR_TEST'}
        chosen=load_torch(d/'selected.pt');model.load_state_dict(chosen['model'])
        from collections import Counter
        fit_count=Counter(ch for ch,t,e in fit.rows);stop_count=Counter(ch for ch,t,e in stop.rows)
        result={'status':'COMPLETE','checkpoint_sha256':sha(d/'selected.pt'),'state_sha256':state_hash(model),'epoch':chosen['epoch'],'fit_windows':len(fit),'stop_windows':len(stop),'FIT_windows_by_channel':{ch:fit_count[ch] for ch in channels},'STOP_windows_by_channel':{ch:stop_count[ch] for ch in channels},'zero_FIT_window_channels':[ch for ch in channels if not fit_count[ch]],'parameters':sum(p.numel() for p in model.parameters()),'effective_trainable_parameters':sum(p.numel() for name,p in model.named_parameters() if role!='removed' or not name.startswith(('command','query','key','message'))),'run_seconds_this_session':time.monotonic()-elapsed0,'peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,'configuration':cfg}
        write(d/'history.json',state['history']);write(d/'COMPLETE.json',result)
        (d/'last.pt').unlink(missing_ok=True);Path(str(d/'last.pt')+'.sha.json').unlink(missing_ok=True)
        runtime.checkpoint(work,'completed_fit:'+d.name,force=True)
        return result
    except PauseForTest:return {'status':'PAUSED_FOR_TEST'}
    except FloatingPointError as exc:
        if role in NEW_ROLES:raise
        result={'status':'FAILED_PUBLISHED_SOURCE','reason':str(exc),'configuration':cfg};write(d/'COMPLETE.json',result)
        runtime.checkpoint(work,'published_failure:'+d.name,force=True);print(d.name,result,flush=True);return result

class PauseForTest(Exception):pass

def train_all(work,raw,meta,paths,device,identity,on_complete=None):
    for index,(m,r,s) in enumerate(jobs()):
        print('JOB',index+1,'OF',len(jobs()),job_name(m,r,s),flush=True)
        already_complete=(Path(work)/'models'/job_name(m,r,s)/'COMPLETE.json').exists()
        train_job(work,raw,meta,m,r,s,paths,device,identity)
        if on_complete and not already_complete:on_complete()
