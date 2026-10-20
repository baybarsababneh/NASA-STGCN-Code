"""Measured full-shape A100 preflight; discarded fictional optimization only."""
import gc, math, os, statistics, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
from core import ROOT,configure,sha,read,write,rng,restore_rng,seeds
from data import training_data,Windows,external_scaler,transform,inject_batch
from train import model_for,NEW_ROLES
from upstream import prepare,catch_losses
from execution_data import gather,epoch_plan
from drive_transport import file_sha

def preflight(work,device,transport,cache):
    work=Path(work);cache=Path(cache);environment=configure(device)
    identity={'environment':environment,'package_manifest_sha256':sha(ROOT/'PACKAGE_MANIFEST.json')}
    result_path=work/'PERFORMANCE_PREFLIGHT.json'
    if result_path.exists():
        old=read(result_path)
        if old.get('status')!='PASS' or any(old.get(k)!=v for k,v in identity.items()):raise RuntimeError('Stored performance preflight source/runtime differs')
        print('REUSE VERIFIED MEASURED PERFORMANCE PREFLIGHT',old['projection'],flush=True);return old
    if device!='cuda':raise RuntimeError('Scientific performance preflight requires the actual A100; no CPU speed estimate can authorize the GPU run')
    if 'A100' not in environment['GPU'] or environment['GPU_total_bytes']<38*2**30:raise RuntimeError('Select an A100 40GB runtime. Scientific dimensions/batches are not reduced on another GPU.')
    import shutil
    free=shutil.disk_usage(work).free
    if free<20*2**30:raise RuntimeError('At least 20 GiB free LOCAL disk is required for streaming CATCH checkpoints and restore staging')
    quota=transport.storage_quota()
    if quota.get('limit') is not None and int(quota['limit'])-int(quota.get('usage',0))<20*2**30:
        raise RuntimeError('At least 20 GiB free Google Drive quota is required for two verified recovery generations and replacement uploads. Free space or extend quota; training has not started.')
    raw,meta=training_data(work);paths=prepare(work);saved=rng();records=[]
    try:
        for mission in ['MSL','SMAP']:
            chans=sorted(ch for ch,r in meta.items() if r['mission']==mission);nf=meta[chans[0]]['columns']
            for role in NEW_ROLES+['gdn','catch']:
                print('PERFORMANCE PREFLIGHT',mission,role,'full frozen shape; fictional model data; weights discarded',flush=True)
                seeds(17);model,cfg=model_for(role,mission,chans,nf,paths,device);model.to(device)
                if role in NEW_ROLES:
                    active=np.stack([np.ptp(raw[ch][:meta[ch]['fit_end'],1:],axis=0)>0 for ch in chans])
                    model.active_commands.copy_(torch.from_numpy(active).to(device))
                scaler=external_scaler(raw,meta,chans) if role in ['gdn','catch'] else None
                z=transform({ch:raw[ch] for ch in chans},scaler)
                fit=Windows(z,meta,chans,role,'fit',cfg.get('balanced_FIT_windows_cap_per_channel'))
                stop=Windows(z,meta,chans,role,'stop',cfg.get('STOP_windows_cap_per_channel'))
                g=torch.Generator().manual_seed(17);sampler=None
                if role in NEW_ROLES:
                    counts=Counter(ch for ch,t,e in fit.rows)
                    sampler=WeightedRandomSampler([1/counts[ch] for ch,t,e in fit.rows],len(fit),replacement=True,generator=g)
                loader=DataLoader(fit,batch_size=cfg['batch_size'],shuffle=sampler is None,sampler=sampler,generator=g,num_workers=0)
                plan=epoch_plan(loader);ids=plan[0]
                old=default_collate([fit[i] for i in ids]);new=gather(fit,ids)
                if any(not torch.equal(a,b) for a,b in zip(old,new)):raise RuntimeError('Vectorised batch differs from frozen Windows/default_collate')
                timings={}
                for label,fn in [('original_CPU_collation',lambda:default_collate([fit[i] for i in ids])),('E01_CPU_collation',lambda:gather(fit,ids))]:
                    values=[]
                    for _ in range(10):
                        beg=time.perf_counter();fn();values.append(time.perf_counter()-beg)
                    timings[label+'_median_seconds']=statistics.median(values)
                # Actual FIT counts/collation; Gaussian fictional payloads for GPU
                # microbenchmark. No NASA model is trained or selected here.
                sample=[torch.randn_like(a) if a.is_floating_point() else a.clone() for a in new]
                mask_optimizer=None;aux=None
                if role=='catch':
                    c=cfg['config'];optimizer=torch.optim.Adam([p for n,p in model.named_parameters() if 'mask_generator' not in n],lr=c['lr'],foreach=False,fused=False)
                    mask_optimizer=torch.optim.Adam(model.mask_generator.parameters(),lr=c['Mlr'],foreach=False,fused=False)
                    for op,lr in [(optimizer,c['lr']),(mask_optimizer,c['Mlr'])]:torch.optim.lr_scheduler.OneCycleLR(op,max_lr=lr,epochs=c['num_epochs'],steps_per_epoch=len(loader),pct_start=c['pct_start'])
                    aux,_=catch_losses(paths,c);mask_step=max(1,min(int(len(loader)/10),100))
                else:optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],foreach=False,fused=False)
                def step(i):
                    print('PREFLIGHT STEP START',mission,role,i+1,'batch',cfg['batch_size'],flush=True)
                    events=[torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    begin=time.perf_counter();events[0].record();item=[a.to(device) for a in sample];events[1].record()
                    model.train();optimizer.zero_grad(set_to_none=True)
                    if role in NEW_ROLES:
                        x,e,y=item;ref,l=model(x,e);inj,truth=inject_batch(x);_,li=model(inj,e)
                        loss=F.huber_loss(ref,y)+.5*(F.binary_cross_entropy_with_logits(l,torch.zeros_like(l))+F.binary_cross_entropy_with_logits(li,truth))
                    elif role=='gdn':x,y=item;loss=F.mse_loss(model(x,None),y)
                    else:
                        x=item[0];out,complex_out,dc=model(x);loss=F.mse_loss(out,x)+c['dc_lambda']*dc+c['auxi_lambda']*aux(complex_out,model.revin_layer(x,'transform'))
                        if (i+1)%mask_step==0:mask_optimizer.step();mask_optimizer.zero_grad(set_to_none=True)
                    value=float(loss.detach());events[2].record()
                    if not math.isfinite(value):raise RuntimeError('Full-shape fictional preflight objective is nonfinite')
                    loss.backward();optimizer.step();events[3].record();torch.cuda.synchronize()
                    return {'step_seconds':time.perf_counter()-begin,'transfer_ms':events[0].elapsed_time(events[1]),'forward_loss_ms':events[1].elapsed_time(events[2]),'backward_optimizer_ms':events[2].elapsed_time(events[3])}
                torch.cuda.reset_peak_memory_stats()
                reference=None;before_cold=None
                if mission=='MSL' and role=='graph':
                    import copy
                    reference=copy.deepcopy(model);before_cold=rng()
                cold=step(0)
                if reference is not None:
                    # Native first-fit shape with original loss/finite/backward/
                    # Adam arithmetic, matched to the timed amended step exactly.
                    after_cold=rng();restore_rng(before_cold);reference.train()
                    reference_op=torch.optim.Adam(reference.parameters(),lr=cfg['lr'],foreach=False,fused=False)
                    xx,ee,yy=[v.to(device) for v in sample];rr,ll=reference(xx,ee)
                    ii,tt=inject_batch(xx);_,lli=reference(ii,ee)
                    loss_ref=F.huber_loss(rr,yy)+.5*(F.binary_cross_entropy_with_logits(ll,torch.zeros_like(ll))+F.binary_cross_entropy_with_logits(lli,tt))
                    if not torch.isfinite(loss_ref):raise RuntimeError('Native original-arithmetic parity loss is nonfinite')
                    loss_ref.backward();reference_op.step()
                    from core import state_hash
                    if state_hash(reference)!=state_hash(model):raise RuntimeError('Native full-batch MSL graph original/amended step state differs')
                    del reference,reference_op,xx,ee,yy,rr,ll,ii,tt,lli,loss_ref
                    restore_rng(after_cold);gc.collect();torch.cuda.empty_cache()
                measured=[step(i) for i in range(1,4)]
                stop_times=[]
                model.eval()
                with torch.no_grad():
                    for _ in range(2):
                        begin=time.perf_counter();item=[a.to(device) for a in sample]
                        if role in NEW_ROLES:
                            x,e,y=item;ref,l=model(x,e);inj,truth=inject_batch(x);_,li=model(inj,e)
                            loss=.5*(F.binary_cross_entropy_with_logits(l,torch.zeros_like(l))+F.binary_cross_entropy_with_logits(li,truth))+.1*F.huber_loss(ref,y)
                        elif role=='gdn':x,y=item;loss=F.mse_loss(model(x,None),y)
                        else:x=item[0];out,_,_=model(x);loss=F.mse_loss(out,x)
                        if not math.isfinite(float(loss.detach())):raise RuntimeError('Preflight STOP objective nonfinite')
                        torch.cuda.synchronize();stop_times.append(time.perf_counter()-begin)
                fit_batch_seconds=statistics.median(v['step_seconds'] for v in measured)+timings['E01_CPU_collation_median_seconds']
                stop_batch_seconds=statistics.median(stop_times)+timings['E01_CPU_collation_median_seconds']
                fit_batches=math.ceil(len(fit)/cfg['batch_size']);stop_batches=math.ceil(len(stop)/cfg['batch_size'])
                record={'mission':mission,'role':role,'batch_size':cfg['batch_size'],'FIT_windows':len(fit),'FIT_batches':fit_batches,'STOP_windows':len(stop),'STOP_batches':stop_batches,'epoch_cap':cfg['epochs'],'cold_step':cold,'steady_steps':measured,'STOP_step_seconds':stop_times,'GPU_peak_GiB':torch.cuda.max_memory_allocated()/2**30,'epoch_compute_seconds_estimate':fit_batches*fit_batch_seconds+stop_batches*stop_batch_seconds,'training_loss_window_compute_seconds_estimate':600+max(v['step_seconds'] for v in measured),'full_scientific_configuration':cfg,'CPU_batch_byte_parity':'PASS',**timings}
                if mission=='MSL' and role=='graph':record['native_full_batch_original_step_state_byte_parity']='PASS'
                records.append(record);print(json_report(record),flush=True)
                if mission=='MSL' and role=='graph':
                    from torch.profiler import profile,ProfilerActivity
                    try:
                        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=True) as prof:step(4)
                        (work/'PERFORMANCE_OPERATOR_SUMMARY.txt').write_text(prof.key_averages().table(sort_by='self_device_time_total',row_limit=30))
                        prof.export_chrome_trace(str(work/'PERFORMANCE_FIRST_MSL_GRAPH_TRACE.json'))
                    except Exception as ex:
                        # Operator tracing is diagnostic, not the measured batch
                        # or resume gate. Some Colab images lack CUPTI tracing.
                        write(work/'PERFORMANCE_OPERATOR_TRACE_STATUS.json',{'available':False,'reason':str(ex),'full_batch_timings_still_measured':True})
                del item,loss
                if role in NEW_ROLES:del x,e,y,ref,l,inj,truth,li
                elif role=='gdn':del x,y
                else:del x,out
                del model,optimizer,sample,old,new
                if mask_optimizer is not None:del mask_optimizer
                gc.collect();torch.cuda.empty_cache()
        # A real API transfer establishes permissions and measured small-object
        # throughput. It does not guarantee multi-GB CATCH upload duration.
        probe=cache/'Drive_transfer_probe.bin';probe.write_bytes(os.urandom(16*2**20))
        begin=time.perf_counter();remote=transport.upload(probe,probe.name,'probe')
        transport.verify(remote['id'],file_sha(probe),probe.stat().st_size)
        probe_seconds=time.perf_counter()-begin;transport.delete(remote['id']);probe.unlink()
        # Each new role has three seeds; each published role has one.
        upper=sum(r['epoch_compute_seconds_estimate']*r['epoch_cap']*(3 if r['role'] in NEW_ROLES else 1) for r in records)
        first=sum(r['epoch_compute_seconds_estimate']*(3 if r['role'] in NEW_ROLES else 1) for r in records)
        result={'status':'PASS',**identity,'timings':records,'projection':{'all_22_one_epoch_compute_hours':first/3600,'all_22_epoch_cap_compute_hours':upper/3600,'early_stopping_duration_unknown':True,'excludes_selection_confirmation_TEST_scoring_checkpoint_serialization_network_and_contention':True,'fictional_full_shape_microbenchmark_not_scientific_training':True},'drive_probe':{'bytes':16*2**20,'upload_and_SHA_verification_seconds':probe_seconds,'multi_GiB_duration_not_guaranteed':True,'storage_quota_at_preflight':quota},'scientific_fits':0,'NASA_TEST_arrays_opened':False,'NASA_label_CSV_opened':False,'checkpoint_target_seconds':600,'hard_wall_clock_recovery_guarantee':False}
        write(result_path,result);print(json_report(result['projection']),flush=True);return result
    finally:restore_rng(saved)

def json_report(value):
    import json
    return json.dumps(value,sort_keys=True)
