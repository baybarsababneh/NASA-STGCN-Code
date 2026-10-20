"""A real first epoch, then single full-batch checks. No profiler or sweeps."""
import copy, gc, math, shutil, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import core, data, train, upstream, execution_runtime as runtime
from execution_checkpoint import save_torch, load_torch
from execution_data import gather

STATE_KEYS=['model','optimizer','mask_optimizer','mask_grads','schedulers','rng',
            'loader_rng','best','bad','epoch','phase','plan','cursor','sums','n']

def exact(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

def cpu_copy(x):
    if isinstance(x,torch.Tensor): return x.detach().cpu().clone()
    if isinstance(x,dict): return {k:cpu_copy(v) for k,v in x.items()}
    if isinstance(x,list): return [cpu_copy(v) for v in x]
    if isinstance(x,tuple): return tuple(cpu_copy(v) for v in x)
    return copy.deepcopy(x)

def smoke(work, raw, meta, paths, device, identity):
    out=work/'engineering/smoke'; out.mkdir(parents=True,exist_ok=True)
    receipt=work/'SMOKE_GATE.json'
    if receipt.exists():
        record=core.read(receipt)
        if record['identity']!=identity or record['status']!='PASS': raise RuntimeError('Smoke identity differs')
        for n,h in record['files'].items(): core.checked(work/n,h)
        print('REUSE VERIFIED REAL-EPOCH SMOKE',flush=True); return record
    d=work/'models/MSL__graph__17'
    start=time.monotonic()
    # Unchanged real training path/settings. This epoch is retained for the fit.
    if (out/'epoch1.pt').exists():
        boundary=load_torch(out/'epoch1.pt')
        if not (out/'selected_epoch1.pt').exists():
            selected=load_torch(d/'selected.pt')
            if selected['epoch']!=1 or selected['identity']!=boundary['identity']:
                raise RuntimeError('Interrupted smoke baseline has no valid epoch-1 selected state')
            core.atomic(out/'selected_epoch1.pt',(d/'selected.pt').read_bytes())
        load_torch(out/'selected_epoch1.pt'); del boundary
        core.atomic(d/'last.pt',(out/'epoch1.pt').read_bytes()); core.atomic(d/'selected.pt',(out/'selected_epoch1.pt').read_bytes())
    else:
        already_done=False
        if (d/'last.pt').exists():
            previous=load_torch(d/'last.pt')
            already_done=previous['phase']=='epoch_done' and previous['epoch']==0 and len(previous['history'])==1
            del previous
        if not already_done:
            with runtime.operation('real_epoch_1',deadline=1200):
                result=train.train_job(work,raw,meta,'MSL','graph',17,paths,device,identity,epoch_limit=1)
            if result['status']!='PAUSED_FOR_TEST': raise RuntimeError('Smoke did not stop at the intended epoch boundary')
    state=load_torch(d/'last.pt')
    if state['phase']!='epoch_done' or state['epoch']!=0 or len(state['history'])!=1:
        raise RuntimeError('Smoke must complete one full FIT+STOP epoch, not a reduced surrogate')
    epoch=state['history'][0]; del state
    # Keep the epoch boundary immutable, even if a replay is interrupted.
    if not (out/'epoch1.pt').exists(): core.atomic(out/'epoch1.pt',(d/'last.pt').read_bytes())
    if not (out/'selected_epoch1.pt').exists(): core.atomic(out/'selected_epoch1.pt',(d/'selected.pt').read_bytes())
    def one_batch():
        core.atomic(d/'last.pt',(out/'epoch1.pt').read_bytes())
        core.atomic(d/'selected.pt',(out/'selected_epoch1.pt').read_bytes())
        return train.train_job(work,raw,meta,'MSL','graph',17,paths,device,identity,interrupt_after_batches=1)
    with runtime.operation('restore_replay_A',deadline=600): one_batch()
    core.atomic(out/'replay_A.pt',(d/'last.pt').read_bytes())
    with runtime.operation('restore_replay_B',deadline=600): one_batch()
    a=load_torch(out/'replay_A.pt'); b=load_torch(d/'last.pt')
    if any(not exact(a[k],b[k]) for k in STATE_KEYS): raise RuntimeError('Actual GPU checkpoint replay differs')
    if a['phase']!='fit' or a['epoch']!=1 or a['cursor']!=1: raise RuntimeError('Replay sampler position differs')
    # Do not discard the second-epoch batch. train-select resumes at batch 2.
    core.atomic(out/'replay_B.pt',(d/'last.pt').read_bytes())
    n=len(a['plan']); del a,b
    record={'status':'PASS','identity':identity,'actual_GPU':device=='cuda',
            'full_real_epoch_completed':True,'epoch1':epoch,
            'FIT_batches_per_epoch':n,'elapsed_smoke_session_seconds':time.monotonic()-start,
            'measured_peak_GPU_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'measured_peak_GPU_reserved_bytes':torch.cuda.max_memory_reserved() if device=='cuda' else None,
            'checkpoint_restore_replay':'exact model/Adam/RNG/sampler/objective state',
            'first_fit_continues_from':'epoch 2 batch 2',
            'NASA_TEST_float_arrays_opened':False,'NASA_labels_opened':False,
            'files':{p.relative_to(work).as_posix():core.sha(p) for p in out.glob('*.pt')}}
    core.write(receipt,record); runtime.checkpoint(work,'real_epoch_and_exact_restore',force=True)
    return record

def workload(raw,meta,paths,device):
    import torch_geometric  # Import-time scalar checks must run outside meta mode.
    records=[]
    for mission in ['MSL','SMAP']:
        channels=sorted(ch for ch in meta if meta[ch]['mission']==mission); nf=meta[channels[0]]['columns']
        for role in train.NEW_ROLES+['gdn','catch']:
            # Shapes/parameter counts only; allocate no large model on host/GPU.
            with torch.device('meta'): model,cfg=train.model_for(role,mission,channels,nf,paths,device)
            parameters=sum(p.numel() for p in model.parameters()); del model
            fit=data.Windows(raw,meta,channels,role,'fit',cfg.get('balanced_FIT_windows_cap_per_channel'))
            stop=data.Windows(raw,meta,channels,role,'stop',cfg.get('STOP_windows_cap_per_channel'))
            records.append({'mission':mission,'role':role,'parameters':parameters,
                'dense_FP32_parameter_gradient_Adam_estimate_bytes':16*parameters,
                'FIT_windows':len(fit),'FIT_batches':math.ceil(len(fit)/cfg['batch_size']),
                'STOP_windows':len(stop),'STOP_batches':math.ceil(len(stop)/cfg['batch_size']),
                'batch_size':cfg['batch_size'],'epoch_cap':cfg['epochs']})
    return records

def probe(work, raw, meta, paths, mission, role, device):
    """One native full batch; next update is replayed exactly, including mask grads.

    Private engineering model, discarded afterwards. These updates are NOT fits.
    The CATCH second update exercises its original pre-backward mask step with
    carried gradients; no altered schedule is used in scientific training.
    """
    channels=sorted(ch for ch in meta if meta[ch]['mission']==mission); nf=meta[channels[0]]['columns']
    core.seeds(17)
    with runtime.operation('probe_model_initialisation',mission=mission,role=role,deadline=180):
        model,cfg=train.model_for(role,mission,channels,nf,paths,device)
        if role in train.NEW_ROLES:
            active=np.stack([np.ptp(raw[ch][:meta[ch]['fit_end'],1:],axis=0)>0 for ch in channels])
            model.active_commands.copy_(torch.from_numpy(active))
        model.to(device)
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    scaler=data.external_scaler(raw,meta,channels) if role in ['gdn','catch'] else None
    z=data.transform({ch:raw[ch] for ch in channels},scaler)
    ds=data.Windows(z,meta,channels,role,'fit',cfg.get('balanced_FIT_windows_cap_per_channel'))
    # Interleaved FIT windows avoid accidentally testing only a constant stream.
    indices=np.linspace(0,len(ds)-1,cfg['batch_size']).astype(int).tolist()
    with runtime.operation('probe_data_preparation'): item=tuple(t.to(device) for t in gather(ds,indices))
    schedulers=[]; mask=None; c=cfg.get('config'); aux=None
    if role=='catch':
        optimizer=torch.optim.Adam([p for n,p in model.named_parameters() if 'mask_generator' not in n],lr=c['lr'],foreach=False,fused=False)
        mask=torch.optim.Adam(model.mask_generator.parameters(),lr=c['Mlr'],foreach=False,fused=False)
        for op,lr in [(optimizer,c['lr']),(mask,c['Mlr'])]:
            schedulers.append(torch.optim.lr_scheduler.OneCycleLR(op,max_lr=lr,epochs=c['num_epochs'],steps_per_epoch=math.ceil(len(ds)/cfg['batch_size']),pct_start=c['pct_start']))
        aux,crit=upstream.catch_losses(paths,c)
    else: optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'],foreach=False,fused=False)
    def capture():
        return {'model':cpu_copy(model.state_dict()),'optimizer':cpu_copy(optimizer.state_dict()),
                'mask_optimizer':None if mask is None else cpu_copy(mask.state_dict()),
                'mask_grads':[] if mask is None else [None if p.grad is None else cpu_copy(p.grad) for p in model.mask_generator.parameters()],
                'schedulers':[v.state_dict() for v in schedulers],'rng':core.rng()}
    def restore(state):
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        if mask is not None:
            mask.load_state_dict(state['mask_optimizer'])
            for p,g in zip(model.mask_generator.parameters(),state['mask_grads']): p.grad=None if g is None else g.to(device)
        for scheduler,s in zip(schedulers,state['schedulers']): scheduler.load_state_dict(s)
        core.restore_rng(state['rng'])
    def update(mask_boundary=False):
        optimizer.zero_grad(set_to_none=True)
        with runtime.operation('probe_forward_and_loss',mission=mission,role=role):
            if role in train.NEW_ROLES:
                x,e,y=item; ref,l=model(x,e); inj,truth=data.inject_batch(x); _,li=model(inj,e)
                loss=F.huber_loss(ref,y)+.5*(F.binary_cross_entropy_with_logits(l,torch.zeros_like(l))+F.binary_cross_entropy_with_logits(li,truth))
            elif role=='gdn': x,y=item; loss=F.mse_loss(model(x,None),y)
            else:
                x=item[0]; out,complex_out,dc=model(x); normalized=model.revin_layer(x,'transform')
                loss=F.mse_loss(out,x)+c['dc_lambda']*dc+c['auxi_lambda']*aux(complex_out,normalized)
                if mask_boundary: mask.step(); mask.zero_grad(set_to_none=True)
        value=float(loss.detach())
        if not math.isfinite(value): raise FloatingPointError('Full-batch engineering loss is nonfinite')
        with runtime.operation('probe_backward',mission=mission,role=role): loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError('Full-batch gradient is nonfinite')
        with runtime.operation('probe_optimizer_step',mission=mission,role=role): optimizer.step()
        return value
    scratch=work.parent/'E03_probe_scratch'; scratch.mkdir(exist_ok=True)
    paths0=[scratch/('resume_'+mission+'_'+role+'.pt'),scratch/('expected_'+mission+'_'+role+'.pt')]
    try:
        model.train(); start=time.monotonic(); update(); warm=time.monotonic()-start
        with runtime.operation('probe_checkpoint_write',deadline=600): save_torch(paths0[0],capture())
        start=time.monotonic(); expected_loss=update(mask_boundary=role=='catch'); measured=time.monotonic()-start
        with runtime.operation('probe_expected_checkpoint',deadline=600): save_torch(paths0[1],capture())
        with runtime.operation('probe_checkpoint_restore',deadline=600):
            state=load_torch(paths0[0]); restore(state); del state
        actual_loss=update(mask_boundary=role=='catch')
        with runtime.operation('probe_resume_comparison',deadline=600):
            expected=load_torch(paths0[1]); actual=capture()
            if expected_loss!=actual_loss or not exact(expected,actual): raise RuntimeError('Full-batch restored update differs')
            del expected,actual
        model.eval(); start=time.monotonic()
        with torch.no_grad(),runtime.operation('probe_STOP_objective'):
            if role in train.NEW_ROLES:
                x,e,y=item; ref,l=model(x,e); inj,truth=data.inject_batch(x); _,li=model(inj,e)
                validation=.5*(F.binary_cross_entropy_with_logits(l,torch.zeros_like(l))+F.binary_cross_entropy_with_logits(li,truth))+.1*F.huber_loss(ref,y)
            elif role=='gdn': x,y=item; validation=F.mse_loss(model(x,None),y)
            else: x=item[0]; out,_,_=model(x); validation=F.mse_loss(out,x)
            if not torch.isfinite(validation): raise FloatingPointError('Full-batch STOP objective nonfinite')
        validation_seconds=time.monotonic()-start
        native_record={}
        if role=='catch':
            model.eval()
            with torch.no_grad(),runtime.operation('probe_native_score'):
                x=item[0]; rec,_,_=model(x)
                score=((x-rec).square().mean(-1)+c['score_lambda']*crit(x,rec).mean(-1))[:,-1]
                if not torch.isfinite(score).all(): raise FloatingPointError('Native score nonfinite on representative FIT input')
                native_record['native_score_finite']=True
                native_record['native_score_batch_size']=len(x)
                for sparse in [False,True]:
                    edge=torch.zeros(2,192,nf,device=device)
                    if sparse: edge[:,:,0]=torch.sin(torch.arange(192,device=device)/13.); edge[:,::17,1]=1.
                    rec,_,_=model(edge)
                    v=((edge-rec).square().mean(-1)+c['score_lambda']*crit(edge,rec).mean(-1))[:,-1]
                    native_record['sparse_finite' if sparse else 'constant_finite']=bool(torch.isfinite(v).all())
        peak=torch.cuda.max_memory_allocated() if device=='cuda' else None
        reserved=torch.cuda.max_memory_reserved() if device=='cuda' else None
        total=torch.cuda.get_device_properties(0).total_memory if device=='cuda' else None
        if total and reserved>total*.95: raise RuntimeError('Less than 5% observed GPU memory margin; preserve settings and stop for capacity review')
        return {'status':'PASS','mission':mission,'role':role,'batch_size':cfg['batch_size'],
                'configuration':cfg,'first_update_seconds':warm,'steady_update_seconds':measured,
                'validation_batch_seconds':validation_seconds,
                'checkpoint_replay_exact':True,'CATCH_carried_gradient_mask_step_tested':role=='catch',
                'peak_allocated_bytes':peak,'peak_reserved_bytes':reserved,'GPU_total_bytes':total,
                'memory_measurement_includes_optimizer_allocation_and_restore':True,
                'projection_excludes_scoring_backup_contention_and_unknown_early_stopping':True,
                **native_record}
    finally:
        for p in paths0: p.unlink(missing_ok=True)

def gdn_equations(paths,device):
    """Retain original official GDN forward/gradient equation check, both missions."""
    from engineering import gdn_reference
    records=[]
    for mission,nf in [('MSL',55),('SMAP',25)]:
        core.seeds(17); model,_=upstream.factory('gdn',mission,nf,paths,device); model.to(device).eval(); layer=model.gnn_layers[0].gnn
        with torch.no_grad(): layer.att_em_i.uniform_(-.2,.2); layer.att_em_j.uniform_(-.2,.2)
        x=torch.randn(nf,5,device=device,requires_grad=True); emb=model.embedding.weight.detach().clone().requires_grad_(True)
        y=layer(x,model.edge_index_sets[0].to(device),emb); y.square().sum().backward(); grad=x.grad.clone(); egrad=emb.grad.clone()
        pgrad={k:p.grad.clone() for k,p in layer.named_parameters() if p.grad is not None}; layer.zero_grad()
        x2=x.detach().clone().requires_grad_(True); e2=emb.detach().clone().requires_grad_(True)
        target=gdn_reference(layer,x2,e2,model.edge_index_sets[0].to(device)); target.square().sum().backward()
        torch.testing.assert_close(y,target,rtol=1e-5,atol=3e-6); torch.testing.assert_close(grad,x2.grad,rtol=1e-5,atol=3e-6)
        torch.testing.assert_close(egrad,e2.grad,rtol=3e-5,atol=5e-6)
        for k,p in layer.named_parameters():
            if k in pgrad: torch.testing.assert_close(pgrad[k],p.grad,rtol=3e-5,atol=5e-6)
        records.append({'mission':mission,'forward_and_gradients':'PASS'})
    return records

def hardware_gate(work,raw,meta,paths,device,identity):
    receipt=work/'ENGINEERING_GATE.json'
    if receipt.exists():
        g=core.read(receipt)
        if g['status']!='PASS' or g['environment']!=identity['environment'] or g['package_manifest_sha256']!=identity['package_manifest_sha256']:
            raise RuntimeError('Hardware gate identity differs')
        for n,h in g['evidence'].items(): core.checked(work/n,h)
        print('REUSE VERIFIED FULL-BATCH HARDWARE GATE',flush=True); return g
    out=work/'engineering/hardware'; out.mkdir(parents=True,exist_ok=True)
    workloads=workload(raw,meta,paths,device); core.write(out/'workloads.json',workloads)
    records=[]
    for row in workloads:
        mission,role=row['mission'],row['role']; path=out/(mission+'__'+role+'.json')
        if path.exists():
            r=core.read(path)
            if r['identity']!=identity or r['status']!='PASS' or r['batch_size']!=row['batch_size']:
                raise RuntimeError('Partial hardware receipt differs')
            records.append(r); print('REUSE FULL-BATCH CHECK',mission,role,flush=True); continue
        gc.collect()
        if device=='cuda': torch.cuda.empty_cache()
        print('FULL-BATCH CHECK',mission,role,'batch',row['batch_size'],'parameters',row['parameters'],flush=True)
        numerical=None
        if role in train.NEW_ROLES:
            from convolution_regression import validate_role
            with runtime.operation('native_corrected_full_batch_regression',mission=mission,role=role,deadline=600):
                numerical=validate_role(raw,meta,paths,mission,role,device)
            print('NATIVE/CORRECTED FP32 + EXACT RNG/ADAM REPLAY PASS',mission,role,flush=True)
        if mission=='MSL' and role=='graph':
            smoke_record=core.read(work/'SMOKE_GATE.json')
            if device=='cuda' and smoke_record['measured_peak_GPU_reserved_bytes']>torch.cuda.get_device_properties(0).total_memory*.95:
                raise RuntimeError('Real-epoch smoke lacks 5% measured memory margin')
            r={'status':'PASS','mission':mission,'role':role,'batch_size':row['batch_size'],
               'full_real_epoch_reused':True,'checkpoint_replay_exact':True,
               'peak_allocated_bytes':smoke_record['measured_peak_GPU_allocated_bytes'],
               'peak_reserved_bytes':smoke_record['measured_peak_GPU_reserved_bytes'],
               'epoch_cap_training_seconds_projection':smoke_record['epoch1']['seconds']*row['epoch_cap']}
        else:
            with runtime.operation('full_batch_check',mission=mission,role=role,deadline=600):
                r=probe(work,raw,meta,paths,mission,role,device)
            r['epoch_cap_training_seconds_projection']=(r['steady_update_seconds']*row['FIT_batches']+r['validation_batch_seconds']*row['STOP_batches'])*row['epoch_cap']
        if numerical is not None:r['convolution_regression']=numerical
        r['identity']=identity
        core.write(path,r); records.append(r); runtime.checkpoint(work,'hardware_'+mission+'_'+role,force=True)
    with runtime.operation('official_GDN_equation_parity',deadline=180): equations=gdn_equations(paths,device)
    result={'status':'PASS','environment':identity['environment'],'package_manifest_sha256':identity['package_manifest_sha256'],
            'full_GPU_validation':device=='cuda','actual_GPU_resume_validation':'all ten native full batches; checkpoint replay exact',
            'official_source_validation':equations,'full_batches':records,
            'real_first_epoch_verified':core.sha(work/'SMOKE_GATE.json'),
            'NASA_labels_opened':False,'NASA_TEST_float_arrays_opened':False,
            'evidence':{p.relative_to(work).as_posix():core.sha(p) for p in out.glob('*.json')}}
    core.write(receipt,result); runtime.checkpoint(work,'all_full_batch_checks',force=True); return result

def verify_hardware_gate(work,identity):
    g=core.read(work/'ENGINEERING_GATE.json')
    if g['status']!='PASS' or g['environment']!=identity['environment'] or g['package_manifest_sha256']!=identity['package_manifest_sha256']:
        raise RuntimeError('Hardware gate source/runtime identity differs')
    expected={(m,r) for m in ['MSL','SMAP'] for r in train.NEW_ROLES+['gdn','catch']}
    records=g['full_batches']
    if len(records)!=10 or {(r['mission'],r['role']) for r in records}!=expected:
        raise RuntimeError('All ten full-batch checks are required')
    for r in records:
        batch=256 if r['role'] in train.NEW_ROLES else 32 if r['role']=='gdn' else 128
        if r['identity']!=identity or r['status']!='PASS' or r['batch_size']!=batch or not r['checkpoint_replay_exact']:
            raise RuntimeError('Full-batch scientific settings or resume check differ')
        if r['role'] in train.NEW_ROLES:
            numerical=r.get('convolution_regression',{})
            if numerical.get('status')!='PASS' or numerical.get('batch_size')!=256 or not numerical.get('corrected_replay_bitwise_equal') or identity['environment']['device']=='cuda' and not numerical.get('actual_GPU'):
                raise RuntimeError('Corrected-role numerical and exact replay regression required')
    for n,h in g['evidence'].items():core.checked(work/n,h)
    core.checked(work/'SMOKE_GATE.json',g['real_first_epoch_verified'])
    if identity['environment']['device']=='cuda' and not g['full_GPU_validation']:
        raise RuntimeError('Actual GPU validation required')
    return g
