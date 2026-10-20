"""Fixed, disclosed FP32 equivalence tolerances; exact RNG/replay checks."""
import copy
import importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F

# These are gates, never adjusted by the benchmark to make a candidate pass.
TOLERANCES = {
    'outputs': (2e-6, 2e-5),
    'gradients': (2e-7, 2e-4),
    # Adam's epsilon makes near-zero gradients sensitive to FP32 reduction
    # order. E03 discloses a 5e-6 parameter gate (0.5% of lr=.001), together
    # with the unchanged output/gradient gates and a separate
    # independent FP64 check of every Adam update below. This is not bitwise
    # equivalence and never authorises native-checkpoint continuation.
    'parameters': (5e-6, 2e-5),
    # First-moment rounding induced by the gradient absolute gate is
    # (1-beta1)*2e-7 = 2e-8. Counters remain exact; relative gate unchanged.
    'adam': (2e-8, 2e-4),
    'exact': (0., 0.),
}

def module(path, name):
    spec=importlib.util.spec_from_file_location(name,Path(path))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

def cpu_copy(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_copy(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)

def compare(a,b,kind='exact'):
    atol,rtol=TOLERANCES[kind]
    report={'pass':True,'bitwise_equal':True,'atol':atol,'rtol':rtol,
            'max_absolute_error':0.,'max_relative_L2_error':0.,'failures':[],
            'tensor_errors':{}}
    def walk(x,y,name):
        if isinstance(x,torch.Tensor):
            if not isinstance(y,torch.Tensor) or x.shape!=y.shape or x.dtype!=y.dtype:
                report['pass']=False;report['bitwise_equal']=False;report['failures'].append(name+': tensor metadata');return
            exact=torch.equal(x,y);report['bitwise_equal'] &= exact
            if x.is_floating_point():
                xf=x.double();yf=y.double();delta=(xf-yf).abs()
                absolute=float(delta.max()) if x.numel() else 0.
                relative=float(torch.linalg.vector_norm(xf-yf)/torch.linalg.vector_norm(xf).clamp_min(1e-12))
                finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                passed=finite and torch.allclose(x,y,atol=atol,rtol=rtol)
                # Adam counters must be exact even under an FP32 moment gate.
                if name.endswith('.step'):passed=exact
                report['max_absolute_error']=max(report['max_absolute_error'],absolute)
                report['max_relative_L2_error']=max(report['max_relative_L2_error'],relative)
                report['tensor_errors'][name]={'absolute':absolute,'relative_L2':relative,'bitwise_equal':exact}
            else: passed=exact
            if not passed:report['pass']=False;report['failures'].append(name)
        elif isinstance(x,dict):
            if not isinstance(y,dict) or x.keys()!=y.keys():
                report['pass']=False;report['bitwise_equal']=False;report['failures'].append(name+': keys');return
            for k in x:walk(x[k],y[k],name+'.'+str(k))
        elif isinstance(x,(list,tuple)):
            if type(x)!=type(y) or len(x)!=len(y):
                report['pass']=False;report['bitwise_equal']=False;report['failures'].append(name+': sequence');return
            for i,(xx,yy) in enumerate(zip(x,y)):walk(xx,yy,name+'.'+str(i))
        elif type(x)!=type(y) or x!=y:
            report['pass']=False;report['bitwise_equal']=False;report['failures'].append(name+': value')
    walk(a,b,'root')
    return report

def step(model,optimizer,item,inject_batch,rng,measure=None):
    """Verbatim frozen FIT equation/order, including the finite scalar gate."""
    import contextlib,math
    def span(name):return measure.section(name) if measure else contextlib.nullcontext()
    x,e,y=item;model.train();optimizer.zero_grad(set_to_none=True)
    with span('clean_forward'): ref,l=model(x,e)
    with span('synthetic_injection'):inj,truth=inject_batch(x)
    with span('injected_forward'):injected_ref,li=model(inj,e)
    with span('loss'):
        loss=F.huber_loss(ref,y)+.5*(F.binary_cross_entropy_with_logits(l,torch.zeros_like(l))+F.binary_cross_entropy_with_logits(li,truth))
        value=float(loss.detach())
        if not math.isfinite(value):raise FloatingPointError('Nonfinite frozen FIT objective')
    with span('backward'):loss.backward()
    gradients=cpu_copy({n:p.grad for n,p in model.named_parameters()})
    before=cpu_copy(dict(model.named_parameters()))
    with span('optimizer'):optimizer.step()
    # Check the optimiser equation independently in double precision; the
    # actual training and optimiser remain FP32 and are never replaced.
    adam_math_error=0.
    for group in optimizer.param_groups:
        if group['weight_decay'] or group.get('amsgrad') or group.get('maximize'):
            raise RuntimeError('Regression expects the frozen ordinary Adam')
        beta1,beta2=group['betas']
        names={id(p):n for n,p in model.named_parameters()}
        for p in group['params']:
            if p.grad is None:continue
            state=optimizer.state[p];count=int(state['step'])
            m=state['exp_avg'].detach().cpu().double()/(1-beta1**count)
            v=state['exp_avg_sq'].detach().cpu().double()/(1-beta2**count)
            expected=before[names[id(p)]].double()-group['lr']*m/(v.sqrt()+group['eps'])
            actual=p.detach().cpu().double()
            torch.testing.assert_close(actual,expected,atol=3e-8,rtol=3e-7)
            adam_math_error=max(adam_math_error,float((actual-expected).abs().max()))
    return {'outputs':cpu_copy((ref,l,injected_ref,li,loss)), 'injection':cpu_copy((inj,truth)),
            'gradients':gradients, 'parameters':cpu_copy(model.state_dict()),
            'adam':cpu_copy(optimizer.state_dict()), 'rng':cpu_copy(rng()),
            'independent_Adam_FP64_equation_max_absolute_error':adam_math_error}

def compare_step(a,b,exact=False):
    kinds={'outputs':'outputs','injection':'exact','gradients':'gradients',
           'parameters':'parameters','adam':'adam','rng':'exact'}
    reports={k:compare(a[k],b[k],'exact' if exact else kind) for k,kind in kinds.items()}
    return {'pass':all(v['pass'] for v in reports.values()),
            'bitwise_equal':all(v['bitwise_equal'] for v in reports.values()),'comparisons':reports}

def validate_role(raw,meta,paths,mission,role,device):
    """Two full frozen-batch updates versus native; two exact candidate replays.

    Private discarded models only. No labels, TEST, selection or checkpoints
    from an earlier run are used. Existing hardware gate owns persistence.
    """
    import gc,time,numpy as np
    import core,data,train
    from execution_data import gather
    native=module(core.ROOT/'frozen_originals/architectures.py','e03_native_reference')
    core.checked(core.ROOT/'frozen_originals/architectures.py',
        'de8c5697f92edb0f1bc35e875bcd29a0f9b0fe1dd02d2f3225dc86057b4e54e4')
    channels=sorted(ch for ch in meta if meta[ch]['mission']==mission)
    nf=meta[channels[0]]['columns'];saved=cpu_copy(core.rng())
    try:
        core.seeds(17)
        candidate,cfg=train.model_for(role,mission,channels,nf,paths,device)
        init_rng=cpu_copy(core.rng());initial=cpu_copy(candidate.state_dict());del candidate
        core.seeds(17);control=native.CommandReferenceGCN(nf,len(channels),role)
        if not compare(initial,control.state_dict())['pass'] or not compare(init_rng,core.rng())['pass']:
            raise RuntimeError('Convolution correction changed initial weights or RNG')
        del control
        active=np.stack([np.ptp(raw[ch][:meta[ch]['fit_end'],1:],axis=0)>0 for ch in channels])
        initial['active_commands']=torch.from_numpy(active)
        ds=data.Windows(raw,meta,channels,role,'fit',cfg['balanced_FIT_windows_cap_per_channel'])
        ix=np.linspace(0,len(ds)-1,cfg['batch_size']).astype(int).tolist()
        item=tuple(t.to(device) for t in gather(ds,ix));start=cpu_copy(core.rng())
        def trajectory(is_native):
            model=native.CommandReferenceGCN(nf,len(channels),role) if is_native else train.model_for(role,mission,channels,nf,paths,device)[0]
            model.load_state_dict(initial,strict=True);model.to(device)
            op=torch.optim.Adam(model.parameters(),lr=cfg['lr'],foreach=False,fused=False)
            core.restore_rng(start);trace=[];times=[]
            for update in range(2):
                print('CONV REGRESSION',mission,role,'native' if is_native else 'corrected','update',update+1,flush=True)
                if device=='cuda':torch.cuda.synchronize()
                began=time.monotonic();trace.append(step(model,op,item,data.inject_batch,core.rng))
                if device=='cuda':torch.cuda.synchronize()
                times.append(time.monotonic()-began)
                print('CONV REGRESSION UPDATE COMPLETE',round(times[-1],4),'seconds including numerical checks',flush=True)
            del model,op;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
            return trace,times
        control,old_seconds=trajectory(True);corrected,new_seconds=trajectory(False)
        numeric=[compare_step(a,b) for a,b in zip(control,corrected)]
        if not all(r['pass'] for r in numeric):
            raise RuntimeError('Full-batch native/corrected FP32 regression failed: '+str(numeric))
        replay,_=trajectory(False)
        deterministic=[compare_step(a,b,exact=True) for a,b in zip(corrected,replay)]
        if not all(r['pass'] for r in deterministic):raise RuntimeError('Corrected full-batch RNG/Adam replay is not exact')
        return {'status':'PASS','mission':mission,'role':role,'batch_size':cfg['batch_size'],
            'actual_GPU':device=='cuda','updates':2,'tolerances':TOLERANCES,
            'native_vs_corrected':numeric,'corrected_replay_bitwise_equal':True,
            'Adam_FP64_equation_errors':[[a['independent_Adam_FP64_equation_max_absolute_error'],b['independent_Adam_FP64_equation_max_absolute_error']] for a,b in zip(control,corrected)],
            'native_regression_inclusive_step_seconds':old_seconds,
            'corrected_regression_inclusive_step_seconds':new_seconds,
            'timings_include_snapshots_and_checks_not_training_throughput':True,
            'NASA_TEST_float_arrays_opened':False,'NASA_labels_opened':False}
    finally:core.restore_rng(saved)
