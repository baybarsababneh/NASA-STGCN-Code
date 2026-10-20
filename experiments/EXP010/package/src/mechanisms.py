"""Fixed telemetry perturbations and descriptive response accounting. No selection."""
import numpy as np
from detectors_exp009 import calibrate,apply,burden

def cases(n):
    start=int(.7*n);end=int(.8*n);assert end-start>=32
    result=[]
    for family in ('spike','offset32','ramp32'):
        duration=1 if family=='spike'else 32
        for amplitude in (.5,2.):
            for sign in (-1,1):
                for position in (1/3,2/3):
                    onset=start+int(np.floor(position*(end-start-duration)))
                    result.append(dict(id=f'{family}_a{amplitude}_s{sign}_p{position:.6f}',family=family,duration=duration,amplitude=amplitude,sign=sign,position=position,onset=onset,end=onset+duration))
    for position in (1/3,2/3):
        onset=start+int(np.floor(position*(end-start-32)))
        result.append(dict(id=f'stuck32_p{position:.6f}',family='stuck32',duration=32,amplitude=None,sign=None,position=position,onset=onset,end=onset+32))
    assert len(result)==26 and len({r['id']for r in result})==26
    return result

def perturb(raw,c):
    changed=raw.copy();a,b=c['onset'],c['end']
    if c['family']=='stuck32':changed[a:b,0]=raw[a-1,0]
    elif c['family']=='ramp32':changed[a:b,0]+=np.asarray(c['sign']*c['amplitude']*np.linspace(1/32,1,32),dtype=np.float32)
    else:changed[a:b,0]+=np.float32(c['sign']*c['amplitude'])
    assert np.isfinite(changed).all()
    assert np.array_equal(raw[:a],changed[:a]) and np.array_equal(raw[b:],changed[b:])
    assert np.array_equal(raw[:,1:],changed[:,1:])
    return changed

def policies():
    x=[dict(id=f'{f}_h{h}',family=f,h=h)for f in ('absolute','ewma10','multiscale')for h in (2,4,8,16)]
    return x+[dict(id='raw_q99',family='historical_raw99'),dict(id='ewma10_q995',family='historical_ewma995')]

def response(base,changed,c,n,model):
    t=base['target_index'];assert np.array_equal(t,changed['target_index'])
    col='persistence'if model=='persistence'else'prediction'
    e0=base['target'].astype(float)-base[col].astype(float);e1=changed['target'].astype(float)-changed[col].astype(float)
    dx=changed['target'].astype(float)-base['target'].astype(float);dp=changed[col].astype(float)-base[col].astype(float)
    assert np.array_equal(base[col][t<=c['onset']],changed[col][t<=c['onset']]),'Causality failure: changed prediction before it can see corruption'
    assert np.array_equal(base[col][t>=c['end']+64],changed[col][t>=c['end']+64]),'Unexpected persistent model state'
    event=(t>=c['onset'])&(t<c['end']);after=(t>=c['end'])&(t<min(c['end']+64,int(.8*n)))
    null=not np.any(dx[event]!=0);input_l1=float(np.abs(dx[event]).sum())
    gain=float(np.abs((e1-e0)[event]).sum()/input_l1)if input_l1 else None
    stats=dict(model=model,status='NULL_PERTURBATION'if null else'MEASURED',input_l1=input_l1,input_rms=float(np.sqrt(np.mean(dx[event]**2))),actual_residual_response_l1=float(np.abs((e1-e0)[event]).sum()),residual_only_response_l1=input_l1,response_ratio=gain,prediction_change_l1=float(np.abs(dp[event]).sum()),post_event_observed_samples=int(after.sum()),post_event_residual_change_l1=float(np.abs((e1-e0)[after]).sum()),max_observed_residual=float(np.abs(e1[event]).max()),causality_passed=True,memory_bound_checked_samples=int(np.sum(t>=c['end']+64)),memory_bound_passed=True if np.any(t>=c['end']+64)else None)
    diagnostics=[]
    if null:return stats,diagnostics
    ref=(t>=int(.6*n))&(t<int(.7*n));region=(t>=int(.7*n))&(t<int(.8*n));er=e0+dx
    for p in policies():
        cal=calibrate(e0,ref,p);_,f0=apply(e0,cal);_,f1=apply(e1,cal);_,fr=apply(er,cal)
        def hitdelay(f):
            hits=np.flatnonzero(f[event]);return bool(len(hits)),int(hits[0])if len(hits)else int(event.sum())
        bh,_=hitdelay(f0);ah,ad=hitdelay(f1);rh,rd=hitdelay(fr)
        diagnostics.append(dict(model=model,policy=p['id'],baseline_hit=bh,actual_hit=ah,residual_only_hit=rh,incremental_actual=ah and not bh,incremental_residual_only=rh and not bh,actual_delay_censored=ad,residual_only_delay_censored=rd,event_samples=int(event.sum()),actual_post_flags=int(f1[after].sum()),baseline_post_flags=int(f0[after].sum()),reference_threshold=float(cal['threshold']),reference_scale=float(cal['scale']),baseline_region_occupancy=burden(f0,region)['occupancy'],baseline_region_starts_per_1000=burden(f0,region)['starts_per_1000']))
    return stats,diagnostics

def nominal_context(base,raw,meta):
    """Descriptive causal command context; never conditions a threshold in EXP010."""
    r=meta['reference_result'];j=meta['job'];t=base['target_index'];n=r['num_values'];sources=r['graph_sources']
    change=np.r_[False,np.any(raw[1:][:,sources]!=raw[:-1][:,sources],axis=1)] if sources else np.zeros(len(raw),bool)
    cs=np.r_[0,np.cumsum(change.astype(int))];rows=[]
    models=['gcn'if j['role']=='fixed'else'message_removed']
    if j['role']=='fixed'and j['seed']==17:models.append('persistence')
    for model in models:
        e=base['target'].astype(float)-base['persistence'if model=='persistence'else'prediction'].astype(float)
        for phase,a,b in [('reference',.6,.7),('assessment',.7,.8)]:
            region=(t>=int(a*n))&(t<int(b*n))
            for lag in (0,1,8,32):
                active=(cs[t]-cs[t-lag])>0 if lag else np.zeros(len(t),bool)
                for state in ([False,True]if lag else [False]):
                    mask=region&(active==state);x=e[mask]
                    rows.append(dict(mission=j['mission'],channel=j['channel'],seed=0 if model=='persistence'else j['seed'],model=model,phase=phase,context_lag=lag,command_active=state,samples=len(x),mean_abs_residual=float(abs(x).mean())if len(x)else None,rms_residual=float(np.sqrt(np.mean(x*x)))if len(x)else None,zero_residual_fraction=float(np.mean(x==0))if len(x)else None))
    return rows
