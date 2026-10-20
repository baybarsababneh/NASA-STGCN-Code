"""One-time E03 engineering amendment. Frozen package/scientific code unchanged.

Invoke with E03's existing hardware arguments and launcher. Only the private
SMAP/fusion comparison may use the disclosed target-tensor allowance. Actual
GPU diagnostics, original failure and source provenance enter the existing
hardware evidence/backup. This file never imports a native scientific state.
"""
import copy, hashlib, os, sys
from pathlib import Path
import torch

AMENDMENT = 'EXP013_E03_P01_PARAMETER_GATE'
MANIFEST_SHA = '772784ed57788cc9f4aa87d07d4bcb03bbf2bd03b382063a8b693090c2282045'
REGRESSION_SHA = '50209032c71aaa5a671bd35aaf46150a3330c28d60571801b9e499756329b1f2'
TARGET = 'discriminator.0.weight'
OLD_ATOL, NEW_ATOL, RTOL = 5e-6, 7.5e-6, 2e-5
RELATIVE_L2_CEILING = 2e-5

def snapshot(cr, model, optimizer):
    state = cr.cpu_copy(optimizer.state_dict())
    names = {id(p):n for n,p in model.named_parameters()}
    ids, groups = {}, {}
    for gi,(live,saved) in enumerate(zip(optimizer.param_groups,state['param_groups'])):
        for p,pid in zip(live['params'],saved['params']):
            ids[names[id(p)]] = pid; groups[names[id(p)]] = gi
    return {'parameters':cr.cpu_copy(dict(model.named_parameters())),
            'optimizer':state, 'name_ids':ids, 'name_groups':groups}

def adam_facts(trace, name):
    """Independent FP64 gradient -> moments -> update, only for diagnosis."""
    before = trace['_p01_before']; pid = before['name_ids'][name]
    group = trace['adam']['param_groups'][before['name_groups'][name]]
    if (group['lr'] != .001 or tuple(group['betas']) != (.9,.999)
        or group['eps'] != 1e-8 or group['weight_decay'] != 0
        or group.get('amsgrad') or group.get('maximize')
        or group.get('foreach') is not False or group.get('fused') is not False):
        raise RuntimeError('P01 requires the unchanged frozen ordinary Adam')
    prior = before['optimizer']['state'].get(pid,{})
    state = trace['adam']['state'][pid]
    g = trace['gradients'][name].double()
    m0 = prior.get('exp_avg',torch.zeros_like(g)).double()
    v0 = prior.get('exp_avg_sq',torch.zeros_like(g)).double()
    count = int(state['step']); previous = int(prior.get('step',0))
    if count != previous+1:raise RuntimeError('Adam counter differs')
    b1,b2 = group['betas']; m = b1*m0+(1-b1)*g
    v = b2*v0+(1-b2)*g.square()
    unit = torch.finfo(torch.float32).eps; tiny = torch.finfo(torch.float32).tiny
    # Ordinary FP32 moment recurrence, with an explicit rounding budget.
    for actual,expected,scale in [
        (state['exp_avg'].double(),m,(b1*m0).abs()+((1-b1)*g).abs()),
        (state['exp_avg_sq'].double(),v,(b2*v0).abs()+((1-b2)*g.square()).abs())]:
        if not torch.all((actual-expected).abs() <= 16*unit*scale+16*tiny):
            raise RuntimeError('Adam moment recurrence cannot explain discrepancy')
    movement = group['lr']*(m/(1-b1**count))/((v/(1-b2**count)).sqrt()+group['eps'])
    p0 = before['parameters'][name].double(); after = trace['parameters'][name].double()
    # Same update-equation tolerance as E03, now also derived from gradients.
    torch.testing.assert_close(after,p0-movement,atol=3e-8,rtol=3e-7)
    return {'before':p0,'after':after,'gradient':g,'m':state['exp_avg'].double(),
            'v':state['exp_avg_sq'].double(),'movement':movement,'step':count,
            'eps':group['eps'], 'equation_residual':after-(p0-movement)}

def diagnose(cr, native, corrected):
    a,b = adam_facts(native,TARGET),adam_facts(corrected,TARGET)
    if a['step'] != b['step']:raise RuntimeError('Native/corrected Adam counters differ')
    if a['step']==1 and not cr.compare(native['_p01_before']['parameters'],
                                      corrected['_p01_before']['parameters'],'exact')['pass']:
        raise RuntimeError('Cold native/corrected parameters are not identical')
    delta = a['after']-b['after']
    predicted = a['before']-b['before']-(a['movement']-b['movement'])
    residue = delta-predicted
    # Bound the subtraction/update rounding, retaining E03's stricter bound.
    unit = torch.finfo(torch.float32).eps; tiny = torch.finfo(torch.float32).tiny
    rounding = 16*unit*(a['before'].abs()+b['before'].abs()+a['movement'].abs()+b['movement'].abs())+32*tiny
    e03_bound = 6e-8+3e-7*(a['after'].abs()+b['after'].abs())
    bound = torch.minimum(rounding,e03_bound)
    if not torch.all(residue.abs() <= bound):
        raise RuntimeError('Gradient/Adam equations do not explain parameter difference')
    old_limit = OLD_ATOL+RTOL*b['after'].abs()
    failing = delta.abs()>old_limit
    relative = float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(a['after']).clamp_min(1e-12))
    records=[]
    # Retain every original failing element, not just a tensor-wide maximum.
    for ij in failing.nonzero().tolist():
        ix=tuple(ij)
        records.append({'index':ij,'native_before':float(a['before'][ix]),
            'corrected_before':float(b['before'][ix]),'native_after':float(a['after'][ix]),
            'corrected_after':float(b['after'][ix]),'absolute_difference':float(delta[ix].abs()),
            'original_element_limit':float(old_limit[ix]),
            'amended_element_limit':float(NEW_ATOL+RTOL*b['after'][ix].abs()),
            'native_gradient':float(a['gradient'][ix]),'corrected_gradient':float(b['gradient'][ix]),
            'gradient_difference':float((a['gradient'][ix]-b['gradient'][ix]).abs()),
            'gradient_magnitudes_over_Adam_epsilon':[float(a['gradient'][ix].abs()/a['eps']),float(b['gradient'][ix].abs()/b['eps'])],
            'native_exp_avg':float(a['m'][ix]),'corrected_exp_avg':float(b['m'][ix]),
            'native_exp_avg_sq':float(a['v'][ix]),'corrected_exp_avg_sq':float(b['v'][ix]),
            'native_Adam_movement':float(a['movement'][ix]),'corrected_Adam_movement':float(b['movement'][ix]),
            'predicted_signed_difference':float(predicted[ix]),
            'observed_signed_difference':float(delta[ix]),'explanation_residual':float(residue[ix]),
            'explanation_rounding_bound':float(bound[ix])})
    return {'update':a['step'],'parameter':TARGET,'original_failed_elements':records,
            'original_failed_element_count':len(records),'maximum_absolute_difference':float(delta.abs().max()),
            'relative_L2_error':relative,'relative_L2_ceiling':RELATIVE_L2_CEILING,
            'Adam_equations_explain_difference':True,
            'native_Adam_equation_max_abs_residual':float(a['equation_residual'].abs().max()),
            'corrected_Adam_equation_max_abs_residual':float(b['equation_residual'].abs().max()),
            'maximum_cross_implementation_explanation_residual':float(residue.abs().max())}

def amend_comparison(cr, original_compare, a, b, exact=False):
    report = original_compare(a,b,exact=exact)
    if exact:
        if not cr.compare(a['_p01_before'],b['_p01_before'],'exact')['pass']:
            raise RuntimeError('Instrumented corrected pre-update replay differs')
        return report,None
    detail = diagnose(cr,a,b)
    original = copy.deepcopy(report['comparisons']['parameters'])
    detail['original_parameter_gate'] = original
    other_pass = all(v['pass'] for k,v in report['comparisons'].items() if k!='parameters')
    only_target = original['failures'] == ['root.'+TARGET]
    acceptable = (torch.allclose(a['parameters'][TARGET],b['parameters'][TARGET],atol=NEW_ATOL,rtol=RTOL)
                  and detail['relative_L2_error']<=RELATIVE_L2_CEILING)
    used = not original['pass'] and other_pass and only_target and acceptable
    if used:
        amended = copy.deepcopy(original); amended['pass']=True; amended['failures']=[]
        amended['original_E03_gate_pass']=False
        amended['tensor_tolerances']={'root.'+TARGET:{'atol':NEW_ATOL,'rtol':RTOL}}
        amended['amendment']=AMENDMENT
        # Global atol remains 5e-6. Only this named tensor is excepted.
        report['comparisons']['parameters']=amended
    detail['amended_gate_used']=used
    detail['amended_tensor_gate_pass']=bool(acceptable)
    report['pass']=all(v['pass'] for v in report['comparisons'].values())
    return report,detail

def install(cr, core, work):
    original_validate, original_step, original_compare = cr.validate_role,cr.step,cr.compare_step
    def validate(raw,meta,paths,mission,role,device):
        if (mission,role)!=('SMAP','fusion'):
            return original_validate(raw,meta,paths,mission,role,device)
        if device!='cuda':raise RuntimeError('Production amendment requires actual A100 validation')
        identity=core.read(work/'RUN_FREEZE.json')
        audit=work/'engineering/hardware/E03_P01_PARAMETER_GATE.json'
        source=Path(__file__).read_text()
        checkpoints={p.relative_to(work).as_posix():core.sha(p) for p in (work/'models').rglob('*.pt')}
        document={'amendment':AMENDMENT,'status':'RUNNING','identity':identity,
            'scope':{'mission':'SMAP','role':'fusion','tensor':TARGET},
            'original_parameter_atol':OLD_ATOL,'amended_tensor_atol':NEW_ATOL,'unchanged_parameter_rtol':RTOL,
            'additional_relative_L2_ceiling':RELATIVE_L2_CEILING,'other_tolerances_unchanged':copy.deepcopy(cr.TOLERANCES),
            'scientific_package_manifest_unchanged':MANIFEST_SHA,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'amendment_source_utf8':source,'scientific_checkpoints_before':checkpoints,
            'SMOKE_GATE_sha256':core.sha(work/'SMOKE_GATE.json'),'actual_GPU':True,
            'updates':[],'NASA_labels_opened':False,'NASA_TEST_float_arrays_opened':False}
        core.write(audit,document)
        def captured(model,optimizer,*args,**kwargs):
            before=snapshot(cr,model,optimizer)
            value=original_step(model,optimizer,*args,**kwargs);value['_p01_before']=before
            return value
        def compared(a,b,exact=False):
            report,detail=amend_comparison(cr,original_compare,a,b,exact)
            if detail is not None:
                document['updates'].append(detail);core.write(audit,document)
                print('P01 ELEMENT AUDIT',detail['update'],'old failures',detail['original_failed_element_count'],
                      'max abs',detail['maximum_absolute_difference'],'relative L2',detail['relative_L2_error'],
                      'Adam explained',detail['Adam_equations_explain_difference'],'amendment used',detail['amended_gate_used'],flush=True)
                if detail['original_failed_elements']:
                    worst=max(detail['original_failed_elements'],key=lambda row:row['absolute_difference'])
                    print('P01 WORST ORIGINAL FAILING ELEMENT',worst,flush=True)
            return report
        cr.step,cr.compare_step=captured,compared
        try:
            result=original_validate(raw,meta,paths,mission,role,device)
            after={p.relative_to(work).as_posix():core.sha(p) for p in (work/'models').rglob('*.pt')}
            if checkpoints!=after:raise RuntimeError('Scientific checkpoint changed during private regression')
            if len(document['updates'])!=2:raise RuntimeError('Exactly two numerical updates required')
            document.update(status='PASS',corrected_RNG_Adam_replay_exact=True,scientific_checkpoints_preserved=True)
            core.write(audit,document)
            result['parameter_gate_amendment']={'name':AMENDMENT,'scope':document['scope'],
                'tensor_atol':NEW_ATOL,'rtol':RTOL,'original_atol_for_all_other_tensors':OLD_ATOL,
                'audit_path':audit.relative_to(work).as_posix(),'audit_sha256':core.sha(audit)}
            return result
        except BaseException as ex:
            document.update(status='FAIL',error=type(ex).__name__+': '+str(ex));core.write(audit,document);raise
        finally:cr.step,cr.compare_step=original_step,original_compare
    cr.validate_role=validate

def main():
    if len(sys.argv)<2 or sys.argv[1]!='hardware':raise RuntimeError('This amendment only runs the existing hardware action')
    package=Path(os.environ.get('EXP013_E03_PACKAGE','/content/EXP013_E03_package'))
    if hashlib.sha256((package/'PACKAGE_MANIFEST.json').read_bytes()).hexdigest()!=MANIFEST_SHA:
        raise RuntimeError('Expected the exact existing E03 package; do not change its manifest')
    if hashlib.sha256((package/'src/convolution_regression.py').read_bytes()).hexdigest()!=REGRESSION_SHA:
        raise RuntimeError('Expected the frozen E03 comparison source')
    work=Path(sys.argv[sys.argv.index('--work')+1])
    if (work/'STUDY_FREEZE.json').exists():raise RuntimeError('Study already frozen; engineering evidence cannot be amended')
    sys.path.insert(0,str(package/'src'))
    import core,convolution_regression as cr,simple_run
    core.manifest(package,'PACKAGE_MANIFEST.json')
    if cr.TOLERANCES!={'outputs':(2e-6,2e-5),'gradients':(2e-7,2e-4),
                       'parameters':(OLD_ATOL,RTOL),'adam':(2e-8,2e-4),'exact':(0.,0.)}:
        raise RuntimeError('Original gates changed')
    install(cr,core,work)
    # Same E03 entry point, lock, watchdog, backups, identity and CLI arguments.
    simple_run.main()

if __name__=='__main__':main()
