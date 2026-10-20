"""Synthetic-only engineering gates. No project anomaly labels or model selection."""
import json,tempfile
from pathlib import Path
import numpy as np
import torch
from hundman_detector import detect
from detector_metrics_v2 import average_precision,event_metrics,higher_threshold
from integrity import normal,commit,read_commit,DataGuard,sha
from models_io import Windows,train_lstm,tensor_hash
from models_exp007 import MessageGatedGCN
from lstm_baseline import ControlledLSTM

def must_fail(fn):
    try:fn()
    except (ValueError,RuntimeError,PermissionError,FileNotFoundError):return
    raise AssertionError('Expected safe rejection')

def run_checks():
    passed=[]
    def ok(name):passed.append(name)
    gold=Path(__file__).resolve().parents[1]/'provenance/hundman_reference_golden.npz'
    with np.load(gold,allow_pickle=False) as z:
        for case in range(6):
            y=z[f'c{case}_target'];d=detect(y,z[f'c{case}_prediction'],np.arange(250,len(y)+250))
            assert np.array_equal(d['flags'],z[f'c{case}_flags'])
            actual=np.array([[w['start']-250,w['end_exclusive']-250,w['epsilon'],w['epsilon_inverse'],w['z'],w['z_inverse']] for w in d['windows']])
            assert np.allclose(actual,z[f'c{case}_window_values'],atol=1e-12,rtol=1e-12)
            ok(f'upstream_parity_synthetic_case{case}')
    for n in [1,69,70,71,139,140,249,251,1801,2101,2171,2501]:
        y=np.sin(np.arange(n)*.01);pred=y-.3;d=detect(y,pred,np.arange(250,250+n))
        assert d['flags'].shape==(n,) and d['smoothed'].shape==(n,)
        assert np.isfinite(d['smoothed']).all()
        if n>=70:assert d['windows'][-1]['end_exclusive']==250+n
        else:assert d['status']=='NO_OUTPUT_SHORT_STREAM' and not d['flags'].any()
        assert np.all(d['issue_index'][d['flags']]>=np.arange(250,250+n)[d['flags']])
    ok('short_degenerate_tail_coverage_and_issue_timing')
    must_fail(lambda:detect([1,2],[1,2],[1,3]));must_fail(lambda:detect([np.nan],[1],[0]));ok('detector_invalid_input_rejected')
    for y in ([0,1],[1,0]):assert average_precision(y,[.7,.7])==.5
    assert average_precision([1,1],[2,1])==1 and np.isnan(average_precision([0,0],[2,1]))
    ok('AP_ties_permutation_and_undefined_no_positives')
    a=event_metrics([0,1,1,0,0,1,0],[1]*7)
    assert a['H']==2 and a['U']==0 and a['corrected_event_f0_5']==0
    assert event_metrics([0,1,0],[0,0,0])['corrected_event_f0_5']==0
    ok('event_all_alarm_correction_and_no_alarm')
    from benchmark_metrics import actual_delays,pooled_events
    delay=actual_delays([1,1,0],[1,0,0],[300,-1,-1],[250,251,252],original_events=[(200,252)])
    assert delay[0]['delay']==100 and not delay[0]['within_event_notification']
    assert pooled_events([event_metrics([0,1,0],[0,0,0])])['corrected_event_f0_5']==0
    ok('notification_delay_uses_original_onset_and_pooled_no_alarm_Fbeta')
    q=higher_threshold([0,0,0,1],.99);assert q['threshold']==1 and q['at_sample_max'];ok('higher_quantile_ties_strict_comparison')
    assert normal({'x':np.int64(3),'z':np.bool_(True)})=={'x':3,'z':True}
    must_fail(lambda:normal(float('inf')));ok('numpy_JSON_scalars_and_nonfinite_rejection')
    raw=np.arange(240,dtype=np.float32).reshape(120,2);x,y=Windows(raw,[0,1],85,110,64)[0]
    assert x.shape==(2,64) and x[0,-1]==raw[84,0] and y==raw[85,0]
    copy=raw.copy();copy[85:]=999;xx,_=Windows(copy,[0,1],85,110,64)[0];assert torch.equal(x,xx)
    ok('window_orientation_target_exclusion_future_input_invariance')
    with torch.no_grad():
        m=MessageGatedGCN(2,4,np.array([[0,1],[0,0]],np.float32),3,(1,2),'fixed',True)
        inp=torch.randn(2,2,64);p=m(inp);changed=inp.clone();changed[:,1]+=100
        assert torch.equal(p,m(changed));assert torch.equal(p,inp[:,0,-1])
        l=ControlledLSTM(2).eval();assert l(inp).shape==(2,)
    ok('matched_removal_has_no_command_bypass_initial_persistence_and_LSTM_shape')
    with tempfile.TemporaryDirectory(prefix='exp012_eng_') as tmp:
        root=Path(tmp);identity={'run_hash':'SYNTHETIC_ONLY','job_id':'toy'}
        b=commit(root,'toy',identity,{'a.json':b'{}'});assert read_commit(root,'toy',identity)==b
        (b/'a.json').write_bytes(b'bad');must_fail(lambda:read_commit(root,'toy',identity));ok('committed_bytes_tamper_rejected')
        assert read_commit(root,'uncommitted',identity) is None;ok('orphan_not_completed')
        g=DataGuard(root,root)
        must_fail(lambda:g.hook('open',(root/'calibration/A.npy',)))
        must_fail(lambda:g.hook('open',(root/'test/A.npy',)))
        must_fail(lambda:g.hook('open',(root/'labeled_anomalies.csv',)))
        ok('phase_allowlist_TEST_labels_calibration_rejected')
    return {'status':'PASS','checks':passed,'count':len(passed),'fixtures':'synthetic only; no scientific fits',
            'golden_sha256':sha(gold),'no_NASA_labels':True}

def resume_check(device='cpu'):
    """Local/Colab engineering check only; two tiny CPU epochs, never project data."""
    settings={'lookback':64,'batch_size':64,'learning_rate':.001,'max_epochs':2,'patience':10,'min_delta':0.}
    row={'channel':'SYNTHETIC','mission':'SYNTHETIC','fit_end':85,'stop_end':110}
    raw=np.column_stack([np.sin(np.arange(120)/5),np.cos(np.arange(120)/7)]).astype(np.float32)
    uid={'run_hash':'ENGINEERING','job_id':'SYNTHETIC'}
    with tempfile.TemporaryDirectory(prefix='exp012_eng_') as tmp:
        p=Path(tmp);model,a,h,pred=train_lstm(raw,row,[0,1],17,settings,p/'all/CURRENT.json',uid,device)
        one={**settings,'max_epochs':1};train_lstm(raw,row,[0,1],17,one,p/'resume/CURRENT.json',uid,device)
        m,b,hh,pp=train_lstm(raw,row,[0,1],17,settings,p/'resume/CURRENT.json',uid,device)
        assert tensor_hash(a['state_dict'])==tensor_hash(b['state_dict']) and h==hh and np.array_equal(pred,pp)
        _,c,hhh,ppp=train_lstm(raw,row,[0,1],17,settings,p/'resume/CURRENT.json',uid,device)
        assert tensor_hash(b['state_dict'])==tensor_hash(c['state_dict']) and hh==hhh and np.array_equal(pp,ppp)
        pointer=json.loads((p/'resume/CURRENT.json').read_text());(p/'resume'/pointer['file']).write_bytes(b'broken')
        must_fail(lambda:train_lstm(raw,row,[0,1],17,settings,p/'resume/CURRENT.json',uid,device))
    return {'status':'PASS','name':'synthetic_'+device+'_epoch_resume_exact_and_tamper_rejected','no_scientific_result':True}

if __name__=='__main__':
    torch.set_num_threads(1);print(json.dumps({'engineering':run_checks(),'resume':resume_check()},indent=2))
