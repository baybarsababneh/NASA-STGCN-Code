import copy,io,json,zipfile
from pathlib import Path
import numpy as np
import pytest
import torch
from exp016.model import STGCN
from exp016.data import Windows,splits
from exp016.io import ROOT,read_json,sha,safe_extract,Mirror,save_json,committed
from exp016.train import configure,seed_all,train_one,load_state
from exp016.detector import detect,runs
from exp016.metrics import counts,aggregate,average_precision,fbeta

@pytest.fixture(autouse=True)
def deterministic():configure('cpu');seed_all(7)

@pytest.mark.parametrize('nodes',[25,55])
def test_sparse_dense_forward_backward(nodes):
    m=STGCN(nodes,hidden=4,dropout=0).double();torch.nn.init.normal_(m.head.weight,std=.1)
    x=torch.randn(2,nodes,250,dtype=torch.float64,requires_grad=True)
    a=m(x);ga=torch.autograd.grad(a.square().sum(),(x,*m.parameters()),retain_graph=False)
    b=m(x,dense=True);gb=torch.autograd.grad(b.square().sum(),(x,*m.parameters()))
    torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,atol=1e-9,rtol=1e-9)
    assert a.shape==(2,10)

def test_full_history_and_all_features_reach_output():
    m=STGCN(25,hidden=2,dropout=0).double()
    with torch.no_grad():
        for p in m.parameters():p.fill_(.05)
    x=torch.ones(1,25,250,dtype=torch.float64,requires_grad=True)
    g=torch.autograd.grad(m(x).sum(),x)[0]
    assert (g>0).all(), 'Every feature and all 250 actual observations must influence output'

def test_no_future_input_and_exact_author_loop():
    c=read_json(ROOT/'config.json');a=np.arange(600*25).reshape(600,25).astype(np.float32);w=Windows(a,c,'cpu')
    for i in (0,1,339):
        x,y=w.batch([i]);np.testing.assert_array_equal(x.numpy()[0].T,a[i:i+250]);np.testing.assert_array_equal(y.numpy()[0],a[i+250:i+260,0])
    assert 600-250-10==340
    b=a.copy();b[250:]+=1e5
    np.testing.assert_array_equal(Windows(b,c,'cpu').batch([0])[0].numpy(),w.batch([0])[0].numpy())

def test_short_channel_purging():
    c=read_json(ROOT/'config.json');fit,val=splits(312,c)
    assert (len(fit),len(val))==(32,11)
    f=set((fit[:,None]+250+np.arange(10)).ravel());v=set((val[:,None]+250+np.arange(10)).ravel())
    assert not f&v
    assert fit[-1]+259==290 and val[0]+250==291
    with pytest.raises(ValueError):splits(260,c)

def test_resumption_exact_mid_epoch(tmp_path):
    c=read_json(ROOT/'config.json');c.update(hidden=4,epochs=3,batch_size=8,checkpoint_steps=2)
    a=np.random.default_rng(2).normal(size=(330,3)).astype(np.float32)
    train_one(a,c,17,tmp_path/'whole','id','cpu')
    train_one(a,c,17,tmp_path/'resume','id','cpu',stop_after_steps=2)
    train_one(a,c,17,tmp_path/'resume','id','cpu')
    p=load_state(tmp_path/'whole'/'last.pt','cpu');q=load_state(tmp_path/'resume'/'last.pt','cpu')
    assert p['history']==q['history'] and p['steps']==q['steps']
    for k in p['model']:torch.testing.assert_close(p['model'][k],q['model'][k],rtol=0,atol=0)
    for k in p['optimizer']['state']:
        for f,v in p['optimizer']['state'][k].items():
            if isinstance(v,torch.Tensor):torch.testing.assert_close(v,q['optimizer']['state'][k][f],rtol=0,atol=0)
    with pytest.raises(RuntimeError):train_one(a,c,17,tmp_path/'resume','changed_identity','cpu')
    assert committed(tmp_path/'resume','id')

def test_corrupt_checkpoint_falls_back(tmp_path):
    from exp016.train import save_state
    p=tmp_path/'last.pt';save_state(p,{'v':1});save_state(p,{'v':2});p.write_bytes(b'corrupt')
    assert load_state(p,'cpu')['v']==1
    p.with_suffix('.previous.pt').write_bytes(b'corrupt')
    with pytest.raises(RuntimeError):load_state(p,'cpu')

def test_mirror_restore(tmp_path):
    m=Mirror(tmp_path/'local',tmp_path/'drive');save_json(m.local/'a.json',{'test':1});m.sync(force=True)
    n=Mirror(tmp_path/'new',tmp_path/'drive');n.restore();assert read_json(n.local/'a.json')=={'test':1}

def test_zip_traversal_rejected(tmp_path):
    p=tmp_path/'unsafe.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('../evil','x')
    with pytest.raises(ValueError):safe_extract(p,tmp_path/'dest')

@pytest.mark.parametrize('length',[312,670,2051,3000])
@pytest.mark.parametrize('channel',['A-1','C-2'])
def test_native_detector_control_flow_parity(length,channel):
    rng=np.random.default_rng(44);a=rng.normal(0,.2,(length,2));pred=a[250:-10,0]+rng.normal(0,.05,length-260)
    if len(pred)>200:pred[100:180]+=1
    d=detect(a,pred,channel,'native_C2');ref=detect(a,pred,channel,'native_C2',reference=True)
    assert d['intervals']==ref
    assert np.all(d['issue_index'][d['flags']]>=np.arange(250,length-10)[d['flags']])

def test_uniform_removes_channel_specific_knowledge():
    rng=np.random.default_rng(2);a=rng.normal(size=(2100,3));p=a[250:-10,0]+.01;p[:100]+=2
    one=detect(a,p,'A-1','uniform');two=detect(a,p,'C-2','uniform')
    np.testing.assert_array_equal(one['flags'],two['flags']);np.testing.assert_array_equal(one['smoothed'],two['smoothed'])

def test_constant_detector_finite():
    a=np.zeros((700,25));d=detect(a,np.zeros(440),'A-1')
    assert not d['flags'].any() and np.isfinite(d['smoothed']).all()

def test_metric_event_bridge_point_and_endpoints():
    ix=np.arange(250,270);flag=np.zeros(20,bool);flag[2:15]=1
    c,_=counts([[252,254],[260,262]],flag,ix)
    assert (c['event_tp'],c['event_fn'],c['event_fp'])==(2,0,0)
    assert (c['literal_event_tp'],c['literal_event_fn'])==(1,1)
    assert (c['point_tp'],c['point_fp'],c['point_fn'])==(6,7,0)
    assert aggregate([c])['event_F05']==1
    assert fbeta(1,1,1)==.5
    c,_=counts([[250,251]],np.array([False,True]+[False]*18),ix)
    assert c['event_tp']==1 # inclusive final endpoint

def test_ap_ties_and_undefined():
    assert average_precision(np.array([1,0],bool),np.array([.5,.5]))==.5
    assert average_precision(np.array([0,0],bool),np.array([1.,2.])) is None

def test_point_counts_independent_random_enumeration():
    rng=np.random.default_rng(5);ix=np.arange(250,350);events=[[260,270],[290,300]]
    for _ in range(20):
        f=rng.random(100)<.1;c,_=counts(events,f,ix)
        truth=np.array([any(a<=t<=b for a,b in events) for t in ix])
        assert c['point_tp']==sum(bool(x and y) for x,y in zip(f,truth))
        assert c['point_fp']==sum(bool(x and not y) for x,y in zip(f,truth))
