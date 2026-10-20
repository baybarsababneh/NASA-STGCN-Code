"""Engineering launch gates; no architecture selection from these checks."""
import json
from pathlib import Path
import numpy as np
import torch
from common import sha,digest,freeze_json,write_json
from models_exp007 import MessageGatedGCN
from v2_models_exp006 import TCNGCNFixedTuned
from v2_models_exp002 import TCNGCNFixed
from v2_data import TRAIN_DIR,list_channels,infer_mission
from v2_graph import build_channel_graph
import training_core as core


def model_checks(device):
    outcomes=[]
    a=np.array([[0,1],[0,0]],dtype=np.float32)
    for dilations in [(1,2),(1,2,4,8)]:
        for mode in ['fixed','static','dynamic']:
            torch.manual_seed(17)
            model=MessageGatedGCN(2,8,a,dilations=dilations,gate_mode=mode).to(device)
            torch.manual_seed(17)
            ref=TCNGCNFixedTuned(2,8,a,dilations=dilations).to(device)
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in ref.state_dict().items())
            probe=torch.randn(8,2,64,device=device)
            assert torch.equal(model(probe),probe[:,0,-1])
            with torch.no_grad():
                model.head.weight.fill_(.1);ref.head.weight.fill_(.1)
                pred,gate=model.forward_with_gate(probe)
                assert torch.equal(pred,ref(probe))
                assert torch.equal(gate,torch.ones_like(gate))
                alter=probe.clone();alter[:,:,:-model.receptive_field]+=123
                assert torch.equal(model(alter),model(probe))
                # Nonzero learned gate before testing removal: not a trivial zero-weight test.
                if mode=='dynamic':model.gate_weight.fill_(.05)
                if mode!='fixed':model.gate_bias.fill_(.2)
                assert torch.equal(model(alter),model(probe))
                model.remove_messages=True
                alter=probe.clone();alter[:,1:]+=17
                assert torch.equal(model(probe),model(alter))
                clone=MessageGatedGCN(2,8,a,dilations=dilations,gate_mode=mode,remove_messages=True).to(device)
                clone.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()})
                assert torch.equal(model(probe),clone(probe))
            outcomes.append(dict(mode=mode,rf=model.receptive_field,identity_causality_removal_reload=True))
    # Historical MSL model and generic RF7 implementation have identical backbone parameters/algebra.
    torch.manual_seed(17);old=TCNGCNFixed(2,8,a).to(device)
    torch.manual_seed(17);new=MessageGatedGCN(2,8,a).to(device)
    mapped={k.replace('temporal.conv1.','temporal.convs.0.').replace('temporal.conv2.','temporal.convs.1.'):v for k,v in old.state_dict().items()}
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in mapped.items())
    with torch.no_grad():
        old.head.weight.fill_(.1);new.head.weight.fill_(.1)
        assert torch.equal(old(probe),new(probe))
    # Check that dynamic dependence and gradients actually exist with a nonzero head.
    model=MessageGatedGCN(2,8,a,gate_mode='dynamic').to(device)
    with torch.no_grad():
        for name,p in model.named_parameters():
            if not name.startswith('gate_'):p.fill_(.1)
    x=torch.ones(4,2,64,device=device)
    model(x).sum().backward()
    assert model.gate_weight.grad.abs().sum()>0 and model.gate_bias.grad.abs()>0
    with torch.no_grad():
        model.gate_weight.fill_(.1)
        _,g1=model.forward_with_gate(x);_,g2=model.forward_with_gate(x*2)
        assert not torch.equal(g1,g2) and torch.all((g1>0)&(g1<2))
    # No incoming edges: all modes have zero message and remain well-defined.
    for mode in ['fixed','static','dynamic']:
        m=MessageGatedGCN(1,8,np.zeros((1,1),np.float32),gate_mode=mode).to(device)
        y,g=m.forward_with_gate(torch.randn(4,1,64,device=device))
        assert torch.isfinite(y).all() and torch.equal(g,torch.ones_like(g))
    return outcomes


def preflight(root,out,config):
    root=Path(root);out=Path(out)
    original={v['channel']:v for v in json.loads((root/'provenance/EXP006_data_manifest.json').read_text())}
    original_graph={(v['channel'],v['fit_fraction']):v for v in json.loads((root/'provenance/EXP006_graph_manifest.json').read_text())}
    channels={};data_rows=[];graphs=[]
    for ch in list_channels():
        raw=np.load(TRAIN_DIR/(ch+'.npy'),allow_pickle=False)
        assert raw.ndim==2 and np.isfinite(raw).all()
        mission=infer_mission(raw);channels[ch]=mission
        h=sha(TRAIN_DIR/(ch+'.npy'))
        assert h==original[ch]['sha256']
        data_rows.append(dict(channel=ch,mission=mission,rows=len(raw),columns=raw.shape[1],sha256=h))
        for f in [.4,.5,.6]:
            graph=build_channel_graph(ch,f)
            target=raw.astype(np.float32)[:int(np.floor(len(raw)*f)),0]
            mean=float(target.mean());scale=float(target.std());scale=scale if np.isfinite(scale) and scale>=1e-8 else 1.
            rec=dict(channel=ch,mission=mission,fit_fraction=f,features=graph.feature_indices.tolist(),
                sources=graph.source_original_columns.tolist(),adjacency_sha256=digest(graph.adjacency.tolist()),
                target_mean=mean,target_scale=scale)
            assert rec==original_graph[(ch,f)],f'Preprocessing changed: {ch}, {f}'
            graphs.append(rec)
    assert list(channels.values()).count('MSL')==27 and list(channels.values()).count('SMAP')==54
    for m,roster in config['screening_channels'].items():
        assert len(roster)==6 and all(channels[ch]==m for ch in roster)
    checks=model_checks(torch.device('cuda'))
    # Two tiny production-loop repeats for each role on each selected RF.
    for m in config['missions']:
        for mode in config['roles']:
            spec=config['baseline_by_mission'][m]
            states=[];metrics=[]
            rng=np.random.default_rng(1707)
            x=rng.normal(size=(24,2,64)).astype(np.float32);y=rng.normal(size=24).astype(np.float32)
            for repeat in range(2):
                core.set_seed(17)
                model=MessageGatedGCN(2,8,np.array([[0,1],[0,0]],np.float32),dilations=spec['dilations'],gate_mode=mode)
                model,val,*_=core._train_with_early_stopping(model,x[:16],y[:16],x[16:],y[16:],17,.001,
                    dict(config['training_common'],batch_size=8),smoke_epochs=3)
                states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()});metrics.append(val)
            assert metrics[0]==metrics[1]
            assert all(torch.equal(v,states[1][k]) for k,v in states[0].items())
    freeze_json(out/'data_manifest.json',data_rows);freeze_json(out/'graph_manifest.json',graphs)
    write_json(out/'preflight.json',{'status':'PASS','model_checks':checks,'cuda_repeatability':'exact',
        'TRAIN_channels':81,'preprocessing_matches_completed_EXP006':True})
    return channels
