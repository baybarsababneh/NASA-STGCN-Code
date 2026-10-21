"""Executed on the actual CUDA device before scientific fitting."""
import tempfile
from pathlib import Path
import numpy as np
import torch
from .model import STGCN
from .train import train_one,load_state,seed_all
from .io import save_json

def run_gate(c,out,device):
    checks=[]
    if device!='cuda':
        save_json(out/'GPU_GATE.json',dict(status='NOT_RUN',reason='No CUDA in engineering environment'))
        return
    for nodes in (25,55):
        seed_all(17);m=STGCN(nodes,c['hidden'],dropout=0).cuda();torch.nn.init.normal_(m.head.weight,std=.01)
        x=torch.randn(2,nodes,250,device='cuda',requires_grad=True)
        a=m(x);ga=torch.autograd.grad(a.square().mean(),(x,*m.parameters()))
        b=m(x,dense=True);gb=torch.autograd.grad(b.square().mean(),(x,*m.parameters()))
        torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-4)
        for u,v in zip(ga,gb):torch.testing.assert_close(u,v,atol=2e-5,rtol=3e-4)
        checks.append(f'CUDA dense/sparse forward and all gradients, {nodes} features')
    with tempfile.TemporaryDirectory() as td:
        cfg={**c,'hidden':4,'epochs':2,'batch_size':8,'checkpoint_steps':2}
        data=np.random.default_rng(4).normal(size=(330,25)).astype('float32');td=Path(td)
        train_one(data,cfg,17,td/'whole','GPU_GATE','cuda')
        train_one(data,cfg,17,td/'resume','GPU_GATE','cuda',stop_after_steps=2)
        train_one(data,cfg,17,td/'resume','GPU_GATE','cuda')
        a=load_state(td/'whole'/'last.pt','cuda');b=load_state(td/'resume'/'last.pt','cuda')
        for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],atol=0,rtol=0)
        if a['history']!=b['history']:raise RuntimeError('CUDA resume history differs')
        checks.append('Exact interrupted CUDA fixture continuation')
    save_json(out/'GPU_GATE.json',dict(status='PASS',checks=checks,full_GPU_experiment_run=False))
