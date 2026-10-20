"""Local extension of supplied models: fixed graph, causal last-step TCN.

Last-output-only valid convolutions avoid computing discarded time positions.
For L >= receptive field this is mathematically identical to the supplied
left-padded encoder. Legacy two-layer checkpoint names remain conv1/conv2.
"""
from pathlib import Path
import importlib.util
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('supplied_round2_models',ROOT/'inputs/models.py')
supplied=importlib.util.module_from_spec(spec);spec.loader.exec_module(supplied)
adjacency=supplied.adjacency

class Temporal(nn.Module):
    def __init__(self,h,k=3,dilations=(1,2)):
        super().__init__();self.k=k;self.dilations=tuple(dilations)
        self.receptive_field=1+(k-1)*sum(dilations)
        for i,d in enumerate(dilations):
            self.add_module(f'conv{i+1}',nn.Conv1d(1 if i==0 else h,h,k,dilation=d))
    def forward(self,x):
        b,n,L=x.shape;R=self.receptive_field
        z=x[:,:,-min(L,R):].reshape(b*n,1,min(L,R))
        # Keep only the suffix needed by later layers. Padding is applied
        # to hidden states at their true start boundary, preserving causal
        # zero-padding semantics even when L is shorter than the total RF.
        spans=[(self.k-1)*d for d in self.dilations]
        for i,span in enumerate(spans):
            needed=min(L,1+sum(spans[i+1:]))
            z=z[:,:,-(needed+span):]
            padding=max(0,needed+span-z.shape[-1])
            if padding:z=F.pad(z,(padding,0))
            z=torch.relu(getattr(self,f'conv{i+1}')(z))
        return z[:,:,-1].reshape(b,n,-1)

class TCN(supplied.TCNSTGCN):
    def __init__(self,n,L,h,a,k=3,dilations=(1,2)):
        # Preserve initial parameter RNG order for the exact two-layer candidate.
        super().__init__(n,L,h,a,k)
        if tuple(dilations)==(1,2):
            old=self.temporal
            replacement=Temporal.__new__(Temporal);nn.Module.__init__(replacement)
            replacement.k=k;replacement.dilations=(1,2);replacement.receptive_field=1+3*(k-1)
            replacement.conv1=old.conv1;replacement.conv2=old.conv2;self.temporal=replacement
        else:
            self.temporal=Temporal(h,k,dilations)

def make(cfg,n,a):
    if cfg['kind']=='tcn_gcn':return TCN(n,cfg['lookback'],cfg['hidden'],a,cfg.get('kernel_size',3),cfg.get('dilations',[1,2]))
    return supplied.make(cfg,n,a)
