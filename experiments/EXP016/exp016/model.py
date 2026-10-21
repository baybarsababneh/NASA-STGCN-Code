"""Command-star ST-GCN; exact sparse evaluation of a causal dilation stack.

Dense reference uses left-zero padding at EVERY layer. The production path
computes only ancestors of the last temporal state, without changing its math.
This is not shorter history, subsampling of windows, or approximate convolution.
"""
import torch
from torch import nn
from torch.nn import functional as F

class STGCN(nn.Module):
    def __init__(self, nodes, hidden=64, history=250, horizon=10, dropout=.3):
        super().__init__()
        self.nodes, self.history, self.horizon = nodes, history, horizon
        self.dilations = (1, 2, 4, 8, 16, 32, 64)
        self.convs = nn.ModuleList([nn.Conv1d(1 if i == 0 else hidden, hidden, 3, dilation=d)
                                    for i, d in enumerate(self.dilations)])
        self.gain = nn.Parameter(torch.ones(nodes, hidden))
        self.bias = nn.Parameter(torch.zeros(nodes, hidden))
        self.self_map = nn.Linear(hidden, hidden)
        self.message = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden, horizon)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Required nonnegative output coordinates at each depth.
        required = [None] * 8
        required[7] = [history - 1]
        for i in reversed(range(7)):
            d = self.dilations[i]
            required[i] = sorted({t-k*d for t in required[i+1] for k in (0,1,2) if t-k*d >= 0})
        for i, d in enumerate(self.dilations):
            prev = list(range(history)) if i == 0 else required[i]
            lookup = {t:j+1 for j,t in enumerate(prev)} # zero slot for layer padding
            gather = [[lookup.get(t-k*d, 0) for k in (2,1,0)] for t in required[i+1]]
            self.register_buffer(f'ix_{i}', torch.tensor(gather, dtype=torch.long))
            self.register_buffer(f'skip_{i}', torch.tensor([lookup[t] for t in required[i+1]],dtype=torch.long))

    def encode(self, x, dense=False):
        b,n,t = x.shape
        z = x.reshape(b*n,1,t)
        for i,(conv,d) in enumerate(zip(self.convs,self.dilations)):
            if dense:
                v = conv(F.pad(z,(2*d,0)))
                if i: v = v + z
            else:
                zp = F.pad(z,(1,0))
                ix = getattr(self,f'ix_{i}')
                # [BN,C,P,K] -> [BN,P,C*K], matching Conv1d weight order.
                patch = zp[:,:,ix].permute(0,2,1,3).flatten(2)
                v = F.linear(patch,conv.weight.flatten(1),conv.bias).transpose(1,2)
                if i: v = v + zp[:,:,getattr(self,f'skip_{i}')]
            z = F.relu(v)
        return z[:,:,-1].reshape(b,n,-1)

    def forward(self, x, dense=False):
        if x.ndim != 3 or x.shape[1:] != (self.nodes,self.history):
            raise ValueError(f'Expected [B,{self.nodes},{self.history}], got {tuple(x.shape)}')
        z = self.dropout(F.relu(self.encode(x,dense)*self.gain+self.bias))
        # A[0,j]=1/(N-1), j>0. Only telemetry readout is needed.
        message = z[:,1:].mean(1) if self.nodes>1 else torch.zeros_like(z[:,0])
        h = F.relu(self.self_map(z[:,0])+self.message(message))
        return x[:,0,-1,None]+self.head(self.dropout(h))
