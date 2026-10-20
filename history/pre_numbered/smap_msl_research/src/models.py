"""Small forecasting controls and a typed, directed command-to-target ST-GCN.

Each node gets a shared length-L temporal convolution (implemented as a Linear
over the lag axis, exactly a valid Conv1d kernel of width L at the forecast time).
Node-specific gains retain command identity. A directed mean graph convolution
passes command embeddings to the telemetry node. No physical topology is claimed.
"""
import numpy as np
import torch
from torch import nn

def adjacency(fit, mode):
    n=fit.shape[1]; a=np.zeros((n,n),np.float32)
    if n<2 or mode=='none': return a
    chosen=np.arange(1,n)
    if mode=='top4':
        # Fit-only lagged association with target changes. No test correlations.
        dy=np.diff(fit[:,0]); scores=[]
        for j in chosen:
            cs=[]
            for lag in [1,2,4,8,16]:
                c=fit[:-lag,j]; y=dy[lag-1:]
                cs.append(abs(np.corrcoef(c,y)[0,1]) if c.std()>1e-8 and y.std()>1e-8 else 0.)
            scores.append(max(cs))
        chosen=chosen[np.argsort(scores,kind='stable')[-min(4,n-1):]]
    a[0,chosen]=1/len(chosen)
    return a

class STGCN(nn.Module):
    def __init__(self,n,L,h,a):
        super().__init__()
        self.temporal=nn.Linear(L,h)
        self.node_gain=nn.Parameter(torch.ones(n,h))
        self.node_bias=nn.Parameter(torch.zeros(n,h))
        self.self_map=nn.Linear(h,h)
        self.message=nn.Linear(h,h,bias=False)
        self.head=nn.Linear(h,1)
        self.register_buffer('adjacency',torch.tensor(a))
        # Residual forecast starts at persistence, useful for short quiet streams.
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        # x: batch, nodes, past time; output: telemetry at the next unseen row.
        z=torch.relu(self.temporal(x)*self.node_gain+self.node_bias)
        z=torch.relu(self.self_map(z)+self.message(self.adjacency@z))
        return x[:,0,-1]+self.head(z[:,0]).squeeze(-1)

class MLP(nn.Module):
    def __init__(self,n,L,h):
        super().__init__();self.net=nn.Sequential(nn.Flatten(),nn.Linear(n*L,h),nn.ReLU(),nn.Linear(h,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return x[:,0,-1]+self.net(x).squeeze(-1)

class LSTM(nn.Module):
    def __init__(self,n,L,h):
        super().__init__();self.rnn=nn.LSTM(n,h,batch_first=True);self.head=nn.Linear(h,1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
    def forward(self,x):
        z,_=self.rnn(x.transpose(1,2));return x[:,0,-1]+self.head(z[:,-1]).squeeze(-1)

def make(cfg,n,a):
    model=STGCN(n,cfg['lookback'],cfg['hidden'],a) if cfg['kind']=='gcn' else (MLP if cfg['kind']=='mlp' else LSTM)(n,cfg['lookback'],cfg['hidden'])
    return model
