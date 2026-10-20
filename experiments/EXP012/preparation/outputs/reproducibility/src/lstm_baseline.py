"""Controlled one-step PyTorch LSTM adaptation; not historical .h5 reuse."""
import torch
from torch import nn

class ControlledLSTM(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.first=nn.LSTM(features,80,batch_first=True)
        self.second=nn.LSTM(80,80,batch_first=True)
        self.dropout=nn.Dropout(.3);self.head=nn.Linear(80,1)
    def forward(self,x):
        if x.ndim!=3: raise ValueError('Expected [batch, features, lookback]')
        z,_=self.first(x.transpose(1,2));z=self.dropout(z);z,_=self.second(z)
        return self.head(self.dropout(z[:,-1])).squeeze(-1)
