"""A one-hop directed command graph with a telemetry-independent normal reference."""
import torch
from torch import nn
import torch.nn.functional as F

def dilated_conv1d_fp32(z,c):
    """The frozen valid Conv1d, expressed as lagged windows and one GEMM.

    z is already left-padded by NodeEncoder. Windows are flattened in
    (input channel, kernel tap) order, exactly matching Conv1d.weight.
    No weights, activations, precision flags, RNG draws or sequence positions
    change. FP32 reduction order can differ from the native convolution.
    """
    d=c.dilation[0];K=c.kernel_size[0]
    if c.groups!=1 or c.stride!=(1,) or c.padding!=(0,):
        raise ValueError('This correction is only for the frozen valid stride-1 encoder')
    windows=z.unfold(2,d*(K-1)+1,1)[...,::d]
    M,C,T,_=windows.shape
    columns=windows.permute(0,2,1,3).reshape(M*T,C*K)
    return F.linear(columns,c.weight.flatten(1),c.bias).reshape(M,T,c.out_channels).transpose(1,2)

class NodeEncoder(nn.Module):
    def __init__(self,h=32,gemm_dilated=False):
        super().__init__();self.gemm_dilated=gemm_dilated;self.convs=nn.ModuleList([nn.Conv1d(1 if i==0 else h,h,3,dilation=d) for i,d in enumerate([1,2,4,8,16])])
    def forward(self,x):
        B,N,T=x.shape;z=x.reshape(B*N,1,T)
        for c in self.convs:
            padded=F.pad(z,(2*c.dilation[0],0))
            z=F.silu(dilated_conv1d_fp32(padded,c) if self.gemm_dilated and c.dilation!=(1,) else c(padded))
        return z[:,:,-1].reshape(B,N,-1)

class CommandReferenceGCN(nn.Module):
    def __init__(self,n_features,n_entities,role='graph',h=32):
        super().__init__();self.role=role;self.n_features=n_features
        self.encoder=NodeEncoder(h,gemm_dilated=True);self.entity=nn.Embedding(n_entities,8);self.command=nn.Embedding(n_features-1,h)
        self.register_buffer('active_commands',torch.ones(n_entities,n_features-1,dtype=torch.bool))
        self.query=nn.Linear(8,h);self.key=nn.Linear(h,h,bias=False);self.message=nn.Linear(h,h,bias=False)
        self.null=nn.Parameter(torch.zeros(h));self.fusion=nn.Linear((n_features-1)*h,h) if role=='fusion' else None
        self.reference=nn.Sequential(nn.Linear(h+8,h),nn.SiLU(),nn.Linear(h,1))
        self.discriminator=nn.Sequential(nn.Linear(2*h+8+8,h),nn.SiLU(),nn.Dropout(.1),nn.Linear(h,1))
    def forward(self,x,entity):
        # x includes current telemetry, while command encoder uses only t-64:t-1.
        cmd=self.encoder(x[:,:-1,1:].transpose(1,2))+self.command.weight[None]
        active=self.active_commands[entity];cmd=cmd*active[:,:,None]
        e=self.entity(entity);q=self.query(e);keys=self.key(cmd)
        logits=(keys*q[:,None]).sum(-1)/(cmd.shape[-1]**.5)
        logits=logits.masked_fill(~active,float('-inf'))
        # Null message prevents forced command mixing; four learned neighbours plus null.
        top=torch.topk(logits,min(4,logits.shape[1]),dim=1).indices
        supported=torch.zeros_like(logits,dtype=torch.bool).scatter_(1,top,True)
        logits=logits.masked_fill(~(supported&active),float('-inf'))
        logits=torch.cat([logits,torch.zeros(len(x),1,device=x.device)],1)
        values=torch.cat([cmd,self.null.expand(len(x),1,-1)],1)
        msg=(logits.softmax(1)[:,:,None]*values).sum(1)
        if self.role=='removed': msg=torch.zeros_like(msg)
        elif self.role=='fusion': msg=self.fusion(cmd.flatten(1))
        msg=self.message(msg)
        ref=self.reference(torch.cat([msg,e],1)).squeeze(1)
        obs=self.encoder(x[:,:,0].unsqueeze(1)).squeeze(1)
        v=x[:,-16:,0];r=v-ref[:,None];d=v[:,1:]-v[:,:-1]
        feats=torch.stack([r[:,-1],r[:,-1].abs(),r.mean(1),r.abs().mean(1),r.square().mean(1).clamp_min(1e-12).sqrt(),v.std(1,unbiased=False),d.abs().mean(1),d[:,-1]],1)
        logit=self.discriminator(torch.cat([obs,msg,e,feats],1)).squeeze(1)
        return ref,logit
