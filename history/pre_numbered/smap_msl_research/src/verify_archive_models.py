"""Read-only port of legacy Keras hard-sigmoid LSTM inference, no TensorFlow needed."""
import json
import h5py
import numpy as np
import torch
from common import ROOT,ARCHIVE,load,dump
torch.set_num_threads(1)

def infer(ch,count=8):
    raw=load(ch,'test');x=torch.tensor(np.stack([raw[i:i+250] for i in range(count)]))
    with h5py.File(ARCHIVE/'models'/f'{ch}.h5','r') as f:
        config=json.loads(f.attrs['model_config'])
        layers=config['config'];w=f['model_weights']
        for layer in layers:
            c=layer['config'];name=c['name'];kind=layer['class_name']
            if kind=='LSTM':
                assert c['activation']=='tanh' and c['recurrent_activation']=='hard_sigmoid'
                d=w[name][name];W=torch.tensor(d['kernel:0'][:]);U=torch.tensor(d['recurrent_kernel:0'][:]);b=torch.tensor(d['bias:0'][:])
                h=torch.zeros(count,c['units']);cell=torch.zeros_like(h);ys=[]
                for t in range(x.shape[1]):
                    z=x[:,t]@W+h@U+b;i,forget,g,o=z.chunk(4,-1)
                    i=(.2*i+.5).clamp(0,1);forget=(.2*forget+.5).clamp(0,1);o=(.2*o+.5).clamp(0,1)
                    cell=forget*cell+i*torch.tanh(g);h=o*torch.tanh(cell);ys.append(h)
                x=torch.stack(ys,1) if c['return_sequences'] else h
            elif kind=='Dense':
                d=w[name][name];x=x@torch.tensor(d['kernel:0'][:])+torch.tensor(d['bias:0'][:])
            elif kind in ['Dropout','Activation']:pass
            else:raise ValueError(kind)
    pred=x[:,0].numpy();saved=np.load(ARCHIVE/'y_hat'/f'{ch}.npy')[:count]
    return dict(channel=ch,n=count,max_abs_difference=float(np.max(np.abs(pred-saved))),matches_at_1e_5=bool(np.allclose(pred,saved,atol=1e-5)),predictions=pred.tolist(),saved=saved.tolist())

if __name__=='__main__':
    result=[infer(c) for c in ['P-1','M-1','C-2','P-2']]
    dump(ROOT/'audit/archive_weight_verification.json',result)
    print([(r['channel'],r['max_abs_difference']) for r in result])
