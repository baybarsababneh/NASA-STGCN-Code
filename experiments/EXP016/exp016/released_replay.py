"""Inference-only NumPy verification, adapted from project followup verify_sources.py. No LSTM fitting."""
import json
import numpy as np

def replay(f,x):
    cfg=json.loads(f.attrs['model_config'])['config']
    a=x.astype(np.float32)
    for layer in cfg:
        c=layer['config'];typ=layer['class_name'];name=c['name']
        if typ in ('Dropout','Activation'):continue
        g=f['model_weights'][name][name]
        if typ=='LSTM':
            assert c['activation']=='tanh' and c['recurrent_activation']=='hard_sigmoid'
            w=g['kernel:0'][:];u=g['recurrent_kernel:0'][:];b=g['bias:0'][:]
            h=np.zeros((len(x),80),np.float32);cell=h.copy();seq=[]
            projected=a@w+b
            for t in range(a.shape[1]):
                v=projected[:,t]+h@u
                i=np.clip(v[:,:80]*.2+.5,0,1);fg=np.clip(v[:,80:160]*.2+.5,0,1)
                cc=np.tanh(v[:,160:240]);o=np.clip(v[:,240:]*.2+.5,0,1)
                cell=fg*cell+i*cc;h=o*np.tanh(cell);seq.append(h.copy())
            a=np.stack(seq,axis=1) if c['return_sequences'] else h
        elif typ=='Dense':a=a@g['kernel:0'][:]+g['bias:0'][:]
        else:raise ValueError(typ)
    return a
