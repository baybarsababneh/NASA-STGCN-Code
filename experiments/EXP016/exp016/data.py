"""No labels are parsed here. All windows stay within their original file."""
from pathlib import Path
import numpy as np
import torch
from .io import ROOT,sha,read_json,save_json,safe_extract

def splits(length,c):
    n=length-c['history']-c['horizon'] # author's exclusive upper bound
    cut=int(n*(1-c['validation_fraction']))
    # Train horizon ends before first validation target; 9 straddling starts purged.
    fit=np.arange(cut-c['horizon']+1,dtype=np.int64)
    val=np.arange(cut,n,dtype=np.int64)
    if len(fit)<1 or len(val)<1:raise ValueError(f'Insufficient TRAIN length {length}: fit={len(fit)}, validation={len(val)}')
    assert fit[-1]+c['history']+c['horizon']-1 < val[0]+c['history']
    return fit,val

def stage_data(dest):
    dest=Path(dest);expected=read_json(ROOT/'inputs/ARRAY_MANIFEST.json')
    if not (dest/'data/data/train/A-1.npy').exists():safe_extract(ROOT/'inputs/archive.zip',dest)
    for name,h in expected.items():
        if not (dest/name).exists() or sha(dest/name)!=h:raise RuntimeError('Input identity mismatch: '+name)
    return dest/'data/data'

def audit_data(root,c,out):
    root=Path(root);rows=[]
    roster=sorted(p.stem for p in (root/'train').glob('*.npy'))
    if len(roster)!=c['training_channels']:raise ValueError('Unexpected training roster')
    for ch in roster:
        tr=np.load(root/'train'/f'{ch}.npy',allow_pickle=False)
        te=np.load(root/'test'/f'{ch}.npy',mmap_mode='r',allow_pickle=False)
        pr=np.load(root/'2018-05-19_15.00.10/y_hat'/f'{ch}.npy',mmap_mode='r',allow_pickle=False)
        if tr.ndim!=2 or tr.shape[1] not in (25,55) or te.shape[1]!=tr.shape[1]:raise ValueError('Bad input dimensions '+ch)
        if not np.isfinite(tr).all() or not np.isfinite(te).all() or not np.isfinite(pr).all():raise ValueError('Nonfinite inputs '+ch)
        if pr.shape!=(len(te)-260,):raise ValueError('Released prediction alignment failure '+ch)
        fit,val=splits(len(tr),c)
        rows.append(dict(channel=ch,mission='SMAP' if tr.shape[1]==25 else 'MSL',nodes=tr.shape[1],
          train_rows=len(tr),test_rows=len(te),fit_windows=len(fit),val_windows=len(val),
          purged_windows=9,first_val_target=int(val[0]+250),last_fit_target=int(fit[-1]+259),
          test_target_start=250,test_target_end_exclusive=len(te)-10))
    save_json(Path(out)/'DATA_AUDIT.json',rows)
    return rows

class Windows:
    def __init__(self,array,c,device):
        self.raw=torch.as_tensor(np.array(array,dtype=np.float32,copy=True),device=device)
        self.h=c['history'];self.p=c['horizon']
        self.offset=torch.arange(self.h,device=device)
        self.future=torch.arange(self.p,device=device)
    def batch(self,starts):
        starts=torch.as_tensor(starts,dtype=torch.long,device=self.raw.device)
        x=self.raw[starts[:,None]+self.offset].transpose(1,2).contiguous()
        y=self.raw[starts[:,None]+self.h+self.future,0]
        return x,y
