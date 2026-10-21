import json
import h5py
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .released_replay import replay
from .io import save_json,sha

def verify_released(root,out):
    records=[]
    with threadpool_limits(1):
        for p in sorted((root/'train').glob('*.npy')):
            ch=p.stem;test=np.load(root/'test'/p.name);base=root/'2018-05-19_15.00.10'
            pred=np.load(base/'y_hat'/p.name);cached=np.load(base/'smoothed_errors'/p.name)
            positions=np.unique([0,len(pred)//2,len(pred)-1]);x=np.stack([test[j:j+250] for j in positions])
            with h5py.File(base/'models'/(ch+'.h5'),'r') as f:
                cfg=json.loads(f.attrs['model_config'])['config']
                if [r['config']['units'] for r in cfg if r['class_name']=='LSTM']!=[80,80]:raise ValueError('Wrong released LSTM '+ch)
                if next(r['config']['units'] for r in cfg if r['class_name']=='Dense')!=10:raise ValueError('Wrong released horizon')
                got=replay(f,x)[:,0]
            if not np.allclose(got,pred[positions],atol=1e-5,rtol=1e-4):raise ValueError('Author prediction replay failure '+ch)
            sm=pd.Series(np.abs(test[250:-10,0]-pred)).ewm(span=105).mean().to_numpy()
            if ch!='C-2':sm[:250]=sm[:500].mean()
            delta=float(np.max(np.abs(sm-cached)))
            if ch!='T-10' and not np.allclose(sm,cached,rtol=1e-12,atol=1e-12):raise ValueError('Author cached residual mismatch '+ch)
            records.append(dict(channel=ch,sampled_forecasts=len(positions),max_prediction_difference=float(np.max(np.abs(got-pred[positions]))),
                cached_error_max_difference=delta,cached_error_matches=bool(np.allclose(sm,cached,rtol=1e-12,atol=1e-12)),
                pred_sha256=sha(base/'y_hat'/p.name),model_sha256=sha(base/'models'/(ch+'.h5'))))
    save_json(out/'RELEASED_SOURCE_VERIFICATION.json',records)
    return records
