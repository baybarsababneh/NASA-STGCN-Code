"""Data policies and metrics. Intervals are inclusive; predictions index raw rows."""
from pathlib import Path
import ast, json, hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/data/data'
ARCHIVE = DATA / '2018-05-19_15.00.10'
PROTOCOL = json.loads((ROOT / 'configs/protocol.json').read_text())

def intervals(mask):
    d = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(int))
    return list(zip(np.flatnonzero(d == 1).tolist(), (np.flatnonzero(d == -1)-1).tolist()))

def labels(policy='union'):
    df = pd.read_csv(ROOT/'data/labeled_anomalies.csv')
    if policy == 'rows':
        return [dict(channel=r.chan_id, spacecraft=r.spacecraft, events=ast.literal_eval(r.anomaly_sequences)) for r in df.itertuples()]
    out=[]
    for ch,g in df.groupby('chan_id',sort=True):
        mask=np.zeros(int(g.iloc[0].num_values),bool)
        rows = g if policy=='union' else g.iloc[:1] if policy=='first' else g.iloc[-1:]
        for r in rows.itertuples():
            for a,b in ast.literal_eval(r.anomaly_sequences): mask[a:b+1]=True
        out.append(dict(channel=ch,spacecraft=g.iloc[0].spacecraft,events=intervals(mask)))
    return out

def prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.
    r=tp/(tp+fn) if tp+fn else 0.
    return dict(tp=int(tp),fp=int(fp),fn=int(fn),precision=p,recall=r,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                f05=1.25*tp/(1.25*tp+fp+.25*fn) if 1.25*tp+fp+.25*fn else 0.)

def event_metrics(truth,pred):
    overlap=lambda a,b: a[0]<=b[1] and b[0]<=a[1]
    tp=sum(any(overlap(t,p) for p in pred) for t in truth)
    fp=sum(not any(overlap(t,p) for t in truth) for p in pred)
    return prf(tp,fp,len(truth)-tp)

def point_metrics(truth,pred):
    truth=np.asarray(truth,bool);pred=np.asarray(pred,bool)
    return prf(np.sum(truth&pred),np.sum(~truth&pred),np.sum(truth&~pred))

def smooth(x,span=10):
    return pd.Series(x).ewm(span=span,adjust=True).mean().to_numpy()

def load(ch,split):
    return np.load(DATA/split/(ch+'.npy')).astype(np.float32)

def dump(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
