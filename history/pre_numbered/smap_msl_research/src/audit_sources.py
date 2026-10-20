"""Recreate inventory and hashes without modifying the supplied source files."""
import argparse,ast,hashlib,json,zipfile,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT,DATA,dump,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sources',type=Path,default=ROOT/'source_material');a=ap.parse_args()
    if not DATA.exists():
        archive=next(a.sources.glob('*archive.zip'))
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                dest=(ROOT/'data'/member.filename).resolve()
                if not dest.is_relative_to((ROOT/'data').resolve()):raise ValueError('Unsafe ZIP path')
            z.extractall(ROOT/'data')
    rows=[];df=pd.read_csv(ROOT/'data/labeled_anomalies.csv')
    for r in df.itertuples():
        tr=np.load(DATA/'train'/f'{r.chan_id}.npy');te=np.load(DATA/'test'/f'{r.chan_id}.npy');events=ast.literal_eval(r.anomaly_sequences)
        fit=tr[:int(len(tr)*.6)]
        rows.append(dict(channel=r.chan_id,spacecraft=r.spacecraft,train_n=len(tr),test_n=len(te),features=tr.shape[1],label_n=r.num_values,events=len(events),out_of_bounds=any(x<0 or y>=len(te) for x,y in events),finite=bool(np.isfinite(tr).all() and np.isfinite(te).all()),context_binary=bool(np.isin(tr[:,1:],[0,1]).all() and np.isin(te[:,1:],[0,1]).all()),constant_context=int((tr[:,1:].std(0)==0).sum()),target_min=float(tr[:,0].min()),target_max=float(tr[:,0].max()),fit_constant_target=bool(fit[:,0].std()<1e-7),fit_active_commands=int((fit[:,1:].std(0)>0).sum()),train_sha256=sha(DATA/'train'/f'{r.chan_id}.npy'),test_sha256=sha(DATA/'test'/f'{r.chan_id}.npy')))
    pd.DataFrame(rows).to_csv(ROOT/'audit/data_inventory.csv',index=False)
    for f in a.sources.glob('*.ipynb'):
        nb=json.loads(f.read_text());text='\n\n'.join(f'### CELL {i} ({c["cell_type"]})\n'+''.join(c['source']) for i,c in enumerate(nb['cells']))
        (ROOT/'audit'/(f.stem+'.txt')).write_text(text)
    print(pd.DataFrame(rows).drop_duplicates('channel').groupby('spacecraft')[['train_n','test_n','fit_constant_target','fit_active_commands']].agg(['sum','min','max']))

if __name__=='__main__':main()
