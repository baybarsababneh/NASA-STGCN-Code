"""NEXT LOCAL EXPERIMENT ONLY: calibrate locked forecasts using TRAIN's last 20%.

Not executed during Round 2. Does not open TEST or label files. It does not
select models, seeds or thresholds using anomaly metrics. The inherited policy
is absolute error, EWMA span 10, empirical calibration quantile 0.995.
"""
import argparse,json
import numpy as np
import pandas as pd
import torch
from round2_train import ROOT,BASE,predict
from round2_models import make

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='local_calibration');args=ap.parse_args()
    lock=json.loads((ROOT/'configs/locked_tcn.json').read_text());channels=pd.read_csv(ROOT/'inputs/channels.csv');out=ROOT/args.output
    out.mkdir(parents=True,exist_ok=False)  # Never silently replace an earlier experiment.
    rows=[]
    for mission,cfg in lock['missions'].items():
        for ch in channels[channels.spacecraft==mission].channel:
            for seed in lock['seeds']:
                path=ROOT/'checkpoints'/cfg['name']/f'{ch}_s{seed}.pt'
                saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['schema']=='round2_tcn_v1' and saved['configuration']==cfg
                meta=saved['metadata'];L=cfg['lookback'];start=meta['validation_end']
                raw=np.load(BASE/'data/data/data/train'/f'{ch}.npy',mmap_mode='r')
                assert len(raw)==meta['total_train_rows']
                x=np.array(raw[max(0,start-L):,meta['active']],dtype=np.float32)
                x[:,0]=(x[:,0]-meta['mean'])/meta['scale']
                if start<L:x=np.pad(x,((L-start,0),(0,0)))
                X=torch.tensor(np.lib.stride_tricks.sliding_window_view(x,L,axis=0)[:-1].copy());truth=x[L:,0]
                model=make(cfg,len(meta['active']),saved['adjacency']);model.load_state_dict(saved['state'],strict=True)
                prediction=predict(model,X);error=np.abs(prediction-truth)*meta['scale']
                score=pd.Series(error).ewm(span=10,adjust=True).mean().to_numpy()
                burn=min(10,len(score)//4)
                floor=max(1e-8,float(meta['scale'])*1e-6)
                threshold=max(float(np.quantile(score[burn:],.995)),floor)
                rows.append(dict(spacecraft=mission,channel=ch,configuration=cfg['name'],seed=seed,n_calibration=len(score)-burn,burn=burn,threshold_floor=floor,ewma_adjust=True,span=10,quantile=.995,threshold=threshold,calibration_mse_raw=float(np.mean((prediction-truth)**2)*meta['scale']**2)))
                np.savez_compressed(out/f'{ch}_s{seed}.npz',time=np.arange(start,len(raw)),prediction=prediction*meta['scale']+meta['mean'],truth=truth*meta['scale']+meta['mean'],score=score)
                pd.DataFrame(rows).to_csv(out/'calibration_thresholds.csv',index=False)
    print('Completed TRAIN-only calibration. No TEST data or labels accessed.')
if __name__=='__main__':main()
