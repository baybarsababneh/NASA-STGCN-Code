"""Predeclare screening without looking at any anomaly/test files."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from round2_train import ROOT,dump

def config(mission,name,**changes):
    c=dict(name=mission+'_'+name,kind='tcn_gcn',lookback=64,hidden=32 if mission=='SMAP' else 16,graph='star' if mission=='SMAP' else 'top4',kernel_size=3,dilations=[1,2],epochs=80,lr=.001)
    c.update(changes);return c

def main():
    f=pd.read_csv(ROOT/'inputs/tcn_training(1).csv');out={};jobs=[]
    for mission,g in f.groupby('spacecraft'):
        g=g[g.baseline_val_mse>1e-4].copy();g['family']=g.channel.str.split('-').str[0]
        families=['A','D','E','E','G','P'] if mission=='SMAP' else ['C','D','F','M','P','T']
        channels=[]
        for family in families:
            options=g[(g.family==family)&(~g.channel.isin(channels))].sort_values(['baseline_val_mse','channel'])
            if len(options):channels.append(options.iloc[len(options)//2].channel)
        candidates=[config(mission,'rf7_native'),config(mission,'rf7_alternate',hidden=16 if mission=='SMAP' else 32),config(mission,'rf7_h64',hidden=64),config(mission,'rf13_native',kernel_size=5),config(mission,'rf63_h16',hidden=16,dilations=[1,2,4,8,16]),config(mission,'rf127_h16',hidden=16,lookback=128,dilations=[1,2,4,8,16,32]),config(mission,'rf255_h16',hidden=16,lookback=250,dilations=[1,2,4,8,16,32,64])]
        out[mission]=dict(screen_channels=channels,candidates=candidates)
        jobs.extend([(ch,mission,c,17) for c in candidates for ch in channels])
    protocol=dict(missions=out,fit_fraction=.6,validation_end=.8,calibration='sealed',test='sealed',training_start=64,short_history='left-pad fit-standardized zero before raw array start',patience=6,min_delta=1e-7,batch_size=256,weight_decay=.0001,gradient_clip=1.,seeds=[17,29,43],selection_floor=1e-4,selection_score='mean clipped log((candidate MSE + 1e-4)/(fixed GCN MSE + 1e-4)), clipping at +/- log(4), on fixed-baseline MSE > 1e-4',screen_rule='retain RF7 native incumbent plus lowest-score alternative; require >=2% ratio improvement over incumbent to promote an alternative on screening',confirmation_rule='all channels seed17, retain alternative only if informative-channel score improves >=2% vs incumbent on nonscreen channels and full mission; then compare incumbent and retained alternative at seeds29,43 and choose using mean across all three seeds; never choose a seed',baseline_comparison='original 40-epoch fixed GCN/MLP/LSTM plus fresh matched 80-epoch fixed GCN; persistence from validation',learned_edges='no further tuning; optional validation-only reference if available')
    p=ROOT/'configs/screen_protocol.json'
    if p.exists():assert json.loads(p.read_text())==protocol
    else:dump(p,protocol)
    dump(ROOT/'configs/screen_jobs.json',jobs)
    print({m:v['screen_channels'] for m,v in out.items()},'jobs',len(jobs))

if __name__=='__main__':main()
