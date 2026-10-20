"""Run the fully screened MSL stage while SMAP's longer screen finishes.

Applies the identical recorded rules. The global queues reuse these checkpoints.
"""
import json,subprocess,sys
import numpy as np
import pandas as pd
from round2_train import ROOT,dump
from select_stages import reference,score
p=json.loads((ROOT/'configs/screen_protocol.json').read_text());mp=p['missions']['MSL'];d=pd.read_csv(ROOT/'results/screen_runs.csv');d=d[d.spacecraft=='MSL'];assert len(d)==42
base=reference();scores={c['name']:score(d[d.configuration==c['name']],base) for c in mp['candidates']};inc=mp['candidates'][0];alt=min(mp['candidates'][1:],key=lambda c:scores[c['name']]);assert scores[alt['name']]<=scores[inc['name']]+np.log(.98)
ch=pd.read_csv(ROOT/'inputs/channels.csv');channels=ch[ch.spacecraft=='MSL'].channel.tolist();jobs=[[c,'MSL',alt,17] for c in channels];dump(ROOT/'configs/msl_early_jobs.json',jobs)
subprocess.run([sys.executable,str(ROOT/'src/run_jobs.py'),'msl_early_jobs.json','--workers','3'],check=True)
a=pd.read_csv(ROOT/'results/msl_early_runs.csv');i=pd.read_csv(ROOT/'results/incumbent_runs.csv');i=i[i.spacecraft=='MSL'];ratios={}
for sub in ['all','nonscreen']:
 aa=a;ii=i
 if sub=='nonscreen':aa=aa[~aa.channel.isin(mp['screen_channels'])];ii=ii[~ii.channel.isin(mp['screen_channels'])]
 ratios[sub]=float(np.exp(score(aa,base)-score(ii,base)))
kept=[inc]+([alt] if all(r<=.98 for r in ratios.values()) else [])
dump(ROOT/'configs/msl_early_decision.json',dict(ratios=ratios,retained=[c['name'] for c in kept]))
repeat=[[c,'MSL',cfg,s] for cfg in kept for s in [29,43] for c in channels];dump(ROOT/'configs/msl_early_repeat_jobs.json',repeat)
subprocess.run([sys.executable,str(ROOT/'src/run_jobs.py'),'msl_early_repeat_jobs.json','--workers','3'],check=True)
