"""Freeze graph selection before opening test scores; run matched ablations/seeds."""
import json
import pandas as pd
from common import ROOT,PROTOCOL,dump,sha
from train import run

def main():
    p=ROOT/'results/training.csv';f=pd.read_csv(p)
    candidates=[c['name'] for c in PROTOCOL['graph_candidates']]
    inv=pd.read_csv(ROOT/'audit/data_inventory.csv').drop_duplicates('channel')
    for c in candidates:
        assert len(f[(f.configuration==c)&(f.seed==17)])==81,c
    summary=f[f.seed==17].groupby(['spacecraft','configuration']).agg(median_val_mse=('val_mse','median'),mean_val_mse=('val_mse','mean'),median_epochs=('best_epoch','median'),seconds=('seconds','sum')).reset_index()
    summary.to_csv(ROOT/'results/validation_summary.csv',index=False)
    selected={}
    for mission in ['SMAP','MSL']:
        s=summary[(summary.spacecraft==mission)&summary.configuration.isin(candidates)].sort_values(['median_val_mse','configuration'])
        selected[mission]=next(c for c in PROTOCOL['graph_candidates'] if c['name']==s.iloc[0].configuration)
    path=ROOT/'configs/frozen_selection.json'
    if path.exists():assert json.loads(path.read_text())['selected']==selected
    else:dump(path,dict(selected=selected,criterion=PROTOCOL['selection'],training_csv_sha256=sha(p),test_metrics_seen=False))
    print('FROZEN',selected,flush=True)
    repeat_path=ROOT/'results/repeats.csv'
    rows=pd.read_csv(repeat_path).to_dict('records') if repeat_path.exists() else []
    done={(r['channel'],r['configuration'],r['seed']) for r in rows}
    for mission,cfg in selected.items():
        control=dict(cfg,name='matched_no_edges_'+mission,graph='none')
        for config,seeds in [(cfg,[29,43]),(control,[17,29,43])]:
            for seed in seeds:
                for r in inv[inv.spacecraft==mission].sort_values('channel').itertuples():
                    if (r.channel,config['name'],seed) in done:continue
                    out=run(r.channel,config,seed);out['spacecraft']=mission;rows.append(out)
                    pd.DataFrame(rows).to_csv(repeat_path,index=False)
                    print(config['name'],r.channel,seed,round(out['val_mse'],5),flush=True)

if __name__=='__main__':main()
