"""Rebuild tables, figures, and configuration lock from validation-only results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from round2_train import ROOT,dump
from select_stages import score,reference

def main():
    dfs=[]
    for p in sorted((ROOT/'results').glob('*_runs.csv')):
        if p.name in ['all_runs.csv']:continue
        d=pd.read_csv(p);d['stage']=p.stem.replace('_runs','');dfs.append(d)
    allr=pd.concat(dfs).drop_duplicates(['configuration','channel','seed'])
    allr.sort_values(['spacecraft','configuration','seed','channel']).to_csv(ROOT/'results/all_runs.csv',index=False)
    dec=json.loads((ROOT/'configs/confirmation_decisions.json').read_text());base=reference();lock={};scores=[];summ=[];channels=[];budget=[]
    old=pd.read_csv(ROOT/'inputs/original_validation_runs.csv');weighted=pd.read_csv(ROOT/'inputs/weighted_training.csv')
    for mission,mp in dec.items():
        candidates=[mp['incumbent']]+([mp['challenger']] if mp['challenger'] else [])
        vals={}
        for cfg in candidates:
            d=allr[allr.configuration==cfg['name']];assert set(d.seed)=={17,29,43}
            assert d.groupby('seed').channel.nunique().eq(27 if mission=='MSL' else 54).all()
            ss=[score(d[d.seed==s],base) for s in [17,29,43]];vals[cfg['name']]=np.mean(ss)
            scores.append(dict(spacecraft=mission,configuration=cfg['name'],score_mean=np.mean(ss),score_sd=np.std(ss,ddof=1),score_seed17=ss[0],score_seed29=ss[1],score_seed43=ss[2]))
        winner=min(candidates,key=lambda c:vals[c['name']]);lock[mission]=winner
        fixed=allr[allr.configuration==mission+'_fixed80'];fmean=fixed.groupby('channel').val_mse.mean()
        def add(name,d,epochs,seeds):
            g=d.groupby('channel').val_mse.mean();b=fmean.reindex(g.index);info=base.reindex(g.index)>1e-4;diff=b-g
            ratios=np.log((g[info]+1e-4)/(b[info]+1e-4)).clip(-np.log(4),np.log(4))
            rng=np.random.default_rng(2026);v=ratios.to_numpy();boot=np.exp(np.mean(rng.choice(v,(2000,len(v))),axis=1))
            rec=dict(spacecraft=mission,model=name,budget=epochs,seeds=seeds,channels=len(g),informative_channels=int(info.sum()),macro_mse=float(g.mean()),median_mse=float(g.median()),informative_macro_mse=float(g[info].mean()),median_pct_gain_vs_fixed80=float((100*diff[info]/b[info]).median()),robust_ratio_vs_fixed80=float(np.exp(ratios.mean())),ratio_bootstrap_low=float(np.quantile(boot,.025)),ratio_bootstrap_high=float(np.quantile(boot,.975)),wins=int((diff>1e-7).sum()),losses=int((diff< -1e-7).sum()),ties=int((diff.abs()<=1e-7).sum()))
            summ.append(rec)
        add('fixed_GCN_80',fixed,80,'17,29,43')
        for cfg in candidates:
            d=allr[allr.configuration==cfg['name']];add(cfg['name'],d,80,'17,29,43')
            for s,ds in d.groupby('seed'):
                g=ds.set_index('channel');b=g.best40_mse;mask=b>1e-4
                budget.append(dict(spacecraft=mission,configuration=cfg['name'],seed=s,runs=len(ds),improved_after40=int(((b-g.val_mse)>1e-7).sum()),median_pct_gain_after40=float((100*(b[mask]-g.val_mse[mask])/b[mask]).median()),best_epoch80=int(g.ceiling_hit.sum()),trained_to80=int((g.epochs==80).sum()),median_epoch=float(g.best_epoch.median()),mean_parameters=float(g.parameters.mean()),sum_run_wall_seconds=float(g.seconds.sum())))
        win=allr[allr.configuration==winner['name']]
        wmean=win.groupby('channel').agg(tcn_mse=('val_mse','mean'),tcn_seed_sd=('val_mse','std'),tcn_min=('val_mse','min'),tcn_max=('val_mse','max'),raw_mse=('val_raw_mse','mean'),persistence_mse=('persistence_val_mse','mean'),parameters=('parameters','first'),best_epoch=('best_epoch','mean'),ceiling_hits=('ceiling_hit','sum'))
        wmean['fixed80_mse']=fmean;wmean['oldfixed40_mse']=base;wmean['informative']=wmean.oldfixed40_mse>1e-4;wmean['absolute_gain']=wmean.fixed80_mse-wmean.tcn_mse;wmean['pct_gain']=np.where(wmean.informative,100*wmean.absolute_gain/wmean.fixed80_mse,np.nan);wmean['spacecraft']=mission;wmean['configuration']=winner['name'];wmean['family']=wmean.index.str.split('-').str[0];channels.append(wmean.reset_index())
        persistence=win[win.seed==17][['channel','persistence_val_mse']].rename(columns={'persistence_val_mse':'val_mse'});add('persistence',persistence,0,'none')
        oldcfg='gcn_l64_top4' if mission=='MSL' else 'gcn_l64_h32'
        for cfg,name in [(oldcfg,'archived_fixed_GCN_40'),('lstm_l64','archived_compact_LSTM_40'),('mlp_l64','archived_MLP_40')]:
            d=old[(old.spacecraft==mission)&(old.configuration==cfg)];add(name,d,40,','.join(map(str,sorted(d.seed.unique()))))
        wd=weighted[weighted.spacecraft==mission][['channel','weighted_val_mse']].rename(columns={'weighted_val_mse':'val_mse'});add('archived_learned_edges_40',wd,40,'17')
    c=pd.concat(channels);c.to_csv(ROOT/'results/channel_comparison.csv',index=False)
    family=[]
    for (m,f),d in c.groupby(['spacecraft','family']):
        i=d[d.informative];family.append(dict(spacecraft=m,family=f,channels=len(d),informative=len(i),tcn_macro_mse=d.tcn_mse.mean(),fixed80_macro_mse=d.fixed80_mse.mean(),median_pct_gain=i.pct_gain.median(),absolute_gain=d.absolute_gain.mean(),wins=(d.absolute_gain>1e-7).sum(),losses=(d.absolute_gain< -1e-7).sum(),ties=(d.absolute_gain.abs()<=1e-7).sum()))
    pd.DataFrame(family).to_csv(ROOT/'results/family_comparison.csv',index=False)
    pd.DataFrame(summ).to_csv(ROOT/'results/mission_comparison.csv',index=False);pd.DataFrame(scores).to_csv(ROOT/'results/seed_selection.csv',index=False);pd.DataFrame(budget).to_csv(ROOT/'results/epoch_budget.csv',index=False)
    seedrows=[]
    for m,cfg in lock.items():
        for s in [17,29,43]:
            d=allr[(allr.configuration==cfg['name'])&(allr.seed==s)].set_index('channel');f=allr[(allr.configuration==m+'_fixed80')&(allr.seed==s)].set_index('channel').val_mse
            info=base.reindex(d.index)>1e-4;r=np.log((d.val_mse[info]+1e-4)/(f[info]+1e-4)).clip(-np.log(4),np.log(4));seedrows.append(dict(spacecraft=m,seed=s,tcn_macro_mse=d.val_mse.mean(),fixed_macro_mse=f.mean(),robust_ratio=np.exp(r.mean()),median_pct_gain=(100*(f[info]-d.val_mse[info])/f[info]).median()))
    pd.DataFrame(seedrows).to_csv(ROOT/'results/seed_robustness.csv',index=False)
    dump(ROOT/'configs/locked_tcn.json',dict(schema='round2_validation_lock_v1',missions=lock,seeds=[17,29,43],fit_fraction=.6,validation_end=.8,training_start=64,batch_size=256,weight_decay=.0001,patience=6,min_delta=1e-7,gradient_clip=1.,dropout=0.,selection='configs/screen_protocol.json; results/seed_selection.csv',calibration_accessed=False,test_accessed=False,seed_policy='report all three seeds; no best-seed selection; no untested ensemble claim'))
    costs=allr.groupby(['spacecraft','configuration']).agg(runs=('channel','size'),sum_run_wall_seconds=('seconds','sum'),median_run_seconds=('seconds','median'),min_parameters=('parameters','min'),max_parameters=('parameters','max'),median_best_epoch=('best_epoch','median'),median_epochs=('epochs','median'),best_epoch80=('ceiling_hit','sum'),early_stopped=('early_stopped','sum')).reset_index();costs.to_csv(ROOT/'results/computational_cost.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(10,4.5))
    for ax,(m,d) in zip(axs,c.groupby('spacecraft')):
        z=d[d.informative];ax.scatter(z.fixed80_mse,z.tcn_mse,s=25,alpha=.8);lo=min(z.fixed80_mse.min(),z.tcn_mse.min())*.7;hi=max(z.fixed80_mse.max(),z.tcn_mse.max())*1.5;ax.plot([lo,hi],[lo,hi],color='gray',ls='--');ax.set(xscale='log',yscale='log',xlabel='Fixed GCN: mean validation MSE',ylabel='Locked TCN: mean validation MSE',title=m);ax.grid(alpha=.15)
    fig.suptitle('Matched 80-epoch limits, seeds 17/29/43; informative channels');fig.tight_layout();fig.savefig(ROOT/'figures/validation_scatter.png',dpi=180);plt.close(fig)
    fam=pd.DataFrame(family);fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    for ax,(m,d) in zip(axs,fam.groupby('spacecraft')):
        z=d[d.informative>0];ax.bar(z.family,z.median_pct_gain,color=np.where(z.median_pct_gain>=0,'#287c8e','#bb5849'));ax.axhline(0,color='gray');ax.set(title=m,ylabel='Median MSE reduction vs fixed GCN (%)',xlabel='Channel prefix (not verified subsystem)');ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(ROOT/'figures/family_gains.png',dpi=180);plt.close(fig)
    print(pd.DataFrame(summ).to_string(index=False));print('LOCK',json.dumps(lock))
if __name__=='__main__':main()
