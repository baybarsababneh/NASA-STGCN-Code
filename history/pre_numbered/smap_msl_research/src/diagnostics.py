"""Post-evaluation diagnostics; never used to choose a model or threshold."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,roc_auc_score
from common import ROOT,labels,intervals,event_metrics,point_metrics,prf,dump

COLORS=['#9aabb6','#197e90','#8159a4','#d0883b','#496cba','#343b44']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})

def bootstrap(f):
    rng=np.random.default_rng(1729);out=[]
    f=f[(f.protocol=='cal_s10_q0.995')&(f.seed==17)]
    for mission in ['SMAP','MSL']:
        g=f[(f.spacecraft==mission)&(f.model=='GCN')].sort_values('channel')
        for other in ['no_edges','persistence','LSTM_small','MLP']:
            b=f[(f.spacecraft==mission)&(f.model==other)].sort_values('channel')
            assert list(g.channel)==list(b.channel)
            for level in ['event','point']:
                a=g[[level+'_'+c for c in ['tp','fp','fn']]].to_numpy();z=b[[level+'_'+c for c in ['tp','fp','fn']]].to_numpy()
                diffs=[]
                for _ in range(2000):
                    ix=rng.integers(0,len(a),len(a));diffs.append(prf(*a[ix].sum(0))['f05']-prf(*z[ix].sum(0))['f05'])
                out.append(dict(spacecraft=mission,comparison='GCN minus '+other,metric=level+'_f05',difference=prf(*a.sum(0))['f05']-prf(*z.sum(0))['f05'],lower95=np.quantile(diffs,.025),upper95=np.quantile(diffs,.975)))
    pd.DataFrame(out).to_csv(ROOT/'results/paired_bootstrap.csv',index=False)

def label_sensitivity(f):
    out=[]
    for policy in ['first','last','union']:
        lab={r['channel']:r for r in labels(policy)}
        for model in ['GCN','no_edges','persistence','MLP','LSTM_small','archived_LSTM']:
            for protocol in ['cal_s10_q0.995','telemanom_compatibility']:
                if model=='archived_LSTM' and protocol!='telemanom_compatibility':continue
                for mission in ['SMAP','MSL']:
                    totals=np.zeros(3,int);pt=np.zeros(3,int)
                    for ch,r in lab.items():
                        if r['spacecraft']!=mission:continue
                        ar=np.load(ROOT/'results/predictions'/model/f'{ch}_s17.npz');t=ar['times'];pred=ar['positive'] if protocol.startswith('cal') else ar['legacy_positive']
                        seq=[(a+int(t[0]),b+int(t[0])) for a,b in intervals(pred)]
                        em=event_metrics(r['events'],seq);totals+=[em[k] for k in ['tp','fp','fn']]
                        truth=np.zeros(len(t),bool)
                        for a,b in r['events']:truth|=(t>=a)&(t<=b)
                        pm=point_metrics(truth,pred);pt+=[pm[k] for k in ['tp','fp','fn']]
                    out.append(dict(policy=policy,spacecraft=mission,model=model,protocol=protocol,**{'event_'+k:v for k,v in prf(*totals).items()},**{'point_'+k:v for k,v in prf(*pt).items()}))
    pd.DataFrame(out).to_csv(ROOT/'results/label_sensitivity.csv',index=False)

def pooled_scores():
    out=[]
    for mission in ['SMAP','MSL']:
        channels=[r['channel'] for r in labels() if r['spacecraft']==mission]
        for model in ['GCN','no_edges','persistence','MLP','LSTM_small']:
            truth=[];scores=[]
            for ch in channels:
                z=np.load(ROOT/'results/predictions'/model/f'{ch}_s17.npz');truth.extend(z['truth']);scores.extend(z['score_ratio'])
            out.append(dict(spacecraft=mission,model=model,pooled_ap=average_precision_score(truth,scores),pooled_auroc=roc_auc_score(truth,scores)))
    pd.DataFrame(out).to_csv(ROOT/'results/pooled_ranking.csv',index=False)

def plots(f,summary):
    models=['persistence','no_edges','MLP','LSTM_small','GCN'];names=['Persistence','No edges','MLP','Small LSTM','ST-GCN']
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),sharey=True)
    for row,mission in enumerate(['SMAP','MSL']):
        sub=summary[(summary.spacecraft==mission)&(summary.protocol=='cal_s10_q0.995')&(summary.seed==17)].set_index('model')
        for col,level in enumerate(['event','point']):
            ax=axes[row,col];x=np.arange(len(models))
            for j,metric in enumerate(['precision','recall','f05']):ax.bar(x+(j-1)*.24,[sub.loc[m,level+'_'+metric] for m in models],.24,label=metric if metric!='f05' else 'F0.5',color=COLORS[j])
            ax.set_xticks(x,names,rotation=15);ax.set_ylim(0,1.05);ax.set_title(mission+' | '+level+' metrics');ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(loc='upper left',ncols=3,fontsize=8)
    fig.suptitle('Fixed calibration threshold: no test-label tuning',fontsize=15);fig.tight_layout();fig.savefig(ROOT/'figures/primary_comparison.png');plt.close(fig)
    v=pd.read_csv(ROOT/'results/validation_summary.csv');configs=['gcn_l32_h16','gcn_l64_h16','gcn_l128_h16','gcn_l64_h32','gcn_l64_top4']
    fig,axes=plt.subplots(1,2,figsize=(10,3.4))
    for ax,mission in zip(axes,['SMAP','MSL']):
        s=v[v.spacecraft==mission].set_index('configuration');vals=[s.loc[c,'median_val_mse'] for c in configs]
        ax.bar(range(5),vals,color=['#197e90' if val==min(vals) else '#bdcbd4' for val in vals]);ax.set_xticks(range(5),['L32/H16','L64/H16','L128/H16','L64/H32','L64/top4'],rotation=20);ax.set_title(mission);ax.set_ylabel('Median validation MSE')
    fig.suptitle('Graph selection uses training validation only');fig.tight_layout();fig.savefig(ROOT/'figures/validation_selection.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),sharey=True)
    for ax,mission in zip(axes,['SMAP','MSL']):
        for j,model in enumerate(['no_edges','GCN']):
            s=summary[(summary.spacecraft==mission)&(summary.model==model)&(summary.protocol=='cal_s10_q0.995')].sort_values('seed')
            ax.scatter(np.full(len(s),j)+np.linspace(-.07,.07,len(s)),s.event_f05,s=60,color=COLORS[j]);ax.plot([j-.15,j+.15],[s.event_f05.mean()]*2,color='black')
        ax.set_xticks([0,1],['Matched no edges','ST-GCN']);ax.set_title(mission);ax.set_ylim(0,1);ax.set_ylabel('Event F0.5');ax.grid(axis='y',alpha=.2)
    fig.suptitle('All three seeds retained; black line = mean');fig.tight_layout();fig.savefig(ROOT/'figures/seed_ablation.png');plt.close(fig)
    for ch in ['P-1','M-1']:
        fig,axes=plt.subplots(3,1,figsize=(11,6),sharex=True,gridspec_kw={'height_ratios':[2,1.6,1]});lab=next(r for r in labels() if r['channel']==ch)
        z=np.load(ROOT/'results/predictions/GCN'/f'{ch}_s17.npz');t=z['times']
        axes[0].plot(t,z['observed'],lw=.7,label='Observed',color='#343b44');axes[0].plot(t,z['prediction'],lw=.7,label='Forecast',alpha=.7,color='#197e90');axes[0].legend(ncols=2,fontsize=8);axes[0].set_ylabel('Telemetry')
        axes[1].plot(t,z['score_ratio'],lw=.8,color='#197e90');axes[1].axhline(1,color='#b54d3d',ls='--');axes[1].set_yscale('symlog',linthresh=1);axes[1].set_ylabel('Score / threshold')
        axes[2].fill_between(t,0,z['positive'],step='mid',alpha=.7,color='#197e90');axes[2].fill_between(t,1.2,z['legacy_positive'].astype(float)+1.2,step='mid',alpha=.7,color='#8159a4');axes[2].set_yticks([.5,1.7],['Calibrated','Legacy']);axes[2].set_ylim(-.1,2.3);axes[2].set_xlabel('Original test-array row (nominal minute steps)')
        for ax in axes:
            for a,b in lab['events']:ax.axvspan(a,b,color='#efbd62',alpha=.22)
        fig.suptitle(f'{lab["spacecraft"]} {ch}: shaded regions are real labelled anomalies');fig.tight_layout();fig.savefig(ROOT/f'figures/trace_{ch}.png');plt.close(fig)

def main():
    f=pd.read_csv(ROOT/'results/per_channel.csv');s=pd.read_csv(ROOT/'results/summary.csv')
    bootstrap(f);label_sensitivity(f);pooled_scores();plots(f,s)
    primary=f[(f.protocol=='cal_s10_q0.995')&(f.seed==17)&(f.model=='GCN')].sort_values('event_fp',ascending=False)
    primary.to_csv(ROOT/'results/gcn_diagnostics_ranked.csv',index=False)
    print('Diagnostics and five figures saved')

if __name__=='__main__':main()
