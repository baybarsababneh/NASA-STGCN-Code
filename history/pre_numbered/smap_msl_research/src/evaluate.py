"""Frozen evaluation with raw-point and paper-style interval metrics.

Primary thresholds use only held-out TRAIN calibration residuals. Compatibility
uses original Telemanom, including its retrospective startup and buffering.
"""
import json, argparse, warnings
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor,as_completed
from sklearn.metrics import average_precision_score,roc_auc_score
from common import ROOT,PROTOCOL,ARCHIVE,load,labels,dump,intervals,event_metrics,point_metrics,prf,smooth
from models import make
from train import predict
from legacy import detect

def predictions(ch,model_name,seed):
    raw=load(ch,'test');t=np.arange(250,len(raw)-10);truth=raw[t,0]
    if model_name=='archived_lstm':return t,truth,np.load(ARCHIVE/'y_hat'/f'{ch}.npy'),None,None
    tr=load(ch,'train');end=int(len(tr)*.8)
    if model_name=='persistence':
        cal=np.abs(tr[end:,0]-tr[end-1:-1,0]);return t,truth,raw[t-1,0],cal,dict(scale=float(tr[:int(len(tr)*.6),0].std()))
    path=ROOT/'checkpoints'/model_name/f'{ch}_s{seed}.pt'
    state=torch.load(path,weights_only=False,map_location='cpu');meta=state['metadata'];cfg=state['config'];L=cfg['lookback']
    x=raw[:,meta['active']].copy();x[:,0]=(x[:,0]-meta['mean'])/meta['scale']
    X=torch.tensor(np.lib.stride_tricks.sliding_window_view(x,L,axis=0)[t-L].copy())
    model=make(cfg,x.shape[1],state['adjacency']);model.load_state_dict(state['state'])
    pred=predict(model,X)*meta['scale']+meta['mean']
    saved=np.load(path.with_suffix('.npz'));is_cal=saved['times']>=meta['validation_end']
    cal=np.abs(saved['pred'][is_cal]-saved['truth'][is_cal])
    return t,truth,pred,cal,meta

def evaluate_channel(r,alias,actual,seed):
    ch=r['channel'];mission=r['spacecraft'];t,y,pred,cal,meta=predictions(ch,actual,seed)
    mask=np.zeros(int(t[-1])+1,bool)
    for a,b in r['events']:mask[a:b+1]=True
    truth=mask[t];err=np.abs(y-pred)
    base=dict(channel=ch,spacecraft=mission,model=alias,configuration=actual,seed=seed,n=len(t),mae=float(err.mean()),rmse=float(np.mean(err**2)**.5))
    rows=[];arrays=dict(times=t,truth=truth,observed=y,prediction=pred,errors=err)
    if cal is not None:
        # Report predetermined sensitivity values, without choosing on test results.
        for span,q in [(10,.995),(10,.99),(10,.999),(30,.995)]:
            sm=smooth(err,span);cs=smooth(cal,span)
            # Remove short EWMA startup from the calibration distribution only.
            burn=min(span,len(cs)//4);threshold=float(np.quantile(cs[burn:],q))
            floor=max(1e-8,float(meta['scale'])*1e-6);threshold=max(threshold,floor)
            positive=sm>threshold;seq=[(a+int(t[0]),b+int(t[0])) for a,b in intervals(positive)]
            row=dict(base,protocol=f'cal_s{span}_q{q}',threshold=threshold,calibration_n=len(cs)-burn,calibration_false_positive_fraction=float(np.mean(cs[burn:]>threshold)))
            row.update({'event_'+k:v for k,v in event_metrics(r['events'],seq).items()})
            row.update({'point_'+k:v for k,v in point_metrics(truth,positive).items()})
            row['average_precision']=float(average_precision_score(truth,sm));row['auroc']=float(roc_auc_score(truth,sm)) if len(np.unique(truth))==2 else np.nan
            row['alarm_fraction']=float(positive.mean());row['predicted_events']=len(seq)
            delays=[int(np.flatnonzero(positive & (t>=a)&(t<=b))[0]+t[0]-a) for a,b in r['events'] if np.any(positive&(t>=a)&(t<=b))]
            row['median_detected_delay_steps']=float(np.median(delays)) if delays else np.nan
            rows.append(row)
            if span==10 and q==.995:arrays.update(score=sm,threshold=threshold,positive=positive,score_ratio=sm/threshold)
    # Evaluate only seed17 in expensive legacy track. Never use for selection.
    if seed==17:
        raw=load(ch,'test');horizons=np.lib.stride_tricks.sliding_window_view(raw[:,0],10)[250:250+len(t)]
        if actual=='archived_lstm' and (ROOT/'audit/archived_replay_intervals.json').exists():
            seq=json.loads((ROOT/'audit/archived_replay_intervals.json').read_text())[ch]
            sm=smooth(err,105)
            if ch!='C-2':sm[:250]=sm[:500].mean()
        else:
            seq,sm=detect(ch,y,pred,horizon_truth=horizons)
        positive=np.zeros(len(t),bool)
        for a,b in seq:positive[(t>=a)&(t<=b)]=True
        row=dict(base,protocol='telemanom_compatibility')
        row.update({'event_'+k:v for k,v in event_metrics(r['events'],seq).items()})
        row.update({'point_'+k:v for k,v in point_metrics(truth,positive).items()})
        row['average_precision']=float(average_precision_score(truth,sm));row['alarm_fraction']=float(positive.mean());row['predicted_events']=len(seq)
        rows.append(row);arrays.update(legacy_positive=positive,legacy_score=sm)
    out=ROOT/'results/predictions'/alias/f'{ch}_s{seed}.npz';out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**arrays)
    return rows

def summarize(frame):
    out=[]
    for (mission,model,seed,protocol),g in frame.groupby(['spacecraft','model','seed','protocol']):
        r=dict(spacecraft=mission,model=model,seed=seed,protocol=protocol,channels=len(g))
        for level in ['event','point']:
            r.update({level+'_'+k:v for k,v in prf(*g[[level+'_'+k for k in ['tp','fp','fn']]].sum().tolist()).items()})
        for col in ['average_precision','alarm_fraction','mae','rmse','median_detected_delay_steps']:
            if col in g:r['macro_'+col]=float(g[col].mean())
        r['false_events_per_1000_steps']=float(g.event_fp.sum()/g.n.sum()*1000)
        out.append(r)
    return pd.DataFrame(out)

def main():
    selected=json.loads((ROOT/'configs/frozen_selection.json').read_text())['selected']
    out=ROOT/'results/per_channel.csv';rows=pd.read_csv(out).to_dict('records') if out.exists() else []
    done={(r['channel'],r['model'],int(r['seed'])) for r in rows}
    jobs=[];pending=[]
    for r in labels():
        mission=r['spacecraft'];graph=selected[mission]['name']
        choices=[('persistence','persistence',[17]),('GCN',graph,[17,29,43]),('no_edges','matched_no_edges_'+mission,[17,29,43]),('MLP','mlp_l64',[17]),('LSTM_small','lstm_l64',[17]),('archived_LSTM','archived_lstm',[17])]
        for alias,actual,seeds in choices:
            for seed in seeds:
                if (r['channel'],alias,seed) in done:continue
                if actual not in ['persistence','archived_lstm']:
                    cp=ROOT/'checkpoints'/actual/f'{r["channel"]}_s{seed}.pt'
                    if not cp.exists() or not cp.with_suffix('.npz').exists():
                        pending.append((r['channel'],alias,seed));continue
                jobs.append((r,alias,actual,seed))
    dump(ROOT/'results/evaluation_pending.json',pending)
    print(f'Evaluating {len(jobs)} available jobs; {len(pending)} await training',flush=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(evaluate_channel,*job):job for job in jobs}
        for future in as_completed(futures):
            r,alias,actual,seed=futures[future]
            rows.extend(future.result());pd.DataFrame(rows).to_csv(out,index=False)
            print(r['channel'],alias,seed,flush=True)
    summary=summarize(pd.DataFrame(rows));summary.to_csv(ROOT/'results/summary.csv',index=False)
    print(summary[(summary.seed==17)&summary.protocol.isin(['cal_s10_q0.995','telemanom_compatibility'])].to_string(index=False))

if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning);main()
