"""Reuse upstream Telemanom thresholding; explicitly retrospective compatibility.

The unmodified upstream source is retained under vendor. The single runtime
compatibility patch casts np.delete indices to int for modern NumPy. Initialization
here avoids upstream writes and permits inspection of its smoothing behaviour.
"""
import types, re
import numpy as np
import pandas as pd
from common import ROOT,ARCHIVE,load,labels,dump,event_metrics,prf

source=(ROOT/'vendor/telemanom/errors.py').read_text()
assert 'np.delete(E_seq, i_to_remove, axis=0)' in source
source=source.replace('np.delete(E_seq, i_to_remove, axis=0)','np.delete(E_seq, i_to_remove.astype(int), axis=0)')
module=types.ModuleType('telemanom_errors_compat')
exec(compile(source,'vendor/telemanom/errors.py','exec'),module.__dict__)

def detect(ch,truth,pred,offset=250,horizon_truth=None):
    config=types.SimpleNamespace(batch_size=70,window_size=30,smoothing_perc=.05,error_buffer=100,l_s=250,p=.13)
    y=truth[:,None] if horizon_truth is None else horizon_truth
    channel=types.SimpleNamespace(id=ch,y_test=y,y_hat=pred)
    obj=module.Errors.__new__(module.Errors)
    obj.config=config;obj.window_size=30;obj.n_windows=int((len(y)-2100)/70)
    obj.i_anom=np.array([]);obj.E_seq=[];obj.anom_scores=[]
    obj.e=np.abs(pred-truth);obj.e_s=pd.Series(obj.e).ewm(span=105).mean().to_numpy()
    if ch!='C-2':obj.e_s[:250]=np.mean(obj.e_s[:500])
    # Original smoothing uses future errors during startup and a channel-specific
    # exception. Kept ONLY in the compatibility track, never in primary scores.
    # Exact early exit: compare_to_epsilon rejects every window when the global
    # maximum is <= .05. Avoid expensive candidate expansion on quiet channels.
    if obj.e_s.max()>.05:
        with np.errstate(divide='ignore',invalid='ignore'):
            obj.process_batches(channel)
    return [(a-250+offset,b-250+offset) for a,b in obj.E_seq],obj.e_s

def code_event_counts(truth,pred):
    # Literal upstream detector semantics: credits only first label a prediction hits.
    matched=set();fp=0
    for a,b in pred:
        hits=[i for i,(c,d) in enumerate(truth) if a<=d and c<=b]
        if hits:matched.add(hits[0])
        else:fp+=1
    return prf(len(matched),fp,len(truth)-len(matched))

def audit():
    rawlog=(ARCHIVE/'params.log').read_text();logged=[]
    pattern=r'Stream # (\d+): ([A-Z]-\d+).*?INFO TP: (\d+)  FP: (\d+)  FN: (\d+)'
    source_rows=labels('rows')
    for row in re.findall(pattern,rawlog,re.S):
        i,ch,tp,fp,fn=row;r=source_rows[int(i)-1]
        assert r['channel']==ch
        logged.append(dict(row=int(i),channel=ch,spacecraft=r['spacecraft'],**prf(int(tp),int(fp),int(fn))))
    pd.DataFrame(logged).to_csv(ROOT/'audit/archived_log_counts.csv',index=False)
    results=[];predictions={}
    for r in labels():
        ch=r['channel'];raw=load(ch,'test');pred=np.load(ARCHIVE/'y_hat'/f'{ch}.npy')
        assert len(pred)==len(raw)-260
        yh=np.lib.stride_tricks.sliding_window_view(raw[:,0],10)[250:250+len(pred)]
        seq,sm=detect(ch,raw[250:-10,0],pred,horizon_truth=yh)
        original=np.load(ARCHIVE/'smoothed_errors'/f'{ch}.npy')
        results.append(dict(channel=ch,spacecraft=r['spacecraft'],smoothing_max_abs_diff=float(np.max(np.abs(sm-original))),**event_metrics(r['events'],seq)))
        predictions[ch]=seq
        print('archive',ch,flush=True)
    pd.DataFrame(results).to_csv(ROOT/'audit/archived_replay_unique.csv',index=False)
    dump(ROOT/'audit/archived_replay_intervals.json',predictions)
    rowresults=[]
    for i,r in enumerate(source_rows):
        seq=predictions[r['channel']]
        rowresults.append(dict(row=i+1,channel=r['channel'],spacecraft=r['spacecraft'],**code_event_counts(r['events'],seq)))
    pd.DataFrame(rowresults).to_csv(ROOT/'audit/archived_replay_rows.csv',index=False)
    summary=[]
    for name,rows in [('archived_log',logged),('replay_unique_corrected',results),('replay_rows_upstream_scoring',rowresults)]:
        f=pd.DataFrame(rows)
        for mission,g in f.groupby('spacecraft'):
            summary.append(dict(source=name,spacecraft=mission,**prf(*g[['tp','fp','fn']].sum().tolist())))
    pd.DataFrame(summary).to_csv(ROOT/'audit/legacy_summary.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False))

if __name__=='__main__':audit()
