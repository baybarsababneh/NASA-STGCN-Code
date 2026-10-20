"""Descriptive four-cell study. There is deliberately no ranking/promotion function."""
import io,json
from pathlib import Path
import numpy as np
import pandas as pd
from detector_metrics_v2 import ewma,higher_threshold,burden

def records_for_trace(metadata,arrays,forecaster):
    job=metadata['job'];r=metadata['reference_result'];n=r['num_values'];start=r['stop_end'];end=r['assess_end']
    middle=int(np.floor(n*(.55 if job['fold']=='R1' else .65)))
    t=arrays['target_index'];y=arrays['target'];pred=arrays['prediction'] if forecaster=='gcn' else arrays['persistence']
    error=np.abs(y.astype(np.float64)-pred.astype(np.float64));cal=(t>=start)&(t<middle);assess=(t>=middle)&(t<end)
    assert cal.sum()==middle-start and assess.sum()==end-middle
    out=[]
    for span in [1,10]:
        score=ewma(error,span)  # Initialised at first stopping target, continuous thereafter.
        for q in [.99,.995]:
            th=higher_threshold(score[cal],q);flags=score>th['threshold'];k=np.flatnonzero(assess)[0]
            b=burden(flags[assess],previous_flag=flags[k-1]);calscore=score[cal]
            # Chronological half-calibration sensitivity; diagnostic only, not new thresholds.
            half=len(calscore)//2
            t1=higher_threshold(calscore[:half],q)['threshold'];t2=higher_threshold(calscore[half:],q)['threshold']
            out.append(dict(mission=job['mission'],channel=job['channel'],fold=job['fold'],
                seed=job['seed'] if forecaster=='gcn' else 0,forecaster=forecaster,span=span,q=q,
                cal_start=start,cal_end=middle,assess_start=middle,assess_end=end,
                score_initialisation_index=int(t[0]),**{k:v for k,v in th.items() if k!='q'},**b,
                calibration_median=float(np.median(calscore)),assessment_median=float(np.median(score[assess])),
                calibration_max=float(calscore.max()),assessment_max=float(score[assess].max()),
                threshold_first_cal_half=t1,threshold_second_cal_half=t2,
                threshold_half_abs_difference=abs(t2-t1),nominal_status='presumed_nominal_not_certified',
                selector_eligible=False))
    return out

def summarise(rows,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);df=pd.DataFrame(rows)
    assert len(df)==288 and not df.duplicated(['mission','channel','fold','seed','forecaster','span','q']).any()
    df.to_csv(out/'nominal_stream_diagnostics.csv',index=False)
    keys=['mission','forecaster','span','q'];metrics=['occupancy','alarm_starts_per_1000','longest_run','threshold_half_abs_difference']
    # Preserve within-channel seeds and origins as repeated observations, not independent units.
    channels=df.groupby(keys+['channel'])[metrics].mean().reset_index();channels.to_csv(out/'nominal_channel_summary.csv',index=False)
    mission=channels.groupby(keys)[metrics].agg(['mean','median','min','max']);mission.to_csv(out/'nominal_mission_summary.csv')
    df.groupby(keys+['fold','seed'])[metrics].mean().to_csv(out/'nominal_origin_seed_summary.csv')
    resolution=df.groupby(keys).agg(streams=('channel','size'),min_cal=('n_cal','min'),max_cal=('n_cal','max'),fraction_at_max=('at_sample_max','mean')).reset_index();resolution.to_csv(out/'quantile_resolution.csv',index=False)
    same=[]
    for key,g in df.groupby(['mission','channel','fold','seed','forecaster','span']):
        a=g[g.q==.99].iloc[0];b=g[g.q==.995].iloc[0]
        same.append(dict(zip(['mission','channel','fold','seed','forecaster','span'],key),same_order_rank=bool(a.rank_zero_based==b.rank_zero_based),same_threshold=bool(a.threshold==b.threshold)))
    pd.DataFrame(same).to_csv(out/'quantile_pair_identifiability.csv',index=False)
    contrasts=[]
    # Only two predeclared paired smoothing contrasts at each q/forecaster/mission.
    # Six channels per mission: descriptive intervals, not significance gates.
    rng=np.random.default_rng(8008)
    for (m,f,q),g in channels.groupby(['mission','forecaster','q']):
        a=g[g.span==1].set_index('channel');b=g[g.span==10].set_index('channel');assert set(a.index)==set(b.index)
        for metric in ['occupancy','alarm_starts_per_1000']:
            d=(b.loc[a.index,metric]-a[metric]).to_numpy();draws=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(axis=1)
            contrasts.append(dict(mission=m,forecaster=f,q=q,metric=metric,contrast='span10_minus_raw',n_channels=len(d),
                mean_difference=float(d.mean()),lower95=float(np.quantile(draws,.025)),upper95=float(np.quantile(draws,.975)),
                min_leave_one_out=float(min(np.delete(d,i).mean() for i in range(len(d)))),max_leave_one_out=float(max(np.delete(d,i).mean() for i in range(len(d)))),
                interpretation='descriptive_conditional_no_pvalue_no_promotion'))
    pd.DataFrame(contrasts).to_csv(out/'paired_descriptive_intervals.csv',index=False)
    # Compact standalone SVG, no extra rendering dependency. All four policies shown equally.
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="450" viewBox="0 0 900 450"><rect width="900" height="450" fill="white"/><text x="20" y="25" font-family="sans-serif" font-size="16">TRAIN-only nominal occupancy: each dot is one channel (seeds/origins averaged)</text>']
    for mi,m in enumerate(['MSL','SMAP']):
        x0=50+mi*430;parts.append(f'<text x="{x0}" y="55" font-family="sans-serif">{m}; no anomaly recall measured</text>')
        for i,(span,q) in enumerate([(1,.99),(1,.995),(10,.99),(10,.995)]):
            x=x0+40+i*95
            parts.append(f'<line x1="{x}" x2="{x}" y1="80" y2="380" stroke="#ddd"/><text x="{x-25}" y="405" font-size="11">s{span} q{q}</text>')
            for f,color,shift in [('gcn','#1764ab',-8),('persistence','#dd7d1d',8)]:
                vals=channels[(channels.mission==m)&(channels.forecaster==f)&(channels.span==span)&(channels.q==q)].occupancy
                for value in vals:parts.append(f'<circle cx="{x+shift}" cy="{380-300*value}" r="3" fill="{color}" opacity=".7"/>')
        parts.append(f'<text x="{x0}" y="78" font-size="10">100%</text><text x="{x0}" y="385" font-size="10">0%</text>')
    parts.append('<text x="50" y="435" font-size="12" fill="#1764ab">Blue: fixed GCN</text><text x="230" y="435" font-size="12" fill="#dd7d1d">Orange: persistence</text></svg>')
    (out/'nominal_occupancy.svg').write_text(''.join(parts))
