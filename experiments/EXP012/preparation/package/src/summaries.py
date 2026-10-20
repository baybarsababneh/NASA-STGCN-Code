"""Descriptive TRAIN-only summaries; never select models or detectors."""
import numpy as np

def forecast_rows(records):
    rows=[]
    for mission in ['MSL','SMAP']:
        for arm in ['gcn','removed','tcn','lstm','persistence']:
            rr=[r for r in records if r['mission']==mission and r['arm']==arm]
            channels=sorted({r['channel'] for r in rr});by_channel=[np.mean([r['validation_mse'] for r in rr if r['channel']==ch]) for ch in channels]
            row={'mission':mission,'arm':arm,'channels':len(channels),'model_or_stream_records':len(rr),
                 'mean_validation_MSE':float(np.mean(by_channel)),'median_channel_seedmean_MSE':float(np.median(by_channel)),
                 'parameters_mean':float(np.mean([r.get('parameters',0) for r in rr])),
                 'epochs_ran_median':float(np.median([r.get('epochs_ran',0) for r in rr])),
                 'hit_epoch_ceiling_records':sum(r.get('hit_epoch_ceiling',False) for r in rr),
                 'no_anomaly_performance_inference':True}
            rows.append(row)
    return rows

def nominal_rows(records):
    rows=[]
    for mission in ['MSL','SMAP']:
        for arm in ['gcn','removed','tcn','lstm','persistence']:
            for split in ['validation','calibration']:
                for detector in ['raw_q99','hundman_adapt_v1']:
                    rr=[r for r in records if (r['mission'],r['arm'],r['split'],r['detector'])==(mission,arm,split,detector)]
                    for seed in ([None] if arm=='persistence' else [17,29,43]):
                        r=[v for v in rr if v['seed']==seed];n=sum(v['samples'] for v in r);f=sum(v['flagged_samples'] for v in r);starts=sum(v['alarm_starts'] for v in r)
                        rows.append({'mission':mission,'arm':arm,'split':split,'detector':detector,'seed':seed,
                                     'channels':len(r),'samples':n,'flagged_samples':f,'alarm_starts':starts,
                                     'sample_weighted_nominal_occupancy':f/n,'alarm_starts_per_1000_samples':1000*starts/n,
                                     'no_output_short_streams':sum(v['detector_status']=='NO_OUTPUT_SHORT_STREAM' for v in r),
                                     'deployability_or_sensitivity_certified':False})
    return rows

def resolution_rows(records):
    rows=[]
    for mission in ['MSL','SMAP']:
        for arm in ['gcn','removed','tcn','lstm','persistence']:
            r=[v['threshold'] for v in records if v['mission']==mission and v['arm']==arm]
            rows.append({'mission':mission,'arm':arm,'thresholds':len(r),'n_cal_min':min(v['n_cal'] for v in r),
                         'n_cal_median':float(np.median([v['n_cal'] for v in r])),'n_cal_max':max(v['n_cal'] for v in r),
                         'fraction_at_sample_max':float(np.mean([v['at_sample_max'] for v in r])),
                         'median_unique_residuals':float(np.median([v['unique_scores'] for v in r])),
                         'median_rank_positions_above':float(np.median([v['rank_positions_above'] for v in r])),
                         'no_threshold_selection_from_this_table':True})
    return rows
