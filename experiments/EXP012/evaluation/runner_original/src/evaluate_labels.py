"""Phase B. Read labels only after all phase-A exports are sealed and verified."""
import sys,os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1');sys.dont_write_bytecode=True
import argparse,json,csv,ast,collections,platform
from pathlib import Path
import numpy as np
from common import *
from export_scores import verify_scores
from metrics import channel,aggregate,count_vector,bootstrap
ROOT=Path(__file__).resolve().parents[1]
def labels(path,contract,stream_rows):
 if sha(path)!=contract['historical_labels_sha256']:raise RuntimeError('Label bytes differ from frozen historical CSV')
 with path.open() as f:rows=list(csv.DictReader(f))
 seen={};duplicates=[]
 for number,r in enumerate(rows):
  ch=r['chan_id']
  if ch in seen:
   if ch!='P-2':raise RuntimeError('Unexpected duplicate label channel '+ch)
   duplicates.append({'channel':ch,'kept_csv_row':seen[ch][0],'ignored_csv_row':number,'ignored_row':r});continue
  seen[ch]=(number,r)
 if set(seen)!=set(stream_rows):raise RuntimeError('Label/channel roster differs')
 events={};total=0
 for ch,(number,r) in seen.items():
  meta=stream_rows[ch]
  if r['spacecraft']!=meta['mission'] or int(r['num_values'])!=meta['original_TEST_rows']:raise RuntimeError('Label TEST shape/mission mismatch')
  seq=ast.literal_eval(r['anomaly_sequences']);ev=[]
  for x in seq:
   if len(x)!=2 or any(type(v) is not int for v in x) or not 0<=x[0]<=x[1]<meta['original_TEST_rows']:raise RuntimeError('Invalid inclusive event interval')
   ev.append((x[0],x[1]+1))
  events[ch]=ev;total+=len(ev)
 if total!=104:raise RuntimeError('Expected104 original events after first-P-2 policy')
 return events,{'CSV_rows':len(rows),'unique_channels':len(seen),'original_events':total,'duplicates':duplicates,'first_P2_policy':True}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scores',type=Path,required=True);ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--evaluate-locked-exports',action='store_true');a=ap.parse_args()
 if not a.evaluate_locked_exports:raise RuntimeError('Phase B requires explicit --evaluate-locked-exports')
 verify_manifest(ROOT,'RUNNER_MANIFEST.json');binding=json.loads((ROOT/'EVALUATION_BINDING.json').read_text());scores=a.scores.resolve();out=a.output.resolve()
 metrics_environment={'python':platform.python_version(),'numpy':np.__version__}
 if metrics_environment!=binding['label_metrics_environment']:raise RuntimeError('Pinned label metrics Python/NumPy differ')
 if out==scores or scores in out.parents:raise RuntimeError('Keep labelled reports separate from sealed scores')
 sf=verify_scores(scores)
 if sf['preparation_binding_sha256']!=sha(ROOT/'EVALUATION_BINDING.json') or sf['runner_manifest_sha256']!=sha(ROOT/'RUNNER_MANIFEST.json') or sf['contract']!=binding['evaluation_contract']:raise RuntimeError('Scoring/export contract differs')
 if sha(scores/'TEST_INPUT_FREEZE.json')!=sf['TEST_input_freeze_sha256']:raise RuntimeError('TEST input freeze changed')
 # Inspect all stream records without opening labels. The whole score manifest was already byte-verified.
 metadata={jid:json.loads((scores/'units'/jid/'record.json').read_text()) for jid in sf['roster']};by_channel={r['channel']:r for r in metadata.values()}
 if len(by_channel)!=81 or len(metadata)!=1053:raise RuntimeError('Incomplete sealed export')
 channels=binding['channels'];expected={f"{arm}__{r['channel']}__{s}" for r in channels for arm in ['gcn','removed','tcn','lstm'] for s in [17,29,43]}|{f"persistence__{r['channel']}" for r in channels}
 if set(metadata)!=expected or set(by_channel)!={r['channel'] for r in channels}:raise RuntimeError('Sealed channel/model roster differs')
 missions={r['channel']:r['mission'] for r in channels}
 for jid,r in metadata.items():
  correct_id=f"{r['arm']}__{r['channel']}"+(f"__{r['seed']}" if r['seed'] is not None else '')
  if jid!=r['job_id'] or jid!=correct_id or r['mission']!=missions[r['channel']]:raise RuntimeError('Invalid sealed stream identity')
 for r in metadata.values():
  peer=by_channel[r['channel']]
  if r['original_TEST_rows']!=peer['original_TEST_rows'] or r['TEST_array_sha256']!=peer['TEST_array_sha256']:raise RuntimeError('Mismatched channel support')
 identity={'version':binding['version'],'runner_manifest_sha256':sha(ROOT/'RUNNER_MANIFEST.json'),'score_export_freeze_sha256':sha(scores/'SCORE_EXPORT_FREEZE.json'),'labels_sha256':sf['contract']['historical_labels_sha256'],'preparation_binding_sha256':sha(ROOT/'EVALUATION_BINDING.json'),'metrics_environment':metrics_environment}
 out.mkdir(parents=True,exist_ok=True);freeze(out/'LABEL_EVALUATION_FREEZE.json',identity)
 if (out/'LABEL_EVALUATION_COMPLETE.json').exists():
  verify_manifest(out,'LABEL_EVALUATION_COMPLETE.json');print('EXISTING LABELLED EVALUATION VERIFIED; no recomputation');return
 with Lease(out):
  events,label_audit=labels(a.labels,sf['contract'],by_channel);channel_rows=[];event_rows=[];outside_rows=[];all_rows={}
  for jid in sf['roster']:
   meta=metadata[jid];s=loadnpz(scores/'units'/jid/'scores.npz')
   for detector,prefix in [('raw_q99','raw'),('hundman_adapt_v1','adaptive')]:
    r,er,outs=channel(events[meta['channel']],s['target_index'],s[prefix+'_flags'],s[prefix+'_issue_index'],s['scores_raw'],s['scores_ewma']);base={k:meta[k] for k in ['job_id','arm','channel','mission','seed']};base['detector']=detector
    public={**base,**r};channel_rows.append(public);r['_delays']=[v['delay'] for v in er if v['delay'] is not None];all_rows[(meta['arm'],meta['channel'],meta['seed'],detector)]={**base,**r}
    event_rows.extend({**base,'event_number':i,**v} for i,v in enumerate(er));outside_rows.extend({**base,**v} for v in outs)
  mission_seed=[];mission_means=[];contrasts=[]
  for mission in ['MSL','SMAP']:
   channels=sorted(ch for ch,r in by_channel.items() if r['mission']==mission)
   for detector in ['hundman_adapt_v1','raw_q99']:
    for arm in ['gcn','removed','tcn','lstm','persistence']:
     seeds=[None] if arm=='persistence' else [17,29,43];aggregates=[]
     for seed in seeds:
      records=[all_rows[(arm,ch,seed,detector)] for ch in channels];v=aggregate(records);mission_seed.append({'mission':mission,'detector':detector,'arm':arm,'seed':seed,**v});aggregates.append(v)
     metric_fields=['corrected_event_f0_5','hundman_overlap_f0_5','point_precision','point_recall','point_f1','point_f0_5','AP_raw_macro','AP_ewma_macro','nominal_occupancy','alarm_starts_per_1000_nominal','within_event_notification_rate']
     means={'mission':mission,'detector':detector,'arm':arm,'seed_records':len(seeds),'persistence_independent_stream':arm=='persistence'}
     for k in metric_fields:
      vals=np.array([v[k] for v in aggregates]);means[k]=float(vals.mean()) if np.isfinite(vals).all() else float('nan');means[k+'_defined_seed_records']=int(np.isfinite(vals).sum())
     mission_means.append(means)
    for control in ['removed','lstm','tcn','persistence']:
     ac=np.array([[count_vector(all_rows[('gcn',ch,s,detector)]) for s in [17,29,43]] for ch in channels]);seeds=[None] if control=='persistence' else [17,29,43];bc=np.array([[count_vector(all_rows[(control,ch,s,detector)]) for s in seeds] for ch in channels]);v=bootstrap(ac,bc,10000,20261005)
     contrasts.append({'mission':mission,'detector':detector,'contrast':'gcn_minus_'+control,'primary_model_contrast':control=='removed',**v})
  table(out/'channel_metrics.csv',channel_rows);table(out/'event_notifications.csv',event_rows,['job_id','arm','channel','mission','seed','detector','event_number','onset','end_exclusive','overlap_hit','first_notification','delay','within_event_notification','late_post_event_notification','fragmentation','left_truncated']);table(out/'events_outside_support.csv',outside_rows,['job_id','arm','channel','mission','seed','detector','onset','end_exclusive','evaluation_support']);table(out/'mission_seed_metrics.csv',mission_seed);table(out/'mission_metrics.csv',mission_means);write(out/'paired_channel_bootstrap.json',contrasts);write(out/'LABEL_ROSTER_AUDIT.json',label_audit)
  atomic(out/'README_RESULTS.md',b'# Locked retrospective NASA evaluation\n\nMission metrics average three seed-level pooled results; persistence is reported once. Primary event overlap is retrospective and separate from actual notification timing. Native scores are not pooled for AP; macro denominators are reported. TRAIN calibration and NASA TEST were historically exposed. Controlled LSTM is a one-step adaptation, not exact published reproduction. No detector winner or retuning is authorised by these results. Undefined metrics are JSON null / blank CSV with denominator counts. Bootstrap pairs channels and groups original seeds; persistence remains one stream. Shared contexts limit independence.\n')
  # The output lock is the one-time completion boundary. Retry only incomplete, same-identity work.
  files={f.relative_to(out).as_posix():sha(f) for f in out.rglob('*') if f.is_file() and f.name not in ['ACTIVE_RUN.json','LABEL_EVALUATION_COMPLETE.json'] and '.tmp-' not in f.name}
  freeze(out/'LABEL_EVALUATION_COMPLETE.json',{'status':'COMPLETE_LOCKED_RETROSPECTIVE_EVALUATION','files':files,'score_export_freeze_sha256':identity['score_export_freeze_sha256'],'settings_changed':False,'scientific_fits':0,'channel_detector_rows':2106,'detector_selection_performed':False})
 print('LABELLED EVALUATION COMPLETE. Do not tune or replace frozen arms in response.')
if __name__=='__main__':main()
