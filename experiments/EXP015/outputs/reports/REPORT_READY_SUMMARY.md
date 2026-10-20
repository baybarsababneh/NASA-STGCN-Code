# EXP015 frozen EXP013 residual/Hundman supplement

All neural weights and previous results remain frozen. This analysis changes the anomaly evidence to the command-reference residual and applies the exact EXP012 detector. It is retrospective and exploratory.

| Mission | Method | Corrected event F0.5 | Event P | Event R | Point P | Point R | Point F1 | Nominal occupancy | Starts/1000 nominal | Seeds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|MSL|EXP013 GCN + residual/Hundman|0.758011|0.891775|0.537037|0.437120|0.311271|0.363561|0.052443|0.236435|3|
|MSL|EXP013 removal + residual/Hundman|0.792642|0.950000|0.527778|0.472974|0.314362|0.377691|0.045941|0.202658|3|
|MSL|EXP013 fusion + residual/Hundman|0.705287|0.820539|0.527778|0.398322|0.299511|0.341439|0.059953|0.270211|3|
|MSL|EXP012 GCN + Hundman|0.769420|0.961874|0.462963|0.463082|0.271440|0.342249|0.041275|0.197029|3|
|MSL|EXP012 controlled LSTM + Hundman|0.707182|0.843290|0.490741|0.419893|0.315220|0.360076|0.057234|0.258952|3|
|MSL|EXP014 GDN + Hundman|0.747087|0.900000|0.500000|0.431063|0.305563|0.357622|0.052894|0.253323|1|
|MSL|EXP013 GCN + original pipeline|0.026475|0.027936|0.935185|0.151701|0.323418|0.206500|0.236817|20.620472|3|
|MSL|EXP013 removal + original pipeline|0.033270|0.034256|0.944444|0.166916|0.339257|0.223342|0.224117|17.957768|3|
|MSL|EXP013 fusion + original pipeline|0.032216|0.034465|0.898148|0.154186|0.339729|0.212035|0.244344|16.156362|3|
|SMAP|EXP013 GCN + residual/Hundman|0.733362|0.791156|0.607843|0.646118|0.202782|0.308107|0.017247|0.087335|3|
|SMAP|EXP013 removal + residual/Hundman|0.783799|0.874376|0.573529|0.696976|0.192226|0.301319|0.012776|0.065501|3|
|SMAP|EXP013 fusion + residual/Hundman|0.755967|0.825917|0.593137|0.669938|0.191046|0.296820|0.014680|0.080057|3|
|SMAP|EXP012 GCN + Hundman|0.715577|0.764007|0.598039|0.542005|0.108377|0.180478|0.014064|0.073689|3|
|SMAP|EXP012 controlled LSTM + Hundman|0.810694|0.854441|0.715686|0.644126|0.219675|0.327593|0.018497|0.080057|3|
|SMAP|EXP014 GDN + Hundman|0.793519|0.860000|0.632353|0.678303|0.200476|0.309482|0.014511|0.068231|1|
|SMAP|EXP013 GCN + original pipeline|0.027665|0.026023|0.887255|0.340414|0.493193|0.402623|0.146026|6.919502|3|
|SMAP|EXP013 removal + original pipeline|0.036937|0.034159|0.848039|0.332614|0.414327|0.368987|0.126879|4.611485|3|
|SMAP|EXP013 fusion + original pipeline|0.024811|0.022799|0.848039|0.332044|0.423942|0.369267|0.131889|7.533579|3|

## What the graph comparison shows

MSL: graph-minus-removal corrected event F0.5 = -0.034631; graph-minus-fusion = +0.052724.
The graph-removal control matches or exceeds the graph on corrected event F0.5; this outcome does not establish graph-specific benefit.
The removal control's change relative to its own original pipeline is at least as favourable as the graph's. Any observed pipeline improvement therefore cannot be attributed to graph messages alone.
Changes versus each frozen original pipeline: graph +0.731536, removed +0.759372, fusion +0.673071.
SMAP: graph-minus-removal corrected event F0.5 = -0.050438; graph-minus-fusion = -0.022606.
The graph-removal control matches or exceeds the graph on corrected event F0.5; this outcome does not establish graph-specific benefit.
The removal control's change relative to its own original pipeline is at least as favourable as the graph's. Any observed pipeline improvement therefore cannot be attributed to graph messages alone.
Changes versus each frozen original pipeline: graph +0.705697, removed +0.746862, fusion +0.731156.

Positive deltas establish sensitivity to this joint score/detector substitution within the frozen retrospective study. Poor or unchanged performance means this specific substitution did not rescue the model; it does not identify a failure mechanism. Substantial graph value requires examining both controls, point discrimination, alarm burden and the descriptive paired intervals.

See `paired_contrasts.csv` and `paired_channel_bootstrap.json` for 95% descriptive intervals, including all seeds grouped and undefined resamples. See `mission_seed_metrics.csv`, `timing_summary.csv`, `event_notifications.csv` and `missed_events.csv` for actual issuance, delays and misses. A label-overlap hit may be issued after event end.

The GCN reference uses past commands only, while GDN and EXP012 forecast from telemetry history. These are complete pipeline comparisons, not a perfectly controlled architecture ablation. Original-unit means released NASA telemetry units, whose TEST-extrema preprocessing is inherited. No physical leakage isolation, independent generalisation, equivalence, SOTA or operational performance claim follows.

The discriminator hybrid was excluded before scoring: signed logits lack the telemetry scale and observed-target variation required by the exact Hundman guards. No score transformation or TEST-selected calibration was added.
