# EXP014 final bounded literature comparison

No new training, model selection, detector tuning or replacement of EXP012/EXP013 results occurred.

| Panel | Mission | Model | Fc0.5 | Strict point F1 | Nominal occupancy | Nominal starts/1000 | Seeds |
|---|---|---|---:|---:|---:|---:|---:|
|common_detector|MSL|gcn|0.769420|0.342249|0.041275|0.197029|3|
|common_detector|MSL|gdn|0.747087|0.357622|0.052894|0.253323|1|
|common_detector|MSL|lstm|0.707182|0.360076|0.057234|0.258952|3|
|common_detector|SMAP|gcn|0.715577|0.180478|0.014064|0.073689|3|
|common_detector|SMAP|gdn|0.793519|0.309482|0.014511|0.068231|1|
|common_detector|SMAP|lstm|0.810694|0.327593|0.018497|0.080057|3|
|complete_pipeline|MSL|catch|0.056722|0.240660|0.237870|8.089440|1|
|complete_pipeline|MSL|gcn|0.769420|0.342249|0.041275|0.197029|3|
|complete_pipeline|MSL|gdn|0.067379|0.144646|0.126628|6.130411|1|
|complete_pipeline|MSL|lstm|0.707182|0.360076|0.057234|0.258952|3|
|complete_pipeline|SMAP|catch|0.051161|0.334864|0.106696|3.496141|1|
|complete_pipeline|SMAP|gcn|0.715577|0.180478|0.014064|0.073689|3|
|complete_pipeline|SMAP|gdn|0.059906|0.286319|0.099311|2.259801|1|
|complete_pipeline|SMAP|lstm|0.810694|0.327593|0.018497|0.080057|3|

Common detector uses the exact frozen EXP012 HUNDMAN_ADAPT_V1 on original-unit telemetry forecast residuals. GDN is an adapted forecaster, not a perfectly controlled architecture ablation. Complete pipelines retain their frozen native-score detector adaptations. Paper-specific numbers remain in the original report and literature decision document; they are not ranked in this table.

All intervals are descriptive 95% paired-channel intervals conditional on the saved fits. No independent generalisation, equivalence or deployment claim follows. Notification timing credits the earliest actual issued event-overlapping flag, and may be after the event. Misses and late notifications are retained.

See mission_seed_metrics.csv for event/point precision and recall, exact count denominators and per-seed notification medians. See paired_channel_bootstrap.json for event, point and alarm contrasts. Historical values and frozen native results are reused unchanged.