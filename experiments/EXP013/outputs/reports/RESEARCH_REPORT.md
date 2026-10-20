# EXP013 completed research report

NASA TEST and final TRAIN were previously exposed. Every NASA comparison below is exploratory retrospective evidence. Synthetic confirmation does not establish real spacecraft generalization. EXP012 is unchanged.

|Mission|Model|Corrected event F0.5|Point F1|Point precision|Point recall|Nominal alarm occupancy|Coverage|
|---|---|---:|---:|---:|---:|---:|---:|
|MSL|graph|0.0265|0.2065|0.1517|0.3234|0.2368|27/27|
|MSL|removed|0.0333|0.2233|0.1669|0.3393|0.2241|27/27|
|MSL|fusion|0.0322|0.2120|0.1542|0.3397|0.2443|27/27|
|MSL|gdn|0.0674|0.1446|0.1370|0.1532|0.1266|27/27|
|MSL|catch|0.0567|0.2407|0.1751|0.3849|0.2379|27/27|
|SMAP|graph|0.0277|0.4026|0.3404|0.4932|0.1460|54/54|
|SMAP|removed|0.0369|0.3690|0.3326|0.4143|0.1269|54/54|
|SMAP|fusion|0.0248|0.3693|0.3320|0.4239|0.1319|54/54|
|SMAP|gdn|0.0599|0.2863|0.2977|0.2758|0.0993|54/54|
|SMAP|catch|0.0512|0.3349|0.3283|0.3417|0.1067|54/54|

GDN/CATCH use pinned official model code and losses under this controlled NASA protocol; the common splits, pooled entity-window training, causal scoring and TRAIN-only thresholds differ from published leaderboard protocols. These are not reproductions of their reported SOTA scores. Published baselines have one initialization; new models have three. Missing/failed coverage is explicit. STGLR and OmniAnomaly are excluded for the documented fidelity/environment gaps.

The primary architecture contrasts are graph minus message removal and direct fusion on both missions. Their bootstrap intervals use98.75% coverage to describe four prespecified contrasts. Shared model fitting, shared commands and earlier TEST exposure prevent an independent superiority claim. Historical LSTM complete-pipeline comparisons retain its original adaptive detector and timing; they cannot isolate architecture alone.

See event notification timing, strict point metrics, alarm burden, calibration/select statuses and separate synthetic confirmation, including background-overlap and invalid-injection counts. A NO_FEASIBLE_CANDIDATE selection remains a negative result regardless of NASA scores.

The official NASA arrays were scaled using TEST extrema before release. Our additional transformations use FIT only, but inherited preprocessing prevents a raw-data leakage-free claim.

No new TEST-dependent threshold, seed, smoothing, postprocessing or model selection occurred. No point adjustment is reported. Independent real-anomaly confirmation is still required.

Synthetic decision: NOT_ESTABLISHED_ON_BOTH_MISSIONS. See DECISION.json for all prespecified gates and retrospective joint outcomes.