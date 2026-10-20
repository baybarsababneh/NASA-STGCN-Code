# Addendum: EXP010 evidence and proposed EXP011

The existing reviewer package and all historical results remain authoritative. Append this folder; do not replace the report's historical methods or tables.

## Newly checked evidence

The full EXP010 export passed its original read-only verifier: 114 nominal replay units plus 2,964 perturbation units, all 3,078 commit hashes and output-manifest entries verified. This checks saved evidence, not a fresh GPU numerical replay. Parent nominal records assert exact historical MSE and exact repeated predictions. No models were trained or NASA data evaluated during this audit.

Recomputed from the channel response table, averaging seeds/positions, then amplitudes/signs within channel, then channels:

| Mechanical cohort, six channels/mission | MSL mean / median | SMAP mean / median |
|---|---:|---:|
| Spike response ratio | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| Offset32 response ratio | 0.3202 / 0.0789 | 0.1722 / 0.2217 |
| Ramp32 response ratio | 0.3464 / 0.0891 | 0.1591 / 0.1836 |

Response ratio is sum of absolute *change in residual* divided by sum of absolute injected telemetry change during the intervention. It is neither detection probability nor a signal-to-noise measure. Spike ratio 1 is expected under causal forecasting: the forecast at the injected sample cannot depend on that sample. Post-event effects are a separate quantity. Means and medians show substantial channel heterogeneity. These are selected-channel diagnostics, not all-mission estimates.

There are 33 null intervention records per learned-model role and 11 for unique persistence probes; these are excluded from response/detection interpretations, not counted as misses. Stuck results depend on whether the original telemetry would have moved at all.

At command-context lag 8, only three of six MSL mechanical channels and two of six SMAP channels have >=20 samples in both active/inactive groups in both reference and assessment. This is a coverage diagnostic, not a sufficient statistical sample-size standard. Effects vary in sign and strength. The data do not currently justify suppressing residuals during command activity or using a universal context-conditioned threshold.

## Interpretation change

The forecasting architecture remains frozen and valid. Causal adaptation to corrupted observations is a detection limitation, not automatically a model defect. EXP008/009 failed calibration/detection findings remain unchanged. EXP010 did not rescue them. Persistent residual accumulation alone cannot guarantee evidence throughout a fault once the forecaster has followed it.

The proposed bounded next question is whether a low-capacity detector can combine observations, telemetry changes and frozen forecast response to distinguish the prescribed end-to-end faults from nominal transitions. It is synthetically supervised, not trained with NASA anomaly labels. Nested channel exclusion and residual-only/message-removed/persistence controls separate information gain from forecaster-specific gain. No actual EXP011 scientific results exist yet.

## Review requests

Please challenge: synthetic fault relevance; nominal-label assumptions; the meaning of per-1,000-sample alarm budgets; the event-level incremental gate; channel transfer with only six mechanical channels per mission; and whether any claimed benefit truly uses the GCN. The new protocol has explicit failure/abstention rules and does not permit a least-bad candidate to advance.

The historical AP tie-handling bug remains documented. Historical outputs are unchanged. EXP011 uses no NASA AP metric; the corrected metric implementation remains required for eventual separately frozen retrospective evaluation.

## Completion limits

The full 440-fit detector pilot was not run during package preparation. Local tests cover causal features, zero-scale safety, deterministic convex optimisation on a synthetic engineering fixture, gate logic, abstention, input hashes and complete feature loading. Colab itself has not been exercised in this environment. Input schema or numerical errors require an engineering amendment, never silent scientific changes.

Relevant methodological background: Goswami et al., Unsupervised Model Selection for Time-series Anomaly Detection, https://arxiv.org/abs/2210.01078 (synthetic selection is a surrogate); Cobb et al., Sequential Multivariate Change Detection with Calibrated and Memoryless False Detection Rates, https://proceedings.mlr.press/v151/cobb22a.html (false-alarm control is a distinct calibration problem, whose guarantees cannot simply be imported into these heterogeneous short streams).
