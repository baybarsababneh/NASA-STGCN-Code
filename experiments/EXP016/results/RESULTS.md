# EXP016 completed evaluation

Retrospective comparison with one author-released LSTM fit per channel. No new LSTM was trained.
Primary: uniform author-derived detector; native_C2 retains the historical label-informed exception.
Event F0.5 uses the standard formula and inclusive any-overlap events. No point adjustment.
Test-extrema preprocessing, historical TEST exposure, different validation split and unknown original fit seeds prevent a pure architecture attribution.

| Mission | Detector | Role | Event F0.5 | Strict point F1 | Nominal occupancy |
|---|---|---|---:|---:|---:|
| MSL | uniform | stgcn | 0.637445 | 0.296654 | 0.048508 |
| MSL | uniform | released_lstm | 0.882353 | 0.431214 | 0.059064 |
| MSL | native_C2 | stgcn | 0.658226 | 0.306926 | 0.050899 |
| MSL | native_C2 | released_lstm | 0.892857 | 0.439303 | 0.060675 |
| SMAP | uniform | stgcn | 0.718982 | 0.180374 | 0.015167 |
| SMAP | uniform | released_lstm | 0.847222 | 0.330725 | 0.019395 |
| SMAP | native_C2 | stgcn | 0.718982 | 0.180374 | 0.015167 |
| SMAP | native_C2 | released_lstm | 0.847222 | 0.330725 | 0.019395 |

Read mission_seed_metrics.csv for every seed and P-2/C-2 sensitivities; event_notifications.csv records actual availability, including detector lookahead.
Bootstrap intervals condition on available fits and channel resampling; shared-incident dependence and LSTM fit variance are unresolved.
DETECTOR_FAILURES.json must be empty for a complete native comparison. Inspect DEGENERATE_BRANCHES.json for primary branch abstentions. Runtime and validation gates do not establish scientific superiority.