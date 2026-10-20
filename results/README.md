# Results index

Original saved tables; no experiments were rerun.

| Comparison | Files |
|---|---|
| Forecasting model selection | [`EXP006`](../experiments/EXP006/outputs/tables/canonical_summary.csv), [`EXP007`](../experiments/EXP007/outputs/tables/promotion_checks.csv) |
| GCN vs project-trained LSTM, common detector | [`EXP012 mission metrics`](../experiments/EXP012/evaluation/reports/mission_metrics.csv) |
| GCN vs author-released LSTM | [`released-LSTM metrics`](../followups/released_lstm/outputs/mission_metrics.csv) |
| Original vs substitute detectors | [`native-detector metrics`](../followups/native_detector/outputs/mission_metrics.csv) |
| Command-reference models | [`EXP013 metrics`](../experiments/EXP013/outputs/reports/mission_metrics.csv) |
| GDN detector substitution | [`EXP014 reports`](../experiments/EXP014/outputs/reports/) |
| Command-reference detector substitution | [`EXP015 reports`](../experiments/EXP015/outputs/reports/) |

**Do not merge the two LSTM comparisons:** their evaluated samples differ. Some saved sensitivity records were superseded by corrected records; use each run's audit/freeze for interpretation. Negative or inconclusive experiments remain recorded as such.
