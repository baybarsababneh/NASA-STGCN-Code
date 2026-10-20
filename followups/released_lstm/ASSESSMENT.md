# Independent LSTM benchmarking reassessment

Analysis: **LSTM_RELEASE_BENCHMARK_AUDIT_V1**, completed 16 October 2026.

## Executive assessment

The concern is justified. The controlled LSTM remains a valid matched-input project baseline, but it is not a faithful reproduction or a reliable proxy for the effectiveness of the author-released LSTM. A new comparison of the actual released forecasts resolves much of this uncertainty without retraining. On common time support under the identical frozen detector, the released LSTM has higher corrected event scores and strict point F1 on both missions. The controlled LSTM substantially understates the released-reference performance on MSL. The GCN's MSL advantage over the controlled baseline must not be generalised to Hundman's released approach.

The GCN's lower nominal alarm burden is a real trade-off. It does not establish equivalent detection performance. The MSL event-score difference is unresolved by conditional paired-channel intervals; that is not proof of equivalence or noninferiority. The SMAP event-score and strict point coverage deficits favour the released reference in this analysis.

No models were fitted, no historical outputs were changed, and no thresholds, offsets or detector parameters were chosen using new NASA results. The protocol was sealed before new metric calculation. This is an additional retrospective analysis, separate from EXP001--EXP015. Historical NASA TEST exposure, published-method development on this benchmark, and inherited TEST-extrema scaling remain limitations.

## 1. Evidence inspected and verified

- The supplied current PDF report, including its updated related work, EXP012/EXP014 comparisons, EXP015 supplement, scoring definition, uncertainty, and reproducibility boundary.
- The supplied Hundman arXiv v3 paper and NASA archive, labels, and source notebooks.
- EXP012 input bundle and full label-blind score export recovered from existing files. These were sufficient for the relevant project predictions; new training was unnecessary.
- The EXP015 input bundle's frozen copy of the EXP012 evaluator and reports. Its detector bytes match the original EXP012 detector.
- Official telemanom source at current commit `2e6c5b6c3558e7835601519b7bdef37c649bdbdc`, first commit `b0f321694718` (full identity in the GitHub history snapshot), and the August 2018 fixed snapshot. The actual immutable commits, source snapshots and hashes are retained.
- All 82 released HDF5 models, all 82 cached scalar prediction and smoothed-error streams, the release log, and initial/current official result CSVs.

Integrity results:

- All **1,053** EXP012 stream commits and their **3,159** committed payload hashes verified.
- All **162** TRAIN/TEST byte bindings for the existing 81-channel roster agree with the supplied archive and freezes.
- Every released model has two 80-unit LSTMs, `tanh` state activation, `hard_sigmoid` recurrent gates, dropout 0.3 after each recurrent layer, and a 10-unit linear output head. Metadata records Keras 2.1.5 and TensorFlow.
- Label-independent first/middle/last prediction checks against every released model: **246 sampled forecasts**, maximum absolute discrepancy **8.35e-7** under the fixed `atol=1e-5, rtol=1e-4` replay tolerance. This is a NumPy forward verification, not retraining or a full GPU replay.
- Reconstructed cached EWMA errors match **exactly for all 81 evaluated channels**. Unlabelled T-10 has a cached-error mismatch of 0.0259943 and remains excluded under the pre-existing roster policy; its model sample checks pass. Its mismatch is not concealed or used to change the roster.
- All **486** relevant saved project GCN/LSTM stream count and notification summaries reproduce on their original full support.
- All **648** new common/native decision streams pass an independent arithmetic check from saved masks.
- All **81** recomputed current official native alarm-interval exports match the pinned official result CSV exactly. This establishes native release replay, not exact reproduction of the printed paper table.

## 2. Precisely what differs

| Component | Paper / author release | EXP012 controlled LSTM | Consequence |
|---|---|---|---|
| Temporal architecture | Two 80-unit Keras LSTMs with dropout 0.3 | Two 80-unit PyTorch LSTMs with dropout 0.3 | Broad capacity is matched, implementation is not identical. |
| Recurrent gates | Saved models use `hard_sigmoid` | PyTorch uses logistic sigmoid | Different state dynamics, saturation and optimisation. |
| Initialisation | Saved Keras configuration specifies Glorot kernels, orthogonal recurrent matrices, unit forget bias | PyTorch default recurrent initialisation | Not a weight-equivalent port. |
| Forecast objective | Paper text says one step; release log/code and all saved heads use ten future outputs | One-step scalar MSE | Published description and actual released training objective differ. |
| Saved scalar forecast | First output from each current window; not an average of ten horizon forecasts | Direct one-step output | Both saved scalar streams forecast the next observation, although their training objectives differ. |
| Lookback | 250 observations | 64 observations | Different historical information. D-12 has 312 official TRAIN rows but only 187 in the project's 60% FIT prefix, so a uniform lookback-250 reproduction cannot fit under that project split. |
| Input features | All 25 SMAP or 55 MSL dimensions: telemetry plus commands | Telemetry plus frozen selected commands: five total inputs for every MSL file and varying eligible SMAP support | The controlled comparison deliberately matches the GCN's feature contract, but may weaken the LSTM relative to the release. |
| Additional scaling | Direct released units after upstream NASA normalisation | Target column additionally FIT-standardised, then inverted for common detection | Changes fitting dynamics. Commands remain binary. A-1 and R-1 have near-constant FIT targets and scales below 2.4e-7. These are potential confounders, not demonstrated causes of the observed gap. |
| Training exposure | Release code shapes windows from the official TRAIN record, shuffles windows before the 20% Keras validation split | Chronological 60% FIT / 20% STOP / 20% CAL; only FIT gradients | More raw training exposure in the release and overlapping shuffled validation windows. No newly independent temporal validation can be inferred from the released training procedure. |
| Optimisation | Adam, 35-epoch ceiling, batch 64, patience 10, minimum improvement 0.0003; saved Adam epsilon 1e-7 | Same broad settings; Adam learning rate 0.001 and default epsilon 1e-8 | Similar settings do not imply equivalent optimisation. |
| Selected weights | Released source saves weights at early termination; no `restore_best_weights` request | Explicitly restores the best eligible chronological STOP checkpoint | Different checkpoint selection. Individual original training histories and fit seeds cannot be recovered from code alone. |
| Repeated fits | One released fit per channel | Three frozen seeds, 17/29/43 | Release fit-to-fit variability is unknown. Its one stream is never treated as three independent runs. |
| Detection | Author pipeline has adaptive residual thresholds, smoothing, inverse branch, buffers and pruning, including a C-2 startup exception | Frozen HUNDMAN_ADAPT_V1 removes that channel exception and handles degenerate/tail cases explicitly | A shared detector is suitable for forecast-evidence comparison, but is not the exact original end-to-end method. |
| Threshold calibration | Adaptive thresholds use unlabelled TEST residual windows; published pruning parameters were investigated on the historical benchmark | Same fixed parameters for the common-detector comparison; raw-q99 is a separate TRAIN-CAL reference | Neither adaptive detector is exclusively TRAIN-calibrated. No NASA label-based new threshold tuning occurred. |
| Primary scoring | Published event overlap, with a version-specific release evaluator and inconsistent printed F0.5 arithmetic | Study-defined corrected event F0.5 penalises nominal occupancy; strict point metrics and issue time are separate | Published numbers must not be pasted into the project table as though they use the same endpoint. |

## 3. Benchmarking strategy and why it is preferable

Retain two distinct comparators. The controlled LSTM estimates performance under the project's narrower data and training contract. The actual released LSTM estimates competitiveness against a concrete author-provided reference. Neither alone isolates architecture, because the two references have different information and training exposure.

The primary supplement applies the **exact existing HUNDMAN_ADAPT_V1 detector and frozen evaluator** to all three forecasting roles. Every stream uses raw target indices `250 <= t < N-10`. The release's array-shaping loop proves this alignment; it was not fitted using labels, correlations, or best anomaly scores. The GCN and controlled LSTM predictions are cropped to that same interval and the detector is rerun on the cropped streams. Cropping previously computed flags alone would not control the final adaptive-batch boundary.

A common detector is appropriate here because all three streams are causal forecasts of the same scalar telemetry in the same released units. It controls smoothing, buffering, pruning, nominal guards, event support and scoring. It does not guarantee detector neutrality: it is a Hundman-derived detector, and the trained models differ. For that reason the analysis also reports strict point metrics, within-channel raw/EWMA average precision, alarm burden, and issued-flag timing rather than relying on a single event score.

The complementary native replay executes the pinned official v2 residual detector on the released LSTM. Its only execution-compatibility patch casts the upstream `np.delete` row indices to integers. It retains the C-2 exception and the original ten-target telemetry-scale guards. New issue-time instrumentation charges the nine additional observed targets in those guards; these issue times are audit reconstructions, not archived operator notifications. Native intervals match all 81 official CSV records exactly. Applying a different native detector to the GCN was not necessary to answer whether the recreated baseline understates the released reference; it would create another GCN hybrid rather than improve the existing primary control.

TRAIN-only raw-q99 thresholds were not invented for the released reference: the project's CAL block may already have been used in original release fitting, and no comparable out-of-fit calibration forecasts were supplied. A new faithful retraining programme could isolate architectural, input or split effects, but is not needed to make the direct frozen-reference comparison now available.

## 4. New verified results

All values below are newly computed on the common `250:N-10` support. Project mission metrics pool channel counts within each seed and then average the three seed metrics. The author release has a single saved fit.

| Common-detector forecaster | MSL corrected event F0.5 | MSL strict point F1 | SMAP corrected event F0.5 | SMAP strict point F1 |
|---|---:|---:|---:|---:|
| Author-released LSTM | 0.843424 | 0.431214 | 0.833718 | 0.334602 |
| Project GCN | 0.769346 | 0.342594 | 0.718832 | 0.180842 |
| Project controlled LSTM | 0.707107 | 0.360485 | 0.814321 | 0.327769 |

These GCN/control values differ slightly from the original report because the last ten targets are omitted for every role and adaptive final batches are reprocessed. The historical full-support scores remain unchanged.

| Common-detector endpoint | MSL release / GCN / controlled | SMAP release / GCN / controlled |
|---|---|---|
| Unpenalised event-overlap F0.5 | 0.8824 / 0.7912 / 0.7371 | 0.8472 / 0.7265 / 0.8261 |
| Event recall | 66.67% / 46.30% / 49.07% | 89.71% / 59.80% / 71.57% |
| Nominal occupancy | 5.91% / 4.14% / 5.74% | 1.96% / 1.40% / 1.84% |
| Nominal alarm starts per 1,000 nominal samples | 0.2375 / 0.1979 / 0.2601 | 0.1038 / 0.0729 / 0.0792 |
| Within-event issued-flag rate | 16.67% / 13.89% / 13.89% | 72.06% / 41.18% / 59.31% |
| Raw residual AP, macro over channels | 0.4019 / 0.2907 / 0.3446 | 0.4080 / 0.2992 / 0.3447 |
| EWMA residual AP, macro over channels | 0.4682 / 0.4473 / 0.4314 | 0.5761 / 0.5420 / 0.4976 |

Nominal means outside supplied labels, not independently verified healthy spacecraft operation. Within-event issued flags are not necessarily new alarm starts. The raw AP uses within-channel rankings, avoiding meaningless pooling across incomparable telemetry scales; it is a descriptive threshold-free diagnostic, not a new calibrated alarm policy.

Native released LSTM scores are **0.851975 MSL / 0.833845 SMAP** corrected event F0.5, with strict point F1 **0.439303 / 0.330725**. Its within-event issued-flag rates are **13.89% / 66.18%**, subject to the reconstruction caveat above. Thus the released-reference result does not depend on using only the project's common detector. The MSL common/native difference disappears when C-2 is excluded, identifying the impact of the label-informed historical exception without selecting it as a preferred benchmark.

## 5. Uncertainty and sensitivity

Paired-channel bootstrap, 10,000 resamples, RNG seed 20261016. All project seeds stay grouped within each sampled channel; the released fit remains one stream.

| Corrected event contrast | Difference | Descriptive 95% interval | Descriptive 98.75% interval |
|---|---:|---|---|
| MSL GCN minus release | -0.07408 | [-0.20721, +0.02283] | [-0.25213, +0.05153] |
| SMAP GCN minus release | -0.11489 | [-0.20466, -0.02761] | [-0.22927, -0.00133] |
| MSL controlled LSTM minus release | -0.13632 | [-0.25452, -0.05276] | [-0.30075, -0.03749] |
| SMAP controlled LSTM minus release | -0.01940 | [-0.10688, +0.06795] | [-0.12832, +0.09067] |

For GCN minus release strict point F1, the 95% intervals are **[-0.18758, -0.00471] MSL** and **[-0.20178, -0.10530] SMAP**. The corresponding 98.75% MSL interval includes zero. These are conditional channel resamples, not evidence from independent model refits or independent missions. Shared incidents, duplicated/context-related records, small MSL channel count and only one released fit limit population-level interpretation. The 98.75% intervals provide a conservative additional view for four prespecified primary event contrasts, not a blanket multiplicity correction for every endpoint or dependence concern. No noninferiority margin was specified; interval overlap with zero cannot establish competitiveness as equivalence.

The prespecified C-2 exclusion leaves MSL corrected scores release **0.854274**, GCN **0.783159**, controlled **0.717712**; the released-reference lead remains. Using the second conflicting P-2 label record changes SMAP scores to release **0.833817**, GCN **0.718909**, controlled **0.814412**. No role or parameter was selected from these sensitivities. Use `sensitivity_verified_seed_metrics.csv`; an initial no-op SMAP C-2 exclusion is preserved but superseded by the independently corrected mission-scoped sensitivity.

## 6. Publication and release provenance are not interchangeable

The paper describes one-step training, whereas the released models demonstrably have ten-output heads. Its printed F0.5 values are arithmetically inconsistent with its precision/recall pairs. The initial 2018 result CSV records MSL event counts H=25, U=1, M=11, giving precision 96.15%, whereas the paper prints 92.6%. Its SMAP CSV retains duplicate P-2 rows. The initial/current source snapshots also differ in some detector outputs, annotations and coordinate conventions.

The initial 2018 labels and alarm intervals use target-relative coordinates; conversion to raw data requires +250. Eleven historical label endpoints then exceed the actual supplied array length; clipping to the real last row exactly recovers the corresponding current endpoints. This is a finite-support metadata correction, not offset tuning. All such checks are exported. Literal old counts, corrected-roster rescoring and new native replay are kept separate.

Rescoring the initial saved intervals on the current 81-channel roster gives corrected event F0.5 **0.854307 MSL / 0.845369 SMAP**, and strict point F1 **0.442314 / 0.273056**. No historical notification times are available for those saved intervals. They are a supplementary receipt check; their differences do not justify replacing the primary common-detector results or claiming exact numerical reproduction of the printed publication.

## 7. Scientific verdict and necessary report changes

**Existing strengths:** the controlled model was explicitly labelled an adaptation, uses matched selected inputs and chronological development support, has three seeds and verifiable outputs, and shares the downstream detector/evaluator. Existing GCN-minus-controlled conclusions remain true within that contrast. The report already avoids a general superiority claim and distinguishes retrospective event overlap from operational notification.

**Existing weakness:** performance against the controlled baseline does not answer competitiveness against the released Hundman reference. New evidence makes this substantive on MSL, where the controlled LSTM score is much lower than the released model's. A broad statement that the GCN is competitive with the original method would now be insufficiently supported.

**Determination:** the GCN achieves useful retrospective detection, with lower nominal burden, but it does not match the author-released reference's point estimates on either mission. SMAP shows a substantial supported deficit under the conditional analysis. MSL corrected event performance remains uncertain relative to the release, and strict point coverage is lower; neither superiority nor equivalence is established. Unequal training/input contracts prevent attributing this solely to graph versus recurrent architecture.

**What changes:** add the direct released-reference benchmark; explain why the controlled and released LSTMs answer different questions; limit the conclusion to the demonstrated trade-offs; explicitly separate the supplement from EXP001--EXP015. Preserve historical scores and figures. `Report_Updates.tex` supplies exact additions and a replacement for the first conclusion paragraph. The rest of the report is unchanged.

**What remains unresolved:** exact reconstruction of the paper's printed results, released-fit training histories/seeds, causal sources of the controlled-LSTM gap, generalisation under independent TRAIN-only preprocessing, incident-level dependence, and practical noninferiority under an explicit alarm budget. These require information or genuinely new data/designs; additional tuning on the already examined NASA TEST set would not resolve them.

## Reproduction

The package includes analysis scripts, frozen detector/evaluator code, relevant release source snapshots, all new decision masks/issue times/window thresholds, per-channel and mission tables, bootstrap intervals, source verification and receipts. It excludes duplicate copies of the large existing NASA and EXP012 archives.

Required existing files, with exact SHA-256 in `outputs/ANALYSIS_FREEZE.json`:

1. Supplied NASA `archive.zip`, SHA-256 `6084d3ee3906381f2c98aa3773b6b2d77c82413503faa78f962196582e873733`.
2. `EXP012_LABEL_BLIND_SCORE_EXPORT.zip`, SHA-256 `e224c1286955d8af9c5bc314d5c24be04ab25b1a3fcba722a4bc217565b4d541`.
3. `EXP012_ANOMALY_BENCHMARK_PREPARATION_V1_input_bundle.zip` and the supplied labels/report. All are existing artifacts, not requests for new training.

Preserve the folder layout documented in `README_REPRODUCTION.md`. Execute `verify_sources.py`, `analyse.py`, `audit_historical_receipts.py`, then `verify_analysis.py`. The exact environment is in `outputs/ENVIRONMENT.json`. Source scripts are hashed by the package manifest. `analyse.py` refuses changed primary inputs/source against its freeze. All scoring is deterministic; sampled weight replay uses a predeclared numerical tolerance. No experiment checkpoint is written or altered.

Primary references: Hundman et al. (2018), DOI 10.1145/3219819.3219845, supplied arXiv v3; official author repository https://github.com/khundman/telemanom, immutable current source commit identified above. The paper, release source and direct file checks support distinct claims; none is a substitute for the others.
