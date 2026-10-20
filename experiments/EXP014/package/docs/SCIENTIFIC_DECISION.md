# Prespecified final comparison decision

## Scope

The principal question concerns **immutable EXP012**, whose historical corrected event F0.5 values are GCN 0.769420/0.715577 and controlled LSTM 0.707182/0.810694 on MSL/SMAP. These values are reused, never targets for selection. EXP013 is a different completed experiment, including all 22 fits, its negative synthetic decision and its disclosed auxiliary seed-export defect. This stage changes neither study.

| Method/evidence | Decision | New computation |
|---|---|---|
| EXP012 GCN and controlled LSTM | Reuse their exact HUNDMAN_ADAPT_V1 exports and reported results | Verify counts/timing from saved flags; no model inference |
| EXP013 trained GDN | Add the exact EXP012 detector on the forecast of telemetry column 0, inverted to released telemetry units | Frozen inference only: two mission checkpoints, seed 17, 81 channels |
| Native-score GDN and CATCH NASA adaptations | Reuse the immutable EXP013 results under their existing TRAIN-selected detector | No rescoring, recalibration or retraining |
| CATCH–Hundman reconstruction-score hybrid | Omit: unnecessary for the minimum question, and not equivalent to forecast residual detection | None |
| STGLR | Record a fidelity exclusion, with the paper verified and code/checkpoint evidence unresolved | None |
| OmniAnomaly | Record a practical official-environment exclusion | None |
| Published paper numbers | Context only, clearly separate from the two NASA panels | No leaderboard ranking |

## Common detector

GDN predicts all 55 MSL/25 SMAP within-file dimensions from the preceding five samples. Its column-0 output is a legitimate next-sample telemetry forecast. Different telemetry files are not established to be synchronised sensors and are never joined. GDN's learned sparse variable graph and EXP012's command-to-target message graph have different semantics.

EXP013 first standardises each target using its per-entity FIT mean/standard deviation with a 0.001 floor, then applies a pooled mission FIT scaler to all dimensions. Preserve the intermediate FP32 cast. For native target prediction `p`, the correct inverse is

`prediction = (float64(p) * pooled_scale[0] + pooled_mean[0]) * entity_scale + entity_mean`.

Use the exact original TEST target value and its original index. The detector receives `abs(target − prediction)` in the released telemetry units, not a max-dimensional native anomaly score, a CAL-normalised deviation or a residual on the GDN scaler. Absolute guards in the detector make scale materially important.

The supplied EXP013 scoring implementation saved absolute FP32 multivariate deviations/native scores, not signed target predictions. Signs and exact subtraction-rounding information cannot be recovered from that export. Re-inference is therefore necessary for an exact target-prediction construction. A claimed original-unit residual reconstructed from a max native score would be invalid.

Inference retains native batch 32 boundaries beginning at t=5 and the original full native score/smoothing history before cropping at t=250. The new Hundman detector is a separate stream beginning at t=250, just like EXP012. Its EWMA, absolute/relative guards, batch 70/window 30, threshold grid, error buffer 100, pruning p=0.13, short-stream rules, last-batch remainder, startup averaging and issue indices are **byte-preserved**. The implementation SHA-256 is `5b238ae7226a60e092b4d0a0954ca276733f3f9a27c0766d98282512dc692714`.

This is common-detector evidence, **not a perfectly controlled architecture ablation**. EXP012 uses frozen per-channel GCN/LSTM models, selected FIT context features, lookback 64 and its inherited FIT scaling. GDN uses all within-file dimensions, lookback 5, shared mission parameters, an all-dimension forecasting loss, the EXP013 scaler/floor and one initialisation. The controlled LSTM is a two-layer 80-unit, dropout 0.3 forecasting adaptation, not a recreation of every original Hundman setting.

No new detector hyperparameter selection or TRAIN calibration is needed. Hundman's historical dynamic rule adapts to **unlabelled TEST residuals**, including future startup averaging and batch availability. That existing behaviour is retained and disclosed; it is not described as a causal instantaneous or TRAIN-fixed-threshold detector. Forecasts remain causal through t−1. Annotation parsing is gated until the new score seal verifies; historical exposure and the presence of prior label-derived reports are explicit.

## Complete pipelines and exclusions

[GDN's pinned official source](https://github.com/d-ailin/GDN/tree/9853899da860682669a134e4af315d036aab4eca) supplies the forecasting architecture. Its existing native NASA score uses CAL median/IQR, denominator 0.01, four-point causal averaging and max dimension. [CATCH's pinned official source](https://github.com/decisionintelligence/CATCH/tree/3647c69be5eb56649b072596cf89098e689e20c3) supplies the reconstruction model. Its existing NASA adaptation uses time/frequency reconstruction evidence at each causal window's right edge, preserving its stochastic evaluation-mask protocol. Both keep the frozen EXP013 TRAIN-selected EWMA/quantile detector. These are documented NASA adaptations, not reproductions of the papers' complete threshold/preprocessing/leaderboard protocols. Their counts, flags-derived event reports and conclusions are reused unchanged.

A CATCH–Hundman hybrid would additionally alter thresholding over a reconstruction score with frequency weighting and different support, rather than isolate forecasting architecture. It would require a separately justified scale/calibration protocol for Hundman's absolute guards. It adds no essential comparison here and is not executed.

[STGLR, Lai et al., Sensors 25(2):310 (2025)](https://doi.org/10.3390/s25020310) uses dynamic graph learning, GraphSAGE, attention/GRU and VAE reconstruction with POT. Its [primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11769452/) links the NASA data in its availability statement. No verifiable author implementation or frozen checkpoint was located in this review. A newly assumed implementation/training protocol would add a fidelity and training study outside this final stage. This is a bounded evidence gap, not proof that source cannot exist or that STGLR is ineffective.

[OmniAnomaly's official source](https://github.com/NetManAIOps/OmniAnomaly) describes stochastic GRU/VAE reconstruction probability, Python 3.5/3.6 and separate best-TEST-F1/POT workflows. Its [requirements](https://github.com/NetManAIOps/OmniAnomaly/blob/master/requirements.txt) pin TensorFlow-GPU 1.12, TensorFlow Probability 0.5 and TFSnippet 0.2-alpha1. No verified existing NASA checkpoint/score stream or validated modern faithful port was supplied. Adopting an unrelated simplified PyTorch VAE would not reproduce the method. A legacy CUDA/TF port plus new training is unnecessary to the minimum comparison and is excluded.

## Evaluation, uncertainty and reporting

Use the exact frozen evaluator bytes in `vendor/frozen/metrics.py`, identical to EXP013's historical evaluator. Retain 81 unique channels, first P-2 label row, 104 inclusive original events converted to half-open intervals, and common support t≥250. T-10 is not added. Unsupported event portions/short-detector status are retained, never dropped to improve performance.

For H hit events, M misses, U wholly nominal alarm runs and FPt flagged nominal samples among N0, event precision is H/(H+U), recall H/(H+M), corrected precision multiplies by (1−FPt/N0), and corrected event F0.5 uses those precision/recall values. Strict point P/R/F1 use actual flags without point adjustment. Nominal starts, occupancy, fragmentation and actual issue-time notifications retain original definitions. An event-overlapping flag can be issued after event end; late notifications and misses remain visible.

Pool channel counts within each original model seed, then average seed-level ratios. Never turn the one GDN/CATCH seed into three observations. AP is channel-macro, not pooled across incomparable scales. Report all original count denominators and per-seed timing medians.

Use the frozen EXP012 descriptive paired-channel bootstrap convention: 10,000 draws, seed 20261005, 95% percentile intervals, all model seeds grouped. Common panel contrasts are GCN−LSTM and GCN−GDN; complete-pipeline contrasts are GCN−native GDN and GCN−native CATCH, for each mission. Event, point and alarm contrasts are exploratory, with no multiplicity-adjusted confirmatory claim. Shared mission fitting/context and one baseline initialisation limit independence and omit optimisation uncertainty.

Keep common-detector NASA evidence, complete-pipeline NASA adaptations and original-paper results separate. The released data inherit TEST-extrema scaling, and the research has historical TEST/label exposure. Neither the new seal nor TRAIN-only additional scalers erase those limitations. Preserve negative comparisons, exclusions and failed feasibility/replay receipts. Do not infer superiority, equivalence, independent generalisation or operational alarm rates from this retrospective stage.

**Stopping rule:** two frozen GDN inference jobs; one additional detector; no training, architecture changes, calibration search or further model additions. After verified export, hand the results to the report writer without rewriting EXP012/EXP013 conclusions.
