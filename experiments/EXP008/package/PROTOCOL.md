# EXP008_READINESS_V1 — frozen detector-readiness and nominal-calibration feasibility protocol

Status: implementation specification frozen before EXP008 residual inspection. No EXP008 GPU results exist at preparation. Parent: completed EXP007_GPU_V1 E01, run hash 45702dc9b498dd1ecb9d0f2d115355ad0c913a288bf017956a194091a7f58a45. This protocol does not reopen architecture selection, train a model or authorise NASA TEST evaluation.

## 1. Question and why this is the next stage

Can the frozen GCN weights reproduce their recorded forecasts, and are a small, explicit family of causal residual thresholds numerically well defined and temporally stable on earlier presumed-nominal TRAIN blocks?

This is the first detector-development stage, but not a search for a best anomaly detector. The byte/metadata audit did not numerically replay predictions. Moreover, with no independent labelled development anomalies, minimising TRAIN alarms can favour an unresponsive detector. Nominal burden alone cannot identify a sensitivity–specificity optimum. Complete the replay and calibration feasibility work while independent review proceeds, then make the operating-policy decision with its limitations explicit.

Historical TEST results motivated examining alarm burden and EWMA. That is historical human exposure and is disclosed. The four-cell family below is an exploratory, historically informed proposal, not a prospectively blind hypothesis. New parameter values, channels, seeds or stopping rules may not be chosen from NASA TEST outcomes. No operational false-alarm target has been supplied by a spacecraft operator.

## 2. Frozen forecasters and budget

- MSL: selected fixed EXP002-style GCN, RF7, hidden32, lookback64.
- SMAP: selected fixed g09 GCN, RF31, hidden64, lookback64.
- Graph support, equal incoming weights, scaling, command inputs, parameters and seeds are unchanged.
- No dynamic gate, learned edge search, refitting, dropout change, seed selection or capacity search.
- 486 canonical checkpoint jobs: final GCN and matched message-removal models, 81 channels ×3 seeds 17/29/43 ×2 roles.
- 48 existing earlier-origin fixed-GCN jobs: 12 original mechanical screening channels ×2 origins ×seeds 17/29. These already use the final architecture for their mission.
- Total: 534 checkpoint replay jobs, zero new fits. Two identical inference passes per required interval. Canonical jobs have one stopping/validation interval; earlier-origin jobs have stopping and assessment intervals. Thus 1,164 interval passes, not 1,164 fits. All use original batch256 and float32 CUDA path.
- Persistence is computed without fitting, once per channel/origin. It does not acquire spurious independent seeds.

The matched-removal canonical checkpoints are replayed for readiness, not compared using their stopping-set alarm rates. No new earlier-origin removal models or broad external baselines are needed for this feasibility question.

## 3. Input and data-use boundaries

The self-contained bundle contains the unchanged selected parent fit ZIPs/commit records, original model source and provenance, and copies of TRAIN prefixes only. Each prefix ends at floor(.8N), preserving original dtype/values. Its original full-array hash, prefix hash, full N and prefix length are recorded. The final 20% of every TRAIN array is physically absent. NASA TEST arrays, labels and historical score arrays are absent.

Stage A may inspect prefix values only to reconstruct parent fit-only scaling/graphs and replay stopping/assessment predictions. Canonical validation [.6N,.8N) was used for early stopping and architecture selection: it is not detector holdout data and is not used to rank scorers here.

Stage B uses existing earlier-origin checkpoints:

| Origin | Parent model fit | Parent early stopping | Pseudo-calibration | Forward nominal assessment |
|---|---|---|---|---|
| R1 | [0,.4N) | [.4N,.5N) | [.5N,.55N) | [.55N,.6N) |
| R2 | [0,.5N) | [.5N,.6N) | [.6N,.65N) | [.65N,.7N) |

Every boundary is floor(fraction × original full N). Assessment targets follow all data used to fit/stop that origin's model and its pseudo-calibration block. Observed past telemetry may enter causal input windows across boundaries. No target at t or later enters the prediction of t. Scoring y_t after it arrives is allowed.

The two origins and the screening channels already informed architecture robustness. They overlap other fitting blocks. They are reused internal development data, not new independent confirmation. The mechanical roster is fixed, not selected by final anomaly performance: MSL C-1, P-10, M-7, P-14, T-13, D-16; SMAP P-3, A-8, G-6, E-1, G-4, A-5.

TRAIN is treated as presumed nominal, not independently certified fault-free. Flagged TRAIN samples are nominal flags, not verified false positives. No NASA anomaly precision, recall, F-score or AP is estimable from this study.

## 4. Stage A — prerequisite numerical replay

Before any nominal diagnostics:

1. Verify the complete input file roster/SHA256, parent freeze, checkpoint archive and embedded result/job/config identities.
2. Check the actual CUDA environment equals the complete parent fingerprint. A100-SXM4-40GB, Python3.13.15, torch2.11.0+cu130, CUDA13.0, cuDNN91900, NumPy2.1.3, pandas2.2.3 and driver580.82.07 were recorded. No CPU, different-GPU or mixed-environment fallback.
3. Restore model state strictly, reproduce original fit-only target mean/scale, graph buffer and feature orientation. Model code is copied unchanged.
4. Replay with original 256-example batches and PyTorch float32 MSE reduction. Require exact equality to recorded stopping/assessment MSE and exact equality of both repeated prediction arrays. These are same-weight, same-environment inference checks, not CPU/CUDA training-equivalence tests.
5. Confirm state tensors are unchanged by inference. Commit the predictions and verification record atomically, hash-verify the saved bytes and write the completion marker last.
6. Require all 534 jobs before Stage B. Any mismatch halts the study. Preserve failures. Do not relax tolerances, reinterpret a failed replay as a new forecast result, refit weights or resume downstream without a reviewed engineering diagnosis.

If the exact parent environment is no longer obtainable, stop and register a separate inference-portability amendment before any residual analysis. That does not reopen architecture tuning. A replay mismatch can be an implementation/environment problem; it is not automatically evidence against the scientific architecture.

## 5. Stage B — four prespecified descriptive policies

Residual r_t = |float64(y_t) − float64(prediction_t)| in the existing fit-standardised target units. Higher residual means more anomalous. Use the same score construction for GCN and persistence.

- Raw: s_t = r_t.
- EWMA10: alpha=2/11; s_first=r_first; s_t=alpha*r_t+(1-alpha)*s_(t-1). Start at the first stopping target of the relevant origin and continue through pseudo-calibration and assessment. Earlier stopping residuals initialise state only, not thresholds. Never reset at the assessment boundary. No centred smoothing or use of future residuals.
- q ∈ {.99,.995}, independently calibrated for each model/channel/seed/origin/scorer using only its pseudo-calibration scores.
- threshold = NumPy quantile(scores,q,method='higher'), equivalently sorted score at zero-based ceil((m−1)q).
- Flag iff score > threshold. No point adjustment, event padding, alarm merging, hysteresis, minimum-run filter, adaptive threshold updating or magnitude clipping.

All four combinations are reported. Raw/q=.99 is a historical continuity reference, not a newly selected final primary detector. EWMA10/q=.995 cannot be promoted because it looked favourable in the historical TEST sensitivity table. No q, span or mission-specific scorer is selected by this script.

Sparse quantiles are a central endpoint. These pseudo-calibration blocks contain 35–216 samples by design. For m<101, both quantiles select the maximum; for m<201, q=.995 selects the maximum. Ties can make thresholds identical even when ranks differ. Record selected order rank, number of available higher ranks, unique scores, observed exceedances, maximum-threshold status and pairwise equality. The IID-continuous maximum exceedance reference1/(m+1) is reported only as a resolution illustration. It is not a temporal false-alarm guarantee. Do not fit a tail distribution or pool channels to hide inadequate resolution.

The canonical final 20% has 63–862 samples in the existing data manifests, but is not opened here. A future q=.995 choice is therefore not automatically justified either. Required calibration size depends on dependence, stability and the intended operating objective, not merely on having one tail observation.

## 6. Endpoints, controls and statistical scope

Primary descriptive endpoints: nominal flagged-sample fraction and alarm starts per 1,000 samples on the forward assessment block. Count a start only on a false-to-true transition, carrying the previous flag across the boundary. Also record intersecting alarm runs and longest run. Samples are not converted into hours because timing metadata are inadequate.

Secondary diagnostics: calibration/assessment median and maximum score, threshold change between chronological halves of the pseudo-calibration block, seed/origin variability, constant-score cases and quantile resolution. Half-block thresholds are diagnostics only and never replace the specified full pseudo-calibration threshold.

Two forecasters enter nominal diagnostics: frozen GCN and persistence. There are48 GCN origin/seed streams ×4 policies =192 rows, and 24 distinct persistence origin streams ×4 =96 rows. Total288. All channels, including ties/constant streams, remain included.

Report MSL and SMAP separately. Average seeds and origins within channel before equal-channel mission summaries. Keep origin/seed tables visible. Descriptive paired contrasts are EWMA10 minus raw at each fixed q for each forecaster. For occupancy and alarm starts, bootstrap the six paired channel differences per mission10,000 times with seed8008 and report percentile95% intervals plus leave-one-channel-out range. Six channels and shared incidents do not justify strong inferential claims. No p-value, multiple-comparison winner, nominal significance gate or seed cherry-picking is permitted. Do not treat24 repeated channel/origin/seed observations as24 independent spacecraft units.

A lower alarm burden can be caused by insensitivity. Never interpret these contrasts as anomaly-detection gains. No synthetic anomalies are used to choose a policy. Synthetic impulses/steps and binary fixtures only test implementation behaviour and event-metric edge cases.

## 7. Metric correction and future evaluation contract

EXP008_METRICS_V2_1 groups tied scores at a common threshold when computing non-interpolated AP. Synthetic ties, permutations, constant scores, perfect ranking and no-positive cases are tested. Local engineering verification also compares 200 random tied-score fixtures against scikit-learn. No-positive AP is explicitly undefined (NaN), rather than silently averaging it as0. AP is not trapezoidal PR-AUC. Historical metric source/results remain unchanged.

Synthetic event helpers preserve the documented overlap counts H/M/U and corrected precision Pe*(1−FPt/N0), with corrected F0.5 and Hundman-overlap F0.5 separate. Undefined populations are explicit. These helpers are not a complete final evaluator; final label import, clipping, event semantics, uncertainty and score export will undergo a separate freeze/test gate before NASA evaluation. No labelled-data adapter is shipped here.

A later benchmark should retain strict pointwise precision/recall/F1/F0.5, tied-score AP, corrected event F0.5, overlap F0.5 as a comparability secondary, nominal occupancy/alarm counts and detection delay where meaningful. Choosing those later metrics or detector settings must not erase the original retrospective benchmark.

## 8. Stop/freeze and possible outcomes

Engineering PASS requires all 534 exact replays, integrity-checked commits, matching parent environment and passing synthetic metrics/scorer tests. Scientific closure requires the fixed288-row inventory, all four policies and both missions, no data-access violations, and complete diagnostic tables. There is no alarm-burden performance gate and no requirement to obtain a favourable result.

Successful terminal state: COMPLETE_NO_DETECTOR_SELECTED. Freeze code, configuration, input hashes, source checkpoint index, environment, replay gate, residual exports, all diagnostic tables and STUDY_CLOSE.json. Stop. A finding that empirical quantiles are unresolved or unstable is a valid and potentially important result. Do not add more quantiles, smoothing windows, new calibration splits or new model fits within this experiment.

Engineering failure: preserve evidence and stop, allowing only a documented versioned correction that does not alter scientific settings. Material methodological feedback from the independent reviewer is handled prospectively in a separate version/stage, not by replacing this result.

## 9. What follows

1. Send EXP008 as an addendum to the researcher, preserving the already sent Core/Full Evidence packages and their hashes. Add replay confirmation, metric correction tests and nominal feasibility evidence, including unfavourable results.
2. With the reviewer, freeze the smallest credible detector policy and operational objective. Without independent labelled development anomalies, either retain a declared nominal-budget rule and report its sensitivity limits, or predeclare a small family as separate endpoints without selecting the best TEST score. Mission-specific policies require a prospective operational/data rationale; six screening channels do not warrant per-mission tuning here.
3. Fix the controlled comparison roster and budgets, including the frozen information-matched EXP005 direct-fusion comparator on the same backend, persistence, the matched removal control and a small justified external baseline set. Do not select external baselines by their performance on the exposed NASA labels. The present matched ablation does not settle those competitive comparisons.
4. Only after reviewer feedback, replay/metric gates and a frozen downstream protocol should the canonical final 20% be used to calibrate final thresholds. It was used historically and is not virgin, but its new role can remain protected from new selection.
5. Hash the final forecasters, scorers, thresholds, metrics, roster and planned contrasts, export NASA TEST scores label-blind, then conduct a single declared retrospective evaluation. Prior TEST exposure remains explicit. There is no irreversible evaluation step in EXP008.
6. Strong external confirmation requires genuinely new independent labelled telemetry. Supervised/semi-supervised downstream components remain permissible if an independent labelled development source and separate assessment data become available. NASA TEST cannot fill both roles honestly.

## 10. Unresolved older issues

The historical AP defect must be corrected in new code before future labelled evaluation; it is addressed by the new synthetic-tested implementation without rewriting old outputs. Missing original EXP001–EXP005 files constrain complete historical replay but do not block this frozen-weight TRAIN-only readiness study. Recover them before claims requiring their reproduction. Architecture optimisation remains closed unless a concrete validity failure is found.

## Primary technical references consulted for this bounded update

- NumPy quantile documentation: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html (explicit method and order-statistic behaviour).
- scikit-learn average_precision_score documentation and official ranking implementation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html and https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_ranking.py (non-interpolated, threshold-grouped AP).
- Chernozhukov, Wüthrich and Zhu (2018), Exact and Robust Conformal Inference Methods for Predictive Machine Learning with Dependent Data, https://proceedings.mlr.press/v75/chernozhukov18a.html (dependent-data validity requires assumptions/methods beyond naive exchangeability).

This study uses empirical quantiles, not a claimed conformal coverage guarantee. These references justify implementation/interpretation constraints, not an anomaly-performance claim.
