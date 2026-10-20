# EXP009_DETECTOR_V1 — nominal-constrained residual detector development

## Scientific decision

Proceed to detector development. Forecasting architecture and weights remain closed. EXP008 establishes numerical readiness, not a successful detector. EXP009 asks: **Can one small causal residual detector per mission satisfy an explicit nominal alarm budget while retaining sensitivity to a broad, fixed residual-disturbance bank?**

This is a surrogate development study. It can produce a provisional operational detector, or a defensible no-selection result. It cannot estimate real spacecraft anomaly recall, prove the selected detector is optimal, or restore a virgin NASA TEST. No additional forecasting fit or GPU is required.

The input is the byte-locked EXP008 export (SHA-256 in config.json). All 534 replay archives and parent links are reverified. No dataset archive, checkpoint loading, NASA TEST loader, anomaly-label reader, final TRAIN calibration loader, or network access exists in the runner. The bundle includes the small residual export; it does not include raw NASA TEST.

## Evidence and motivation

An independent local audit verified all 534 saved prediction archives/markers, their identities and recorded exact-replay assertions, and recomputed the 288 EXP008 nominal rows. This did not independently rerun GPU checkpoint inference. The q=.99/.995 thresholds were identical in 50% of the earlier-origin MSL GCN policies and about 39.6% of SMAP GCN policies (seeds/origins/spans counted). Stronger tail modelling cannot manufacture effective independent tail observations. EWMA also cannot guarantee removal of nonstationarity.

The proposed alternative uses central robust statistics and a bounded multiplier search, then checks alarms on later segments. Its thresholds are empirical operational choices: **not Gaussian sigma probabilities, conformal coverage, guaranteed false-positive rates, or EVT return levels.** Dependence and drift remain measurable limitations.

## Data boundaries and honest independence claims

Use the 81-channel roster already frozen upstream; 27 MSL and 54 SMAP. No channel is excluded for an unfavourable detector result. Keep the three seeds 17,29,43; never select a seed or average predictions into a newly tuned ensemble.

For canonical saved-weight residuals, N is the original full TRAIN length; boundaries are floor(fraction*N):

| Use | Targets |
|---|---|
| Existing forecasting fit | [0,.60N), unchanged |
| Detector reference statistics | [.60N,.70N) |
| Detector candidate selection | [.70N,.75N) |
| Locked detector audit | [.75N,.80N) |
| Final detector calibration | [.80N,N), EXCLUDED from EXP009 |

Earlier-origin fixed-GCN replays are audit-only:

| Origin | Detector reference | Detector audit |
|---|---|---|
| R1 | [.50N,.55N) | [.55N,.60N) |
| R2 | [.60N,.65N) | [.65N,.70N) |

These cover six mechanically selected channels per mission and seeds 17,29. They are existing earlier-origin weights, not newly refitted final weights. Both origins must pass separately.

Important limitations: canonical [.60,.80) was used for forecast early stopping; earlier origins informed architecture robustness decisions; historical project analysis has exposed NASA TEST; final TRAIN calibration was used in historical detector work. These are reused development records. Splitting them now controls **new detector selection**, not upstream model-selection dependence. The audit is a detector reservation, not an independent prospective holdout. Future publication must retain this distinction.

Compute residual e_t = float64(y_t) - float64(prediction_t), in the existing fit-standardised telemetry units. Do not rescale using TEST or combine unrelated channels into one residual distribution.

## Frozen 12-candidate family

Reference location m = median(e). Reference scale s = max(1.4826 median(|e-m|), median(|e|), 1e-6). The floor is numerical protection in fit-standardised units; report floor incidence and absolute injected amplitudes. Do not interpret near-zero channels as evidence of sensitivity to physically meaningful tiny faults.

Let z_t=(e_t-m)/s. Three causal scores:

1. **absolute:** |z_t|.
2. **ewma10:** causal EWMA of |z_t|, alpha=2/11, initialized to the first available residual score.
3. **multiscale:** max over w in {1,8,32} of |sum(z_{t-w+1:t})|/sqrt(w). Only full windows are used; startup uses available shorter windows. sqrt(w) is a fixed sensitivity weighting, not an independence or normality claim. This branch targets coherent signed errors while retaining isolated-error detection.

For each score, reference centre a=median(score) and width b=max(IQR(score)/1.349,0.1*a,1e-6). Threshold a+h*b, with h in {2,4,8,16}. Strict score > threshold. No hysteresis, alarm merging, retrospective padding, point adjustment, refractory interval, TEST adaptation, or automatic parameter updates.

A single family and multiplier is selected per mission. Per-channel/seed location, scale and threshold are computed independently by the same rule. No per-channel family search. Mission-specific selections are allowed; calibration remains channel-specific. MSL/SMAP share the code and search budget.

State starts at the first saved stopping-block target and carries forward causally. Reference constants are estimated before scoring an assessment; their use within the reference block is calibration, not a prequential prediction claim. No use of later residual values in earlier scores. Canonical selection calculations truncate before the reserved audit segment.

## Selection objective and operating budget

The following numbers are prespecified research operating requirements, not a spacecraft operator's approved alarm-cost specification. Changing them after seeing outcomes requires a new exploratory version, not a successful EXP009 claim.

Average repeated seeds within a channel, then weight channels equally within each mission. A candidate must have:

- mean nominal flagged occupancy <= 1%;
- mean alarm starts <= 2 per 1,000 samples;
- channel 90th percentile occupancy <= 5%;
- channel 90th percentile alarm starts <= 10 per 1,000 samples;
- each seed's mission mean occupancy <= 1% and starts <= 2/1,000.

A start is a false-to-true transition. Carry the preceding flag across assessment boundaries; an ongoing alarm is not counted as a new start. No calendar-time conversion without a verified sampling cadence. TRAIN is presumed nominal, not certified anomaly-free: call these **nominal alarm burden**, not measured false-positive rates. Report durations, channel outliers and raw counts from the saved flags/statistics where available; do not silently remove difficult channels.

Synthetic sensitivity prevents the trivial never-alarm solution. It is not a substitute for labelled anomaly validation.

## Fixed residual stress bank

Inject one disturbance per copied residual trace, leaving the reference and thresholds unchanged. This is a detector-level counterfactual; **predictions are NOT recomputed**, and changes do not feed back through telemetry history. Thus this tests how a detector uses residuals, not how a complete forecaster reacts to sensor faults, stuck values or command anomalies.

Selection families: spike length 1, constant shift length 8, constant shift length 32, linear ramp length 32 (from amplitude/32 to full amplitude). Both signs, amplitudes {2,4,8} times a common reference residual scale, two placements at floor(1/3 or 2/3 of available onset range): 48 probes per stream/policy. The common scale is the median of the fixed-GCN reference scales over available seeds for that channel/origin; controls use the same additive amplitudes. There is no cherry-picking of unflagged locations. Durations are capped at assessment length, explicitly recorded alongside requested duration. This rule keeps the short T-9 (MSL) and D-12 (SMAP) channels: their canonical selection/audit segments contain 22/22 and 16/15 targets respectively. Do not drop them or interpret their shortened probes as full 32-sample events. Preflight requires at least 16 reference and 8 assessment targets. These length-based rules were determined from split metadata before running detector selection.

Audit adds alternating-sign and residual-variance perturbations of length 32 with the same amplitudes/signs/placements: 72 probes per stream/policy. Variance probes add sign*amplitude*(e-local_reference_location); these are model-relative diagnostic stresses, not physically matched injections across forecasters. Held-out families do not choose a fallback candidate and do not become extra hyperparameters.

A probe counts as an incremental event hit only if the injected event has at least one alarm and the unmodified event interval has none. Otherwise its incremental hit is zero. This conservative metric does not reward an already noisy detector. Report baseline-hit frequency, incremental recall by family/amplitude/sign, and delay; non-detections are censored at duration (not dropped). Duplicated placements caused by short segments stay as deterministic repeated probes; never treat them as independent evidence.

At amplitude 8, worst selection-family incremental recall must be >= .50. This is a weak anti-degeneracy requirement, not a claim of acceptable real-anomaly recall. Among candidates passing all gates, rank by (1) largest worst-family incremental recall averaged across all amplitudes; (2) largest mean family recall; (3) fewer mean alarm starts; (4) less occupancy; (5) frozen enumeration order: absolute, EWMA10, multiscale, and h ascending. No stochastic seed search.

## Staging, controls and fit budget

1. Verify package/input hashes and synthetic unit checks. Record environment and freeze it for resume.
2. Evaluate 12 candidates on 243 canonical fixed-GCN streams: **2,916 selection units**.
3. Commit DETECTOR_CANDIDATE_FREEZE.json before auditing. At most one candidate per mission; zero is allowed.
4. Apply the selected candidate and two reference policies (raw |e| q=.99, EWMA10 |e| q=.995, higher empirical quantile, strict >) to canonical reserved audit streams: 243 GCN, 243 matched message-removed, 81 unique persistence. These policies are newly calibrated on the declared reference blocks; they are NOT reproductions of historical TEST scores.
5. Apply the same three policies to 48 earlier-origin GCN and 24 unique persistence streams. No missing earlier-origin message-removal control is invented.
6. Both audit stages and each earlier origin must satisfy the same nominal/high-amplitude gates on GCN streams. Controls and new stress families are descriptive only; no demand that GCN wins.

Maximum **4,833 detector units**, **277,992 residual probes**, **zero model fits**, no GPU inference. If a mission has no selected candidate, audit only its two reference policies and return NO_FEASIBLE_CANDIDATE. If its selected candidate fails any audit gate, return AUDIT_FAILED_NO_FALLBACK. Do not try the runner-up. If all gates pass, return PROVISIONAL_FREEZE_REQUIRES_REVIEW. No terminal status authorises NASA TEST execution.

Each selected policy is applied unchanged to no-message and persistence controls with their own reference statistics. This isolates whether frozen residual information is useful under a common detector. These controls are not separately detector-optimised competitors. EXP005 and external methods remain useful later controlled benchmarks; do not block this small detector study waiting for them or claim SOTA without them.

## Statistics

Mission results remain separate. Unit of comparison is channel, with seeds averaged within channel; early origins are repeated conditions, not new independent channels. Save per-seed, per-channel, per-family and per-origin raw rows. Primary selection is a constrained decision, not a significance test.

Report paired selected-minus-reference nominal differences with 10,000 channel bootstrap replicates (seed 9009), percentile 95% intervals and leave-one-channel-out ranges. Intervals are descriptive and conditional on development: channel independence is not guaranteed and selection uncertainty is not eliminated. Do not bootstrap synthetic probes as if they were independent spacecraft events. For a final report, use saved tables to inspect channel-family concentration and seed dispersion; do not alter selection after that inspection.

The corrected EXP008 AP/event module is copied unchanged for continuity and its tests must pass. EXP009 does not calculate NASA AP/F scores. Synthetic incremental recall is separately named and cannot be presented as NASA corrected event recall. No historical AP output is rewritten.

## Freeze/stop and next step

A pass freezes only the detector family/multiplier and calibration rule provisionally. An audit failure stops that mission's promotion; no larger threshold grid or injected-pattern redesign in this version. Preserve failed outcomes as informative evidence. One documented, separately reviewed revision could be justified by a concrete failure mode; do not permit repeated unbounded attempts.

Before final NASA evaluation:

1. Independent researcher reviews code, arbitrary operating-budget choices, stress assumptions, results and negative controls. Resolve serious metric/data issues; historical AP bug remains disclosed and unchanged.
2. For a passing candidate, perform a SMALL versioned **raw-telemetry perturbation audit** with frozen weights and the already locked detector (not a new search): perturb TRAIN telemetry causally, retain commands, replay changed inputs, include spike/offset/ramp/stuck/variance examples. This validates end-to-end sensitivity and reveals forecast adaptation. Prespecify positions/amplitudes without TEST events. This needs the frozen GPU environment, unlike EXP009. Residual-only results cannot replace this check. Failure permits stopping or an explicitly new exploratory study, never silent retuning.
3. Freeze the entire detector and common baseline comparison protocol before accessing final calibration. Compute reference statistics/thresholds on the final 20% TRAIN once; no new choice of family, h, smoothing, event merging or seed. It is historical calibration, not a new independent validation set. Diagnose invalid/nonfinite values as engineering failures, not an excuse to choose a better TEST threshold.
4. Corrected metric regression checks, index alignment, score/hash manifest and label-blind score export precede the separately authorised **retrospective** NASA TEST evaluation. Keep target index >=250 for comparability. Freeze primary corrected event F0.5 plus event precision/recall and alarm burden; report strict point metrics and tie-correct AP as secondary. Any change to the existing event metric requires a separately versioned protocol before labels. Never select the next detector from TEST comparisons.

EXP009 deliberately omits final calibration and NASA evaluation code. We are not yet scientifically ready for that final step.

## Why not the other options now?

- EVT/POT, flexible tail mixtures: too few effectively independent tail observations; richer assumptions require independent diagnostics, not merely more fitted parameters.
- Conformal/e-process claims: dependence, repeated model selection and drift invalidate casual exchangeability claims. Blocking alone does not create universal validity. A future method must state its actual assumptions.
- Online adaptation: can absorb anomalies into the reference distribution; avoid until there is an independent way to validate the update rule.
- Supervised classifier: permitted by the broad project brief, but no independent labelled anomaly-development set currently exists. Splitting historically exposed NASA TEST now does not remove prior exposure. External/domain-expert labels would materially improve this situation.
- More architecture tuning: no evidence here warrants reopening EXP001–EXP007.

## Targeted sources (not a repeated literature review)

Goswami et al., *Unsupervised Model Selection for Time-series Anomaly Detection*, https://arxiv.org/abs/2210.01078 — synthetic anomaly performance is an imperfect surrogate, not a replacement for independent real labels.
Chernozhukov et al., *Exact and Robust Conformal Inference Methods for Predictive Machine Learning With Dependent Data*, https://arxiv.org/abs/1802.06300 — time dependence requires an explicit validity argument.
Cobb et al., *Sequential Multivariate Change Detection with Calibrated and Memoryless False Detection Rates*, https://proceedings.mlr.press/v151/cobb22a.html — false-detection calibration is itself a substantive methodological problem.

The specific score family, operating budget, stress grid and gates above are introduced for this project; they are not claimed as the published authors' procedure.
