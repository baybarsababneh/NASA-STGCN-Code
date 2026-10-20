# EXP011_FUSION_PILOT_V1 — frozen bounded feasibility protocol

## Decision and scope

Test whether a small, synthetically supervised detector can distinguish injected faults from nominal transitions using the **joint behaviour of observations and the frozen forecaster**, beyond residual magnitude alone. This is a feasibility pilot, not a detector promoted for NASA evaluation. No forecasting retraining, new forecast replay, final TRAIN calibration, NASA TEST, or NASA anomaly labels are allowed.

The design is newly proposed after inspecting EXP010. It is not a pre-EXP010 hypothesis or a prospective holdout. Existing development channels and synthetic mechanisms have influenced the design. Holding a channel out of classifier fitting and threshold selection prevents direct cross-channel fitting leakage, but does not erase this historical design exposure. Synthetic discrimination does not establish real-fault performance.

One bounded pilot is authorised. Do not introduce alternate classifiers, features, thresholds, amplitude ranges, loss weights or regularisers after observing its results. Failure ends this automated development branch; it is not permission to search for a passing variant.

## Evidence motivating the question

EXP010 retained instantaneous spike evidence but attenuated sustained corruption after it entered the causal lookback. It did not establish that all sustained faults are detectable from residuals, that an onset can reliably identify a fault, or that a fault remains present until recovery. Legitimate telemetry transitions can generate the same signals.

A detector can observe the current telemetry, past telemetry, the one-step forecast and its recent response. These contain more information than |residual| alone. A small classifier can test whether their joint pattern discriminates the prescribed synthetic faults from actual development-stream transitions. It cannot recover information absent from these measurements. Do not latch alarms indefinitely to infer fault duration, gate out all command transitions, use recursive forecasts outside their trained objective, or assume a statistical tail model repairs a nonstationary residual process.

## Input and data roles

The sole numerical input is the exact EXP010 full results ZIP identified in input.json. The runner verifies its SHA256, output manifest, all 3,078 unit commits and saved arrays. The 114 nominal parent replay assertions must pass. A frozen copy of the 114-entry parent plan supplies mission, channel, seed, role, and original TRAIN length. No dataset archive, original labels, TEST directory, or forecasting framework is needed.

Mechanical development cohort, six channels per mission:
- MSL: C-1, P-10, M-7, P-14, T-13, D-16.
- SMAP: P-3, A-8, G-6, E-1, G-4, A-5.

The seven EXP010 mechanism-enriched channels are a separate posthoc stress audit: MSL M-4, M-3, T-8; SMAP G-3, G-7, F-3, D-11. They never enter classifier fitting or threshold selection. Their enrichment prevents interpreting their aggregate as a mission estimate.

Each channel retains original indices and fit-only telemetry standardisation. Training negatives on permitted training channels use nominal saved targets from floor(.6*N)+32 to floor(.8*N). The first 32 samples warm up causal features and are excluded from classifier fitting. Outer assessment is floor(.7*N) to floor(.8*N). This is reused forecasting-development data, not new unseen TRAIN data. Final 20% remains inaccessible.

Synthetic positives use the existing end-to-end interventions: spike, offset32, ramp32, amplitudes 0.5 and 2 in fit-standardised telemetry units, both signs, both positions. All affected samples are positive, but each event has equal total weight, so a 32-sample event does not receive 32 times the weight of a spike. This deliberately tests the difficulty of absorbed sustained corruption; no oracle recovery or counterfactual difference is a detector input. Stuck-value probes are excluded from training and threshold selection and retained as an out-of-family audit. Null interventions are counted/excluded, never treated as misses.

The nominal class uses original, unmodified streams. The positive class uses separately corrupted copies. Recovery samples are not relabelled as nominal training observations. Original nominal streams are only *assumed nominal*; alarm counts on them are nominal alarm burden, not verified false positives.

## Four fixed detector arms

R: GCN residual features only.
F: GCN evidence fusion, the only arm eligible to advance this GCN detector branch.
N: identical fusion using the frozen matched message-removed forecasts.
P: identical fusion using persistence forecasts, requiring no learned forecasting model.

At time t, e=y-prediction. All transformations use saved fit-standardised units, with no MAD/IQR denominator or near-zero residual floor. Apply asinh to each nonnegative magnitude below.

R has three predictors:
1. |e_t|.
2. |mean of e over the trailing eight samples|.
3. maximum |e| over the trailing eight samples.

F/N/P add:
4. |y_t-y_(t-1)|, using saved persistence as y_(t-1).
5. |prediction_t-prediction_(t-1)|.
6. |y_t-y_(t-8)|.
7. trailing-eight telemetry range.
8. trailing-eight fraction of exactly unchanged telemetry samples (untransformed).

All windows are causal and include t. Partial windows are allowed at the saved-trace boundary; warm-up exclusion keeps these out of fitting/assessment. Lagged values are left-padded only at that boundary. The first forecast difference is zero. No selected command-context feature is introduced: EXP010 context counts/directions do not yet justify one shared conditioning rule. Mission and channel identity are not classifier inputs.

One logistic linear classifier with an intercept is fitted per training-channel set, mission, arm and forecast seed. Score is the linear logit, not a calibrated probability of a real spacecraft fault. Weighted objective:

sum_i w_i [log(1+exp(score_i))-label_i*score_i] + 0.005 ||beta_nonintercept||^2.

Each class receives total weight 0.5. Channels receive equal class weight; within the synthetic class, the 24 events receive equal weight. No standardisation learned from held-out channels. L-BFGS-B, zero initial coefficients, maxiter=1000, ftol=1e-12, gtol=1e-8, maxls=40. Numerical failure stops the run, never changes the optimiser. This small hypothesis class is a purposeful bound, not a claim to optimal classification.

The detector family/regularisation is shared across missions; coefficients and threshold decisions are mission-specific. Frozen forecast seeds 17,29,43 are all retained, never selected. Persistence is fitted once and reused across seed-labelled comparisons; these are not independent persistence replications.

## Nested channel exclusion and threshold selection

For each mechanical outer-held channel:
1. Exclude it entirely from fitting and threshold selection, across all seeds.
2. On the other five channels, obtain inner leave-one-channel-out scores: each inner model is fitted on four channels and scored on the fifth.
3. Pool inner results with equal channel weights after seed averaging. Evaluate only logits h in {0,2,4,6,8}, with strict score > h. These fixed cutoffs replace channel-wise extreme-quantile calibration; they are not estimated tail probabilities.
4. Choose a feasible cutoff using the rules below. If none is feasible, record an abstention, output no alarms for that outer fold for bookkeeping, and fail the arm's pilot gate. Do not reinterpret abstention as a successful low-alarm detector.
5. Fit on all five permitted channels and apply the chosen cutoff to the outer-held channel, without any target-channel adaptation.

No hysteresis, gap merging, minimum duration, cooldown, alarm padding, retrospective boundary extension, online normalisation or threshold adaptation. Contiguous positive sample runs are alarms. No inferred full-duration fault state is produced.

For the separate enriched-channel audit, choose a cutoff from the six-channel leave-one-out scores, fit on all six mechanical channels, and score the seven enriched channels. This audit cannot rescue a failed mechanical pilot.

## Gates and ranking

Retain EXP009's numerical nominal gates:
- equal-channel mean alarm occupancy <=1%;
- equal-channel mean alarm starts <=2 per 1,000 samples;
- channel p90 occupancy <=5%;
- channel p90 alarm starts <=10 per 1,000 samples;
- each seed also satisfies the two mission mean limits.

Channel rates are computed on the assessment interval; a flagged first sample counts as an alarm start. p90 uses pandas' linear interpolation. Report integer counts and exposure alongside rates.

Sensitivity gate: minimum, across spike/offset32/ramp32 at amplitude 2, of the equal-channel, seed-averaged **incremental event hit fraction** >=0.5. A hit must occur in [onset,end); recovery-only alarms do not count. Incremental hit requires at least one perturbed-stream alarm and no nominal-stream alarm in the same interval. Each sign/position has equal weight. Report unconditional hits, background hits and conditional-clean hit rates as well; the incremental gate is not pure detectability.

This is a new end-to-end sensitivity task. Amplitude 2 in fit-standardised telemetry units is NOT EXP009's amplitude 8 in a residual scale; passing cannot be described as retrospectively satisfying EXP009. No historical requirement is changed. The old failed detector families remain failed.

Among feasible inner cutoffs rank: largest worst-family incremental fraction; largest mean family fraction; lowest nominal starts; lowest occupancy; largest cutoff. No other tie-break or fallback.

An arm passes the mechanical pilot only if every outer fold obtained a feasible inner cutoff and the combined outer results satisfy the same burden/sensitivity gates. F is eligible for all-channel TRAIN-only confirmation only if it passes. Controls are never silently substituted as a winning GCN detector. A stronger P or N result limits the scientific claim even when F passes.

## Budget and stop rules

Two missions × [15 four-channel subsets + 6 five-channel subsets + 1 six-channel set] × [three forecast-based arms × three seeds + persistence once] = **440 unique detector fits**. Identical training subsets are cached; no duplicated inner fit is needed. There are zero forecasting fits, zero new forecast passes and zero GPU requirements. Each enriched channel is scored, not trained upon.

The notebook does not claim a wall-clock duration in advance. It records fit elapsed time and convergence diagnostics. It makes no model-selection decisions requiring user intervention.

Mission outcome:
- F passes: ELIGIBLE_FOR_TRAIN_ONLY_CONFIRMATION, not detector-selected or ready for NASA TEST.
- F fails: STOP_REQUIRE_NEW_INFORMATION. Stop this automated search; obtain independent domain review/annotations of nominal transitions or additional development exposure before proposing a materially new method. Failure does not prove all possible detectors impossible.
- Numerical/integrity failure: engineering stop, versioned correction only.

There is no additional tuning round within EXP011. Do not relax gates, remove difficult channels, select a seed, lengthen allowed delay, or increase the search after seeing failure.

## Analysis and audit

Report per mission, channel, seed, arm and fault family: occupancy, alarm starts, exposure, nominal overlap, unconditional/incremental hits, censored delay, and null-case counts. Delays are samples, not seconds. A miss is censored at event duration; do not present mean censored delay as mean detection delay among detected events. Stuck and amplitude-0.5 results are diagnostics, not selection gates.

Compare F with R, N and P using paired channel resampling after seed averaging: 10,000 bootstrap draws, RNG seed 11011, occupancy, starts and worst-family incremental difference. With only six mechanically selected channels per mission, these intervals are descriptive and weak; seeds, injected cases and repeated telemetry are not independent sample sizes. Show all six channel results and seed-specific outcomes. No significance-based gate or SOTA claim.

Audit source/input hashes, 440 unique committed fits, exact roster/set/arm identities, coefficient finiteness/convergence, equal class/channel/event weights, no outer-held channel in fitting/calibration, no enriched-channel influence, threshold rule recomputation, abstention treatment, all comparison arms and posthoc coverage. Inspect whether gains rely on one channel/family or the telemetry-only control. A pilot pass with unconvincing controls or catastrophic enriched-channel failure warrants review before spending on confirmation; it is not an invitation to edit this pilot.

## Reproduction and execution

CPU-only, single BLAS/OMP/MKL thread; pinned NumPy 2.3.5, pandas 2.2.3 and SciPy 1.17.0. The first run records Python/platform/dependency versions and source/input hashes. Resume requires the same record. Atomic, hashed fit JSONs are the durable units; unfinished fits may be recomputed but committed fits are verified and reused. Drive storage survives Colab process loss; no promise is made that Colab will run unattended indefinitely.

An exclusive ACTIVE_RUN file prevents concurrent writers. A hard runtime interruption may leave it behind. Release only after confirming the previous runtime is terminated, using the notebook's explicit RELEASE_STALE_LOCK switch; the old lock is archived. Never automatically steal a lock based merely on elapsed time. A normal exception removes it. Output integrity manifests accompany final exports.

Engineering checks have passed locally on synthetic fixtures and the full EXP010 input, including a complete 440-fit mock-data orchestration and byte-preserving resume. The actual 440-fit scientific experiment has NOT been run while preparing this package. Colab execution itself has not been exercised here.

## What follows, separately frozen

If F passes, audit this pilot and freeze one all-81-channel confirmation protocol before running it. That protocol must use channel-excluded classifier/threshold fitting and end-to-end synthetic probes, retain all controls and the existing nominal gates, and prohibit architecture optimisation. Do not use the six-channel pilot rates as mission-wide performance estimates. No full-channel expansion is authorised automatically by this notebook.

Only after confirmation and researcher review should the final detector coefficients/rule be frozen. Keep the untouched final TRAIN block for a one-time nominal qualification of that frozen rule, rather than repeatedly using it to redesign/calibrate and validate the same detector. Failure at qualification remains a reported failure; it does not trigger threshold retuning. Short exposure may support only a limited or inconclusive claim of alarm control.

NASA TEST remains a separately frozen, explicitly retrospective evaluation. Freeze corrected metrics, score generation, thresholds, event definitions, controls and reporting before labels are read. Prior historical TEST exposure must remain disclosed. Historical AP outputs and EXP001–EXP010 are preserved. No final NASA evaluation is part of EXP011.
