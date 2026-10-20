# EXP010_MECHANISMS_V1 — frozen inference-only mechanism study

## Status and question

This prospective computation is specified after inspecting the closed EXP009 TRAIN-only results. Its rationale is documented in DIAGNOSIS.md. It is **post-hoc motivated**, not an independent confirmatory anomaly benchmark.

Question: how do frozen forecasts respond to controlled telemetry corruptions, how does that differ from residual-only injection, and how do nominal residuals vary with causal command context? No detector is selected. No operational alarm limits are changed.

## Fixed models and data

Keep final fixed GCN weights and their matched message-removal controls. Seeds 17,29,43. Use only existing canonical checkpoints and original first-80%-TRAIN prefixes. No optimiser, model fitting, graph reconstruction or scaler refitting beyond exact verification of the original fit scaler. Original selected graph support, input features, model class and GPU deterministic configuration remain unchanged.

Two explicitly separate cohorts:

- **Mechanical cohort**, inherited from earlier studies: MSL C-1, P-10, M-7, P-14, T-13, D-16; SMAP P-3, A-8, G-6, E-1, G-4, A-5.
- **Post-hoc mechanism cases**, chosen from the diagnosed EXP009 failures: MSL M-4, M-3, T-8; SMAP G-3, G-7, F-3, D-11.

Do not pool the enriched cases with the mechanical cohort to claim mission-wide performance. The mechanical cohort is also a reused development roster, not an independent holdout. Final 20% TRAIN and NASA TEST/labels are physically absent from the input package.

## Reference and probe regions

N is original TRAIN length. Boundaries use floor(fraction*N). The original forecast fit is [0,.6N). Existing predictions cover [.6N,.8N). Diagnostic scorer reference is [.6N,.7N). Inject only within [.7N,.8N). All cases in this 19-channel roster have at least 32 samples in that interval, checked before perturbation.

The frozen checkpoints already used [.6N,.8N) for early stopping. This study makes no new independent generalisation claim from those data. Historical NASA TEST exposure remains disclosed.

## Perturbation bank: exactly 26 cases per checkpoint

Each case starts from an independent copy of the original telemetry array. Commands and all other columns remain unchanged. Perturb the observed telemetry before constructing forecasting windows, then reforecast through the original pipeline. Continue to use observed corrupted history, not an invented open-loop roll-out.

For each of spike (1 sample), offset (32 samples), and ramp (32 samples):

- amplitude 0.5 or 2.0 in existing fit-standardised target units;
- sign -1 or +1;
- onset at floor(1/3 or 2/3 of the available onset range within [.7N,.8N)).

This gives 24 cases. A ramp increases linearly from amplitude/32 to amplitude. All additive corruptions end at the declared interval, so recovery effects are retained and measured.

Add two 32-sample stuck-value cases at the same two onset fractions. Replace telemetry with the last pre-onset observation. Do not manufacture additional amplitudes or signs for this family.

The fit-scale amplitudes are controlled stress levels, not validated physical fault magnitudes. If original fit variance used its numerical fallback, retain that provenance and avoid a physical standard-deviation interpretation. Float32 rounding follows the original input representation. Null stuck interventions are explicitly marked and excluded from hit-rate interpretations, never counted as detector misses.

## Controls and outputs

For every probe, compare:

1. Actual reforecast residual e_new = y_new - forecast(y_new history, unchanged commands).
2. Residual-only counterfactual e_old + (y_new-y_old), using the same additive telemetry change with predictions held fixed.
3. The unchanged nominal residual.

Run the matched message-removed checkpoint on the identical corruption. Persistence is computed once per channel/case using the fixed-role seed-17 record, not replicated as three independent observations.

Record actual input L1/RMS, residual-response L1, prediction change, response/input ratio, on-event and post-event response, observed recovery exposure, null-intervention status and target indices. Post-event analysis lasts at most 64 samples or until .8N, whichever comes first. Censored recovery must not be described as complete recovery.

Apply all 12 original EXP009 policies plus its two reference policies, without selecting or modifying any of them. Reference statistics are recalculated identically on unchanged [.6N,.7N) for each model/seed. These failed policies serve only as measurement tools. The GPU environment uses the original NumPy 2.1.3 environment; EXP009's CPU analysis used NumPy 2.3.5. Their historical outputs remain unchanged. Do not claim a new bitwise replication of EXP009 detector tables across those environments.

Report unconditional event hits, nominal-window overlap, incremental hits and censored delay separately. Compare actual versus residual-only hits and responses. Do not calculate a 'best threshold', rank a winning policy, or apply NASA anomaly labels. No AP or NASA F-score is needed for this mechanism question.

Describe causal command context using changes in the already selected incoming command sources over the preceding 1,8,32 samples. At target t, include changes only through t-1. Report counts and residual summaries by context in [.6N,.7N) and [.7N,.8N). Do not condition a threshold on those categories in EXP010, infer a physical subsystem identity, or tune a context window from these results.

## Exact budget

19 channels × 3 seeds × 2 model roles = **114 existing checkpoints**.

- 114 nominal replay units. Each uses the inherited double-prediction exact-MSE/repeatability check.
- 114 × 26 = **2,964 perturbed model-stream units**, one inference pass each.
- Total **3,078 committed units**, with no conditional expansion.
- 19 × 26 = **494 unique persistence probe records**, derived without additional model inference.
- **Zero training fits**.

Counts refer to units, not independent spacecraft observations. Duplicate causal behaviour across seeds is not new temporal exposure.

## Engineering and provenance gates

A100 and the exact governing EXP007/EXP008 GPU environment are required initially. The complete fingerprint and pinned package list are checked. No silent CPU fallback, tolerance relaxation or environment migration.

Before each checkpoint's perturbations, nominal replay must match the stored historical validation MSE exactly and repeat predictions exactly. Verify checkpoint/source/input hashes. Changed forecasts must match nominal forecasts through the onset target (the model cannot yet have seen the altered observation). Beyond corruption end plus lookback 64, forecasts must match nominal again wherever that interval is available. All state tensors must remain unchanged.

Use the inherited lease, atomic writes, per-unit SHA-256 markers, safe checkpoint loading and numerical checks. Resume verifies identities/hashes and reuses complete units. Scientific freezes include input/source manifest and environment. The output tree is separate from all prior experiments.

Engineering failure stops the run. A null perturbation is a recorded scientific condition, not an engineering failure. The only successful scientific terminal state is **COMPLETE_DIAGNOSTIC_NO_SELECTION**.

## Analysis and next decision

Report mechanical and post-hoc cohorts separately, and MSL/SMAP separately. Average seeds within channel for paired summaries. Retain all channels, signs, amplitudes and families, including unfavourable and null cases. Report per-seed dispersion and leave-one-channel-out influence. Bootstrap intervals, if calculated from the saved response tables, are descriptive channel-cluster intervals; they are not significance gates. Synthetic probe instances are not independent samples for p-values.

Interpretation is bounded:

- Consistent command-context differences can justify a later causal context-conditioned reference model, subject to adequate state exposure and independent review.
- Large actual-versus-residual-only differences require changing the future *development simulator*, not selecting a favourable failed EXP009 policy.
- Long alarm tails around isolated nominal errors motivate explicit bounded persistence/degenerate-reference handling in one later detector proposal, not unbounded smoothing searches.
- Indistinguishable nominal and corrupted responses, or inadequate nominal state coverage, justify seeking additional context or labelled development data rather than more threshold tuning.

No numerical detector-promotion criterion is introduced here. Do not use EXP010 to retrospectively relax EXP009 gates. Before any new detector search, write a separate bounded protocol informed by these complete diagnostics and reviewer feedback. Final calibration and the explicitly retrospective NASA TEST evaluation remain separate, unauthorised stages.
