# EXP012_ANOMALY_BENCHMARK_PREPARATION_V1

## Scientific question and boundary
Prepare a fair, traceable comparison of the **frozen command-to-telemetry GCN anomaly pipeline** against its matched message-removal control, the already selected command-aware TCN, a controlled LSTM, and persistence. Forecasting remains an intermediate task. This experiment produces no NASA anomaly performance estimates and authorises no TEST inference.

EXP001–EXP011 remain unchanged. EXP008 verified saved weights; EXP009 found no feasible residual detector; EXP010 showed adaptation to sustained telemetry faults; EXP011 did not establish a deployable fusion detector. Those failures are not erased by adopting a published detector for a bounded retrospective comparison. There is no new detector search or synthetic sensitivity optimisation here.

## Model arms

| Arm | Weights | Information | Purpose |
|---|---|---|---|
| Final GCN | EXP007 canonical fixed, three seeds | Frozen graph inputs/support | Principal GCN result: MSL RF7/h32; SMAP RF31/h64 |
| Matched removal | EXP007 final matched ablation, three seeds | Same input tensor; command message suppressed | Contribution of message pathway under this architecture |
| Command-aware TCN | Historical EXP005 selected checkpoints, three seeds | Telemetry plus exactly the incoming commands | Independently tuned, equally informed competitor; not a matched ablation |
| Controlled LSTM | 243 new fits, one fixed configuration | Exactly those telemetry/command features | Publication-informed controlled recurrent baseline |
| Persistence | No fits; one stream/channel | Last telemetry observation | Simple reference, not three independent replications |

Only the LSTM is trained. MSL/SMAP GCN, removal and TCN weights are neither refitted nor reselected. No CPU EXP006 results enter. Learned or dynamically gated graphs are not revived.

The controlled LSTM is two stacked 80-unit recurrent layers, dropout 0.3 after each, linear one-step telemetry output, Adam 1e-3, no weight decay/clipping, batch64, at most35 epochs, patience10, min_delta0.0003, chronological validation, seeds17/29/43, shuffled FIT windows. Select the last eligible improvement, never epoch0. It has no persistence bypass. This is a **one-step PyTorch adaptation**, not an exact reproduction of the original ten-output Keras LSTM or its published scores. No claim of an independently optimised strongest LSTM is permitted.

**Necessary feasibility correction before this version freezes:** lookback64 for the controlled LSTM, fixed across both missions. D-12 has only187 FIT samples under the established split, so lookback250 produces no training windows. Keeping250 would require exclusion, TRAIN repartitioning or a per-channel substitute. None is introduced. The two-layer80/80 design and other publication defaults remain unchanged. Existing GCN/TCN lookback is64; their receptive fields are not their full input lengths. Common future TEST target start remains250.

## Data roles
For original TRAIN length N: FIT=[0,floor(.6N)), validation=[floor(.6N),floor(.8N)), calibration=[floor(.8N),N). FIT alone establishes telemetry mean/SD; reuse the audited original float32 scalers and graph support. Commands keep the existing encoding and are not newly normalised. Prediction at t uses only [t-64,t); no target/future samples enter the forecasting input. Earlier observations may provide causal history for later split targets.

The immutable bundle has81 development-prefix arrays and81 separately stored calibration slices, each with64 earlier context rows. Context is not additional calibration target data. Original TRAIN hashes match EXP008/EXP007 provenance. TEST arrays and anomaly labels are physically absent. Python's audit hook permits only development arrays before the model freeze, then permits calibration slices; all array opens are logged incrementally. Original full TRAIN arrays were loaded in the build workspace only to copy/validate split payloads and verify FIT scalers; no calibration residuals or calibration-derived choices were computed there. In Colab, calibration array loading is blocked before model freeze. Cryptographic hashing of calibration bytes during integrity checks is allowed and is not statistical calibration use.

NASA TRAIN is treated as nominal, not certified fault-free. Its final20% was used in earlier historical calibration; it is **not globally virgin**. Recent detector studies protected it, but EXP012 cannot retroactively restore independence. NASA TEST labels/results were historically seen: eventual evaluation is locked retrospective, never a virgin holdout.

## Detector arms (both retained; no performance selection)

**Primary: HUNDMAN_ADAPT_V1.** Byte-pinned official config/errors sources are recorded in provenance. Independent implementation uses original-scale absolute telemetry residuals; EWMA span105 (=70×30×.05, adjust=True); sliding up-to2100-sample error windows; batch70; z2.5:0.5:11.5 with original unsupervised objective and fallback12; forward/inverted error branches; anomaly buffer100; peak pruning p=.13; original telemetry-scale guards. Uniform initial smoothing/ignore rules apply; no channel C-2 exception. `l_s=250` here is the detector startup rule, not this adaptation's model lookback. See UPSTREAM_ADAPTATIONS.md for necessary numerical/index safeguards.

Adaptive thresholds are estimated from **unlabelled values/scores within each processing window**. This is part of a frozen batch detector, not TRAIN-only fixed-threshold estimation or label-driven TEST optimisation. It is not a per-sample online detector. Initial smoothing uses up to500 stream observations; first decision waits for its window. Buffered flags can point backward: actual issue indices are exported and mandatory for delay/operational interpretation. No claim of instantaneous causal alarms is allowed.

**Reference: raw_q99.** Absolute FIT-standardised telemetry residual; per-model/channel/seed threshold from all final20% calibration targets; q=.99, NumPy `method='higher'`, strict score>threshold. Persistence once/channel. No EWMA, buffers, adjustment or post-processing. Quantile rank, ties, maximum collapse and exceedances are retained. Known calibration weaknesses are not hidden. No q=.995 arm, channel exceptions or post-hoc rescue is introduced.

Run each detector independently on validation and calibration streams from a fresh stream state. These nominal diagnostics expose implementation/operating behaviour; they **do not select thresholds, models, detector arms, missions or channels**, and do not certify false-alarm control or useful anomaly sensitivity. A high alarm burden is a reported limitation, not permission to retune.

## Execution and budgets
1. Verify input/sidecar manifests and exact pinned A100 CUDA environment. No CPU scientific fallback.
2. Run synthetic engineering tests, including exact CUDA epoch-resume test. Load all486 GCN/removal and243 selected TCN checkpoints and validate config/support/scalers/metadata before optimisers.
3. Exact validation-MSE replay of all486 frozen GCN/removal parents on the original backend. Any difference stops; no tolerance relaxation. TCN gets byte/state/metadata verification and repeated GPU predictions; its historical CPU MSE is not required to equal CUDA MSE.
4. Replay243 TCNs, train243 controlled LSTMs, commit all972 model preparation units; commit81 persistence validation streams. No fitting after `MODEL_FREEZE.json`.
5. Open calibration; commit1053 model/channel/seed calibration qualification units and4212 nominal diagnostic rows (2 splits×2 detectors). Lock thresholds and detector rules.
6. Independently recompute hashes, rosters, thresholds, detector flags/window logs/timing, diagnostics and freeze dependencies; export compact summary and full results.

Exact totals: **243 new scientific fits**, **2106 committed units** =972 model preparation+81 persistence validation+1053 calibration qualification. These are different denominators. There is no screening/confirmation stage, promotion criterion, reserve fit budget, detector winner or selection loop. Failed/interrupted uncommitted work is recomputed under the same identity; never counted as an additional unique scientific fit.

## Resume and provenance
Only one writer may hold ACTIVE_RUN.json. Commit each result atomically with a final hash marker. Save LSTM state, optimiser, selected state, history and Torch/loader/CUDA RNG at every completed epoch; write a hashed JSON pointer last. Resume the next epoch; a partly completed epoch is discarded. Orphan writes never qualify as completion. Preserve completed parent/result bytes.

Runtime identity includes input/source/config hashes, sidecar checksum, full package environment, Python/Torch/NumPy/pandas/CUDA/cuDNN, A100 type/memory, driver/build/determinism settings. UUID/session timestamps are session evidence, not a requirement to get the same physical accelerator. Exact software/device-family/run identity changes stop rather than silently mix backends. TF32/AMP/compile are disabled. No tolerances are widened to resume.

## Stop, freeze and interpretation
Engineering/numerical/inventory/leakage failure => safe stop, preserve results, investigate with a versioned amendment if required. Ordinary high nominal alarm burden does not trigger development. Success => `COMPLETE_PREPARED_AWAITING_REVIEW`, both detector arms frozen, **no TEST authorisation**. The end state establishes preparation integrity, not detector feasibility, anomaly improvement, operational safety or SOTA.

Before a separately authorised NASA evaluation: independent researcher audit; resolve serious validity issues; freeze score-export/evaluation software against evaluation_contract.json; byte-verify all selected weights and thresholds; export scores/flags/issues label-blind; hash exports before labels are opened. Use corrected AP tie handling. Preserve historical anomaly outputs/bug disclosures. Do not select an arm after seeing new TEST outcomes.

The evaluation contract specifies corrected event F0.5 primary, Hundman overlap secondary, strict point metrics and grouped-tie AP, actual notification delay, false-alarm burden, mission-separated results and10,000 paired channel bootstrap. It keeps retrospective overlap performance separate from timely detection. The matched ablation cannot establish superiority to every equally informed non-graph model. Report independently tuned competitor and controlled LSTM limitations honestly. Strong external controlled methods, if undertaken, belong to a separate preregistered study; this preparation does not claim a literature leaderboard reproduction.
