# EXP015 scientific decision — frozen before new outcome evaluation

Version: EXP015_EXP013_GCN_HUNDMAN_SUPPLEMENT_V1. Decision date: 11 October 2026.

## Primary signal and alignment

Execute only the command-reference residual with the byte-preserved EXP012 `HUNDMAN_ADAPT_V1` detector. This decision follows the actual frozen `CommandReferenceGCN.forward`, `data.Windows`, `scoring.scores` and `pipeline.export_test` sources, not a new NASA outcome. No neural fitting, checkpoint edits, detector search, additional synthetic training or new calibration is performed.

For target index t, the original model receives 65 rows t−64 through t, with current command slots zeroed. Its command encoder reads only rows t−64 through t−1. The reference head depends on that encoded command context and the entity embedding. Current telemetry and its past observations enter the discriminator, not the reference. Retain the exact GEMM dilated-convolution implementation, top-four command support plus null, active-command mask, all trained weights, FP32 and inference batch 256 boundaries beginning at t=64.

The graph and fusion arms use their own selected trained parameters. The separately fitted removal arm zeroes the message before the bias-free message linear layer. Its reference is therefore an entity-conditioned constant. That is a valid residual baseline, not a newly constructed forecast model. Do not remove messages from the trained graph checkpoint as a substitute for the separately trained removal control.

For entity c the frozen preprocessing is:

`z = float32(original TEST array)`

`z[:,0] = float32((float64(original[:,0]) − mean_c) / scale_c)`

Commands retain their released encodings and FP32 cast. `scale_c` is the original FIT-only standard deviation with its declared 0.001 floor. Graph/removal/fusion checkpoints have `scaler=None`; the pooled GDN scaler is inapplicable. A checkpoint claiming otherwise fails, rather than inventing another inverse.

For the model reference r[t], calculate:

`reference_released[t] = float64(r[t]) * scale_c + mean_c`

`error[t] = abs(float64(original_TEST[t,0]) − reference_released[t])`

Subtract the original released target, not a round-tripped FP32 target. “Original units” means released NASA telemetry units. The supplied data inherit TEST-extrema scaling and are not raw physical telemetry. This matters because the detector's absolute safeguards are not invariant to arbitrary rescaling.

The frozen standard EXP013 score export contains only target indices, signed discriminator logits (`native_score`), the old EWMA score, flags and issue indices. Neither the reference nor an invertible residual is retained. The logits cannot reconstruct the reference. Therefore the exact primary signal requires inference from the 18 selected checkpoints. Each forward returns the reference and native logit together; no separate duplicate model pass is needed for the scientific channel. Full-shape engineering checks additionally repeat one original batch per fit.

Retain native inference from t=64 and old EWMA history to reproduce the complete saved t≥250 native output bitwise. Then crop references at t=250 and start a fresh Hundman detector, exactly matching EXP012/EXP014's common domain. Saved new EXP015 reference/residual arrays permit verification and subsequent analysis without rerunning models.

## Secondary discriminator hybrid: excluded

The trained discriminator returns a signed logit, with higher values representing the synthetic-anomaly class. It is not a forecast or a nonnegative telemetry residual. Its classification objective uses synthetic clean/injected examples, not a validated spacecraft probability-calibration experiment.

The exact detector uses `abs(target−prediction)`, the observed telemetry's standard deviation and 95th−5th percentile range, an absolute maximum-error guard of 0.05, relative guards and inverted-error thresholding. Passing signed logits as a residual would lose the intended anomaly orientation. Shifted logits, softplus, sigmoid and CAL-standardised scores impose different offsets/scales, and consequently different absolute-guard behaviour. A sigmoid fixes a numeric range but does not supply an observed telemetry signal or establish calibrated real-anomaly probabilities. Giving it the actual telemetry as the guard target mixes incompatible units; giving it a fabricated target changes the safeguard's meaning.

A separately specified hybrid could be studied as a different detector with its own scale/guard/calibration justification. That is not uniquely determined by the frozen sources and would enlarge this bounded supplement. We therefore exclude it before new scoring. This is a compatibility/scope decision, not a finding that dynamic thresholding of classifier scores is universally meaningless. No transformed-logit result will be selected after seeing labels.

## Detector, controls and evaluation

The frozen detector SHA-256 is `5b238ae7226a60e092b4d0a0954ca276733f3f9a27c0766d98282512dc692714`; evaluator SHA-256 is `d8de4c09a197a9647109369ac77082b0588c8b2ad2f1b940bd479bc47cdbceef`. Both source files are copied byte for byte from EXP014, whose detector is the frozen EXP012 implementation. Settings remain batch 70, window 30, smoothing percentage 0.05, error buffer 100, pruning p=0.13, l_s=250. Startup averaging, absolute/relative guards, threshold grid, inverse-error evidence, pruning, short-stream status, final remainder and actual batch issue times remain unchanged.

Evaluate graph, separately fitted removal and direct fusion at seeds 17, 29 and 43. Each arm uses the same residual construction, detector, causal support and evaluator. Do not tune any arm differently. The models remain those selected by the original clean-reference Huber plus synthetic-classification fitting and synthetic STOP objective. The reference head was not trained for this new detector.

Use the original 81-channel roster (27 MSL, 54 SMAP), 104 original inclusive events converted to half-open intervals, first P-2 row and t≥250 common support. Do not add T-10. Score/commit/seal all 729 new streams before parsing annotations. Historical reports and prior TEST exposure already exist; this sequencing is not physical label isolation or independent confirmation.

Report corrected event F0.5, event precision/recall, corrected precision, strict point precision/recall/F1, nominal occupancy, nominal starts per 1,000 nominal samples, exact count denominators, actual notification times, late issues, misses and unsupported events. No point adjustment or event-evaluator change. Pool channel counts within each seed, then average seed ratios. Keep EXP012 GCN/LSTM, EXP013 original three arms and EXP014 GDN/Hundman results separate and unchanged. GDN has one seed; never replicate it into three seed observations.

Paired descriptive uncertainty follows EXP012/EXP014: 10,000 channel draws, seed 20261005, 95% percentile intervals, all fitted seeds grouped. Prespecified contrasts are graph-minus-removal, graph-minus-fusion, each new arm minus its own original pipeline, and graph minus EXP012 GCN, controlled LSTM and EXP014 GDN, for each mission. Include event F0.5 and auxiliary point F1/occupancy/start differences, undefined draws and conditional/dependence limitations. These are exploratory intervals, not confirmatory tests.

## Interpretation and stopping rule

This evaluates a joint change in anomaly-evidence representation and detector, so an improvement cannot isolate thresholding from the representation or prove that scoring was causally the primary failure. Graph value is assessed against both controls under the compatible common detector. If removal improves equally/more, or matches/exceeds graph, state that explicitly. Poor results mean this exact substitution did not rescue performance; they do not identify a mechanism.

A command-only reference is not equivalent to GDN/EXP012 telemetry-history forecasting. Cross-experiment rows are complete pipeline context, not a perfectly controlled architecture-only ablation. The within-EXP013 controls are the fairest frozen matched comparison available. Shared mission fits, entity/context dependence, original checkpoint selection, three versus one seeds, inherited preprocessing and historical TEST exposure constrain interpretation. The old auxiliary synthetic seed-export defect is retained as historical evidence and is not used to identify NASA model seeds here.

Stop after 18 frozen selected-model inference jobs, one exact residual detector, all native replay checks, a complete new score seal, the unchanged evaluator, report export and integrity audit. Positive, negative and mixed outcomes are all final evidence. Do not edit the Overleaf report during this task.
