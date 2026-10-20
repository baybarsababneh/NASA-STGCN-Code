# EXP010 audit requirements

1. Verify the new package manifest and retained original EXP008 source/input hashes. Check that only 19 first-80%-TRAIN arrays and 114 authorised existing checkpoints are present. No final calibration or NASA TEST arrays may be added.
2. Reconcile the 114 checkpoint jobs, three seeds, two roles and 26 cases/job. Verify 114 nominal and 2,964 perturbation unit identities and their hashes. Use `src/verify_exp010.py` for the saved output directory.
3. Nominal exact-MSE and repeat-prediction gates must pass for every checkpoint. Verify the recorded governing GPU environment. No tolerance or software change is automatically acceptable.
4. Independently regenerate a subset of perturbations from the declared family/amplitude/sign/onset rule. Commands and pre/post-support telemetry must be unchanged. Check float32 rounding and null stuck cases.
5. Check onset causality, the lookback-64 return-to-baseline bound and unchanged state tensors. Verify observed rather than recursive forecast history.
6. Recompute actual-minus-nominal response and residual-only counterfactuals. Null cases cannot inflate apparent misses or hits. Truncated recovery exposure must remain explicit.
7. Verify persistence appears once per channel/case, and seeds are averaged within channels for interpretation. Separate the mechanical cohort from post-hoc mechanism cases. Do not claim population representativeness for the enriched cases.
8. Inspect unconditional hits, background overlap and incremental hits together. Failed EXP009 scorers are diagnostic instruments, not candidates being reconsidered. No policy ranking or promotion may be inferred from the output summary.
9. Check command context is strictly causal (through t-1), uses only original selected incoming sources, and does not become an undeclared threshold conditioning rule.
10. Add diagnosis and audited EXP010 results as a separate reviewer addendum. Preserve EXP001–EXP009, E01 records, historical AP outputs and the final forecasting architecture freeze. Do not label this a real-anomaly performance benchmark.

Further paired uncertainty analysis should resample channels after seed averaging, separately by mission/cohort. Leave-one-channel-out checks are particularly important in the small mechanical cohort. These intervals are descriptive, not detector-promotion gates.
