# Round 2 implementation audit

Only TRAIN fit/validation observations and earlier validation-only tables were used in this round. No TEST arrays, anomaly intervals, anomaly metrics, or calibration observations were loaded by the new workflow. Original research files and uploaded inputs are unchanged; input hashes are recorded. The NASA test set was evaluated in the earlier study, so a later assessment must not be described as a never-before-examined test benchmark.

## What the supplied TCN actually computes

Input is `[batch, node, time]`. Reshaping to `[batch * node, 1, time]` applies the same temporal filters independently to each node; it does not convolve across nodes. Each convolution is padded only on the left, and the final time position predicts the next telemetry value. Inputs at or after the target time are excluded.

The supplied two convolutions have kernel 3 and dilations 1, 2. Their theoretical receptive field is `1 + (3 - 1) * (1 + 2) = 7` observations. A nominal 64-, 128-, or 250-step window therefore still supplies only seven effective observations to this exact encoder. This is a valid short-context model, not a causality defect. Increasing lookback alone is not a longer-memory experiment.

The target forecast is the last observed telemetry value plus a learned residual. A zero-initialized output head gives exact persistence before training. Epoch zero is measured as a diagnostic, but only completed training epochs are eligible for early-stopping selection, matching the original trainer; the restored model is therefore not guaranteed to outperform persistence. Graph self and message transforms are unchanged. Node-specific gains/biases distinguish command indicators and telemetry. The temporal convolution weights are shared across nodes. This is a temporal-convolution graph forecaster; it is not a claim to reproduce a canonical traffic-network STGCN architecture. There are no residual connections between the temporal convolution layers themselves.

## Fixed graph and training boundaries

Column zero is the telemetry target; the other retained columns are binary command-context indicators that vary during the first 60% of TRAIN. They are not separate telemetry sensors. SMAP retains equal command-to-target star weights; MSL retains up to four command-to-target edges selected using fit-only absolute lagged correlations with target changes, at lags 1, 2, 4, 8, 16. Edges are statistical input permissions, not a physical spacecraft network or proof of causation. No graph spans separate channel files, since synchronization/physical topology is unavailable.

The same 60/20/20 chronological split is retained. Fit-only target mean/standard deviation and command filtering are used; constant target standard deviation is replaced by one. Validation can use earlier validation observations as past context: evaluation is rolling one-step forecasting, not an open-loop rollout. The final 20% of TRAIN is reserved and not scored here.

All new configurations use identical fit targets beginning at index 64. Longer lookbacks receive standardized-zero left padding only before the start of the TRAIN array. This prevents lookback comparisons from also changing the fitted target population and retains short channels such as D-12. Some long windows on short streams therefore contain limited real history. The 64-step incumbent uses no added startup padding. This extension is ours, not part of the original experiment.

## Compatibility and efficiency

`inputs/models.py` is the unchanged uploaded implementation. The new module imports its graph builder and baseline classes. For the supplied two-layer architecture, state names `temporal.conv1`/`conv2`, initialization order, graph/head parameters, and strict state-dictionary loading are preserved. Checkpoint wrappers differ: old wrappers must be read using their original field names; new wrappers carry schema `round2_tcn_v1`. Wider/deeper architectures require their own matching config and are not interchangeable with old checkpoints.

For lookback at least the receptive field, the new encoder computes only the suffix that can affect the last output, using valid convolutions. This is mathematically equivalent to the original padded convolutions at the last output and avoids discarded computation. When lookback is shorter than the receptive field, the original layer-by-layer causal-padding semantics are retained. After worker-session expiry, suffix truncation was also extended to shorter-than-RF inputs, keeping only the required hidden-state suffix and padding at each true start boundary. The remaining four RF255 screen fits use this equivalent optimization; timings across this implementation change are not directly comparable. Output and gradient comparisons pass tolerance checks; floating-point differences can still alter a long optimization trajectory slightly.

The uploaded run scripts resume by channel within a CSV and can accidentally reuse stale results after changing settings; pilot/full run names also overlap. New checkpoint names include configuration and seed, and resume asserts an exact configuration match. The 40-versus-80 comparison uses best epochs within the first 40 and all 80 epochs of one continuous optimizer trajectory, not an invalid continuation from an already-restored best-40 model.

## Meaningful verification

Seven executable tests cover strict legacy state loading/output agreement, seven-step effective context, optimized versus padded reference outputs at all receptive fields, no unintended cross-node convolution, persistence initialization, exact target/window alignment and split exclusion, and short-channel inclusion. Results are in `results/audit_tests.txt`.

No edge-learning, dropout or learning-rate sweep was added. Larger receptive fields necessarily increase depth in the chosen dilated-convolution family; results identify useful complete configurations, not the isolated causal effect of context length. Fixed-GCN controls are freshly trained to the same 80-epoch limit and three seeds. Archived MLP/LSTM/learned-edge runs retain their original 40-epoch budgets and are explicitly secondary references. The compact LSTM is not the published Hundman LSTM anomaly pipeline.
