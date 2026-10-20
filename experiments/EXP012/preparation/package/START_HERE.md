# EXP012 — start here

1. On your Mac, package the **existing** selected EXP005 weights once:
   ```bash
   python3 EXP012_transfer_selected_EXP005.py --project-root "/path/to/NASA Research Project"
   ```
   If the directory differs, use `--checkpoint-root "/path/to/existing/confirm"`. This uses only the Python standard library and reads243 explicit `.pt` paths. It opens no telemetry, TEST arrays or labels. Missing weights => no archive; do not retrain to fill it.
2. Open `EXP012_ANOMALY_BENCHMARK_PREPARATION_V1_Run_All.ipynb` in Colab. Select an **A100** GPU. This frozen environment requires A100; there is no CPU/T4 fallback.
3. Press **Run all**. Grant Drive access and, when prompted once, upload `EXP012_ANOMALY_BENCHMARK_PREPARATION_V1_input_bundle.zip` plus `EXP005_SELECTED_WEIGHTS_FOR_EXP012.zip`. These are cached on Drive. No original NASA archive is needed.
4. Leave it running. It prepares the isolated pinned environment, verifies parents, trains243 controlled LSTMs, freezes weights, calibrates both prespecified detectors and audits/exports results. There is **no NASA TEST runner**.

Results/checkpoints/epoch progress persist at `MyDrive/EXP012_ANOMALY_BENCHMARK_PREPARATION_V1/results`. A disconnect cannot guarantee unattended completion; reconnect with A100 and Run all to continue. Scientific identity/environment changes fail safely. A hard runtime kill may leave ACTIVE_RUN.json: terminate the old runtime, then set the notebook's RELEASE_STALE_LOCK=True once. Never release a live lock or launch concurrent copies. Completed units are verified and reused, not overwritten. A partly finished epoch is retried from its last committed epoch.

Final files: `results/exports/EXP012_PREPARATION_summary.zip` and `EXP012_PREPARATION_results.zip`, each with SHA256, plus `EXP012_execution_provenance.zip` containing provisioning/execution logs and best-effort executed-notebook capture. Keep this input bundle: full result archives deliberately reference original GCN archives in it. Save/download the **actually executed notebook** after completion for provenance. The runner records manual `virtualenv` provisioning, host/scientific Python, package versions, drivers and session history.

The only selected LSTM setting is fixed80/80 units, dropout.3, Adam1e-3, context64,35epochs, chronological validation and published early-stopping defaults. Context64 is a feasibility correction: D-12 has only187FIT samples. No channel exclusions or parameter search.

Success means **COMPLETE_PREPARED_AWAITING_REVIEW**, not a detector winner or permission to evaluate NASA TEST. Send the compact summary to the reviewer; supply full results and input bundle for reproduction. Keep all historical experiments unchanged.

Current build checks were run on synthetic fixtures and existing checkpoint bytes. This workspace has no CUDA accelerator; production CUDA numerical/restart gates run in Colab. Historical EXP005 checkpoint payloads were unavailable here; the transfer sidecar supplies them and is fully checked before any scientific fitting.
