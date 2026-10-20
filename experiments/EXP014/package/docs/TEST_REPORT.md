# Development scientific and engineering validation

**PASS: 17 CPU regression cases, 17.58 seconds, zero failures/errors. No actual A100 execution has been performed.** The authoritative machine-readable receipt and complete test log are in `audit/TEST_REPORT.json` and `audit/CPU_REGRESSION.log`. Python 3.12.14, PyTorch 2.6.0+cpu and the pinned numerical libraries were used.

## Scientific tests actually executed

- Exact detector/evaluator byte hashes; all 486 historical EXP012 GCN/LSTM channel count rows and the four frozen mission-level corrected event results.
- Six independent saved EXP012 detector references spanning both models/missions and short/long channels: EWMA, flags, issue indices and adaptive windows reproduced bitwise, without historical model inference.
- Target residual construction and both inverse scaling stages, including the intervening FP32 cast; causal five-sample forecast alignment and future invariance.
- Native GDN four-point history/EWMA before cropping, fresh Hundman startup at 250, pruning, ties, buffers, short stream and tail/remainder support.
- Corrected event counting, strict points without adjustment, actual late notification timing, grouped-seed aggregation, paired bootstrap reproducibility and undefined point-metric handling.
- Native GDN CPU batches `[32,55,5]` and `[32,25,5]`; exact checkpoint/state/output/RNG restoration and corrupt tagged-checkpoint rejection.
- Existing native GDN PyG compatibility edits checked against an independent dense attention oracle for outputs and input/embedding/parameter gradients. The source and original tolerances are unchanged. These are CPU measurements:

| Nodes | Forward max abs | Input gradient max abs | Embedding gradient max abs | Parameter gradient max abs |
|---:|---:|---:|---:|---:|
|55|4.47034836e-08|2.38418579e-07|5.96046448e-08|9.53674316e-07|
|25|4.47034836e-08|2.38418579e-07|1.1920929e-07|7.15255737e-07|

The original oracle gates are atol 3e-6/rtol 1e-5 for forward/input gradients and atol 5e-6/rtol 3e-5 for embedding/parameter gradients. No optimiser step is performed by EXP014 and no numerical model correction is introduced. The E03 SMAP fusion tolerance amendment is irrelevant to this unchanged GDN inference stage and is not broadened.

## Engineering and complete execution tests actually executed

A four-channel miniature fixture uses **untrained synthetic native GDN weights**, full 55/25-dimensional inputs, FP64 raw telemetry and explicitly synthetic historical tables. It covers source-ZIP extraction/verification, frozen-input/model checks, full native shapes, prediction/scoring/replay, detector export, sealing before annotation parsing, counts/uncertainty/tables/figures, atomic reports and a verified final archive.

After its first committed channel, an injected interruption triggers a verified backup. Local work/cache are deleted to simulate VM loss. The run restores from the durable copy, skips its verified completed channel, finishes, and then skips every unit on another complete resume. All numerical arrays and final metric values match a separate uninterrupted run exactly. Every final archive member is hashed, and the report-only exporter is tested.

Independent engineering cases reject incomplete/stale/corrupt units, duplicate/traversal archive paths, annotation opens/parsing before the gate, changed original inputs despite a valid cache, corrupt checkpoint payloads and production use of CPU receipts. A corrupted newest backup falls back to the previous verified generation; a changed runtime is rejected. These tests use a local filesystem as Drive's interface. Actual mounted-Drive latency/reliability remains unmeasured.

The notebook's five code cells are compiled before release; its paired ZIP and manifest hashes are pinned. The isolated-venv creation command was executed with the current Python interpreter. The complete command-line fixture workflow also passed from a source ZIP through verified final export. Its PNG figure was visually checked; near-one values are labelled inside the bars so titles remain clear. Real CUDA-wheel provisioning and Google authentication require Colab and were not executed locally.

## Requirements that remain unexecuted

Both original GDN checkpoint files and all 81 native score streams are missing from the provided report archive. The notebook must recover and hash-verify those 252 constituents from the completed EXP013 archive, identify a real A100 40 GB, check CUDA 12.6/CuDNN 9.5.1/PyTorch2.6, measure native full-batch memory/time, restore exactly, and reproduce all 81 native score/flag/issue arrays **bitwise** before publishing any new common-detector comparison.

GPU working-set estimates are conservative engineering bounds with 1 GiB slack, not benchmarks. Actual peak allocation/reservation and process RSS are recorded during the real gates/inference. There is no A100 feasibility PASS, new GDN NASA result, measured GPU speedup or completion ETA in this delivery. Visible ETA begins only after actual completed channel timings.
