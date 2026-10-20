# EXP012 A100 40GB compatibility amendment 1

Date: 6 October 2026 (Australia/Melbourne). Version: **EXP012_RETROSPECTIVE_EVALUATION_V1_A10040GB_AMENDMENT_1**.

**Build/static/CPU checks pass. Actual 40GB GPU compatibility replay is PENDING; Phase A use is NOT YET AUTHORISED.** This workspace has no CUDA device or GPU execution capability (`nvidia-smi` is unavailable). A CPU check, a historical 40GB parent run, or a synthetic test is not an actual 40GB replay. No COMPATIBILITY_PASS.json has been created for real data here.

This is a separately versioned runtime amendment to the delivered 80GB runner. The original 80GB notebook, ZIP and audit remain byte-identical. Original preparation outputs, selected checkpoints, detector source and all frozen scientific settings are unchanged. No retraining, scaler fit, threshold fit, model/detector selection or NASA label access occurred. NASA TEST arrays were not opened.

## Exact permitted runtime difference

| Runtime field | Original EXP012 preparation | Amended target |
|---|---|---|
| GPU name | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-40GB |
| CUDA-visible memory bytes | 85,094,825,984 | 42,405,855,232 |

The amended target is taken directly from the original frozen GCN-parent runtime record. Every other environment field must equal the completed EXP012 environment, including compute capability8.0, driver580.82.07, Python3.13.15, Torch2.11.0+cu130, CUDA13.0, cuDNN91900, NumPy2.1.3, pandas2.2.3, full dependency/build records, libc, one thread, deterministic settings and disabled AMP/TF32/compile. No driver/software exception or accelerator-family substitution is added. An A100 label alone is insufficient if its exact name/capacity or the remaining stack differs.

## Preserved science and evaluation

`common.py`, `metrics.py` and `evaluate_labels.py` are byte-identical to the 80GB bundle. Removing the documented runtime-gate/provenance substitutions from the amended exporter restores its original source exactly. The exporter changes only the compatibility directory argument, replacement of the GPU identity check with a narrowly scoped policy plus mandatory replay receipt, and recording the receipt hash in run provenance. Prediction, scoring, detector and export calculations are unchanged.

The complete old evaluation contract and every preparation/input checksum are preserved. Checkpoints, frozen graph/features/scalers, context64, batch256 inference, original TEST target start250, q99 thresholds, strict comparison, adaptive detector settings/windowing/startup/timing, metrics, ties, undefined-value handling, event policy and all bootstrap/aggregation rules are unchanged. Original scientific source is imported from the verified completed preparation; it is not patched or copied into a new scientific implementation.

## Local checks actually performed

- Seven synthetic compatibility tests passed: permitted GPU-only differences; rejection of other runtime changes, wrong devices/schema/capacity; exact comparison including signed zero, dtype and shape; missing/changed/incomplete receipt rejection; and nominal-only I/O guard.
- All nine inherited synthetic runner/metric tests passed. Synthetic label fixtures contain no NASA label data.
- CPU saved-reference checks passed over all2,106 nominal split streams,4,212 detector/split records and2,080 adaptive windows. They checked original indices, frozen scaling/inversion, persistence, locked-threshold flags, adaptive EWMA/flags/issue times, window audits and diagnostic counts. These are saved-trace contract checks, not newly generated 40GB model forecasts.
- Exporter source diff is runtime/provenance only. Common metrics and the label evaluator are byte-identical; original80GB artifact hashes still match.

## Mandatory real GPU validation before use

The new notebook contains a separate **VALIDATE_40GB_RUNTIME** stage. It accepts only the new runner ZIP, existing immutable preparation input bundle and completed full-result ZIP. It has no NASA archive or label input. It loads only frozen TRAIN development/calibration arrays and existing selected weights; the guard rejects CSV labels and TEST/external arrays.

On an actual matching A100-SXM4-40GB, the gate must validate all972 selected checkpoint states and all1,053 streams (persistence once), each on both saved nominal validation and calibration splits. Forecasts are generated twice, compared against saved80GB forecasts using exact dtype/shape/bytes, and states verified before/after. Validation MSE must equal the saved value exactly. Inverse forecasts, raw residuals, locked q99 flags, adaptive EWMA, flags, actual issue times, window/startup records and diagnostic burdens must also match exactly. Thresholds are consumed unchanged, never recomputed. Peak CUDA memory is recorded. An out-of-memory error, non-GPU environment mismatch, numerical mismatch or incomplete roster prevents a pass. There is no tolerance, batch reduction or bypass switch.

Only a complete real gate writes **COMPATIBILITY_PASS.json** with1053streams,972states,2106split streams,4212detector records,2080windows and zero mismatches/fits. It binds the actual environment, runner version, policy and all frozen source/reference manifests and per-unit evidence. Completed replay units are committed on Drive and reused under the same identity. A completed receipt is verified without rerunning replay.

The amended Phase A exporter verifies that receipt and its full evidence against its current environment and frozen inputs before any TEST array is opened. It refuses to proceed when the receipt is absent, altered, incomplete or belongs to another runtime/source identity. Successful validation authorises only this amended runtime for the unchanged, explicitly requested label-blind Phase A computation. It does not authorise scientific retuning or claim TEST performance.

## Exact Colab steps

1. Download/open the new **EXP012_A10040GB_AMENDMENT_1_Colab.ipynb** and select A10040GB.
2. Set `VALIDATE_40GB_RUNTIME=True`; leave `EXECUTE_LABEL_BLIND_TEST=False` and `EVALUATE_LOCKED_EXPORTS=False`. Run all.
3. Upload **EXP012_A10040GB_AMENDMENT_1_runner_bundle.zip**, the existing **EXP012_ANOMALY_BENCHMARK_PREPARATION_V1_input_bundle.zip**, and **EXP012_PREPARATION_results.zip** when prompted. Matching existing preparation files on Drive are reused read-only. No EXP005 sidecar, NASA archive or label CSV is needed for validation.
4. Wait for `PASS_40GB_NOMINAL_BITWISE_REPLAY` and preserve/download **EXP012_A10040GB_COMPATIBILITY_EVIDENCE.zip**. If validation stops, preserve the evidence; do not relax the gate or change scientific settings. A stale lock may be released only after the old runtime is terminated.
5. Only after the pass exists, set `VALIDATE_40GB_RUNTIME=False`, `EXECUTE_LABEL_BLIND_TEST=True`, keep label evaluation disabled, and Run all. The original NASA archive is now required. The exporter independently rechecks the receipt. No retraining occurs.

All three execution switches default False. The new Drive folder is separate from the80GB runner. The unchanged Phase B code remains included for later authorised scoring; it is disabled in the delivered notebook and was not run on NASA labels here. Do not mix80GB and amended40GB score exports or reuse an80GB seal in the new folder.

## Authorization boundary

This deliverable is **validation-ready**, not a pre-issued40GB clearance. The actual GPU replay cannot be performed in this workspace. The notebook performs it on your Colab40GB and issues the verifiable receipt only after success; Phase A stays blocked until then. Local evidence is in LOCAL_AMENDMENT_AUDIT.json and CPU_TRACE_CONFORMANCE.json. Original80GB identities and exact source changes are included for review.
