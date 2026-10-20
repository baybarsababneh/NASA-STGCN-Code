# EXP012 final preparation readiness audit

Date: 5 October 2026 (UTC). Decision: **READY to use the separately frozen runner for the one-time locked retrospective NASA TEST comparison.** This audit and runner build do not execute NASA TEST. No NASA TEST arrays or anomaly-label contents were opened here; only archive member names and opaque file checksums were inspected. No models were trained, no completed preparation was rerun, and no frozen scientific settings or original outputs were changed.

This report supersedes the preliminary `EXP012_PREPARATION_READINESS_AUDIT.md`: the exact full-result and execution-provenance archives previously missing are now supplied and audited. No required preparation file remains inaccessible.

## Independent verification

| Evidence | Audit result |
|---|---|
| Full results | ZIP CRC passed; all 7,099 manifest payload hashes verified, plus exact file roster. |
| Original input bundle | All 680 payload hashes verified, plus exact file roster. |
| Freeze chain | Run, model, calibration, benchmark and original package identities agree. |
| Checkpoint states | All 972 selected state hashes verified: 486 frozen GCN/removal, 243 historical selected EXP005 TCN, 243 completed controlled LSTM. Restricted nonexecuting storage decode used locally. |
| Completed units | All 2,106 committed units and per-file hashes verified. Persistence appears once for each of 81 channels. |
| LSTM selection | All 243 saved epoch histories, improvement decisions, patience stops and selected epochs independently checked. 113 fits reached the fixed 35-epoch ceiling. No training extension. |
| Thresholds | All 1,053 frozen q99 thresholds independently recomputed from saved calibration traces, using higher quantile and strict exceedance. |
| Nominal detector outputs | All 2,106 saved validation/calibration streams independently recomputed: flags, EWMA scores, actual issue times, 4,212 diagnostic rows and 2,080 adaptive windows. Targets and persistence forecasts checked against immutable nominal arrays. |
| Execution evidence | Full setup/execution logs, launch snapshot and provisioning hashes correlated; actually executed notebook source matches the audited original. |
| Runtime evidence | Recorded exact GPU replay gates checked. This local audit did not perform new GPU inference. Exporter requires the actual EXP012 A100-SXM4-80GB environment, rather than silently substituting a backend. |
| Contract implementation | Separate label metrics correctly preserve undefined corrected F0.5 when N0=0, even if recall=0. Original preparation source is untouched. |
| Runner validation | Nine synthetic tests passed, including a complete 81-channel/1,053-stream label-scoring fixture, 2,106 rows, all 16 prespecified bootstrap contrasts, duplicate-P-2 policy, resume and corruption rejection. No real TEST or labels used. |

## Scope of scientific readiness

The benchmark is ready for a bounded retrospective comparison of both frozen detector arms and all frozen models. Readiness does not establish detection quality, online deployability or superiority. Historical NASA TEST and prior final-TRAIN exposure remain declared. Adaptive output is batch-based and includes startup averaging and buffered retrospective flags: event overlap is mandatory alongside actual notification timing, with late notifications and original onsets preserved.

High nominal alarm burden, discrete or tied calibration scores, short/no-output streams and the LSTM training ceiling are reportable limitations. No detector winner is selected from nominal diagnostics or future TEST results. The controlled command-aware, one-step/context64 LSTM is not an exact reproduction of the published multi-output LSTM. No literature SOTA claim is supported. Paired channel bootstrap retains all original seeds and acknowledges shared contexts and dependence; persistence is a single reference stream.

Every target from original TEST index 250 remains in the denominator. Startup suppression does not remove targets. Fully unsupported original events, zero nominal support and no-positive AP are explicitly reported. Native channel scores are never pooled for AP.

## Separate runner and unchanged settings

Phase A loads existing selected weights and FIT scalers, verifies their state hashes, produces repeatable frozen forecasts, applies the original unchanged detector code and q99 values, then seals all 1,053 exports before labels are allowed. It makes no training, graph-construction, scaling-fit or threshold-calibration call. Original input and preparation results are read-only sources.

Phase B accepts only the sealed export and the historically pinned label CSV; it has no model or NASA-array input. It preserves first-P-2, inclusive-to-half-open intervals and 104 original events. MSL and SMAP remain separate; counts are pooled per seed before averaging metrics. The primary contrast is GCN minus matched message removal; LSTM, TCN and persistence are secondary. Both adaptive and raw_q99 arms remain intact. All 10,000-draw paired channel bootstrap contrasts use seed 20261005.

The standalone metric correction implements the frozen zero-N0 contract: undefined precision is checked before zero-recall handling. It is a correctness repair in the new evaluation code, not a change in scientific settings. Alarm starts on nominal samples and wholly nominal alarm runs are exported separately; original event endpoints, truncation and late notifications are retained.

## Run instructions

1. Open `EXP012_RETROSPECTIVE_EVALUATION_V1_Colab.ipynb` in Colab. For Phase A select **A100 with 80 GB**. Exact driver/software checks stop safely on a mismatch; do not alter the freeze to get past the check.
2. Upload the new runner bundle, the existing immutable preparation input bundle, `EXP012_PREPARATION_results.zip` and the original NASA `archive.zip` when prompted. Cached preparation files on Drive are reused if their bytes match. **No EXP005 sidecar, retraining or preparation notebook rerun is needed.**
3. Set `EXECUTE_LABEL_BLIND_TEST=True`, keep `EVALUATE_LOCKED_EXPORTS=False`, and Run all. This deliberately performs the one-time TEST score export. Completion produces `SCORE_EXPORT_FREEZE.json` and a label-free score archive on Drive. Committed units survive disconnects; a completed sealed export is verified and reused without inference.
4. After the seal exists, set `EXECUTE_LABEL_BLIND_TEST=False`, `EVALUATE_LOCKED_EXPORTS=True` and Run all. Phase B may run on CPU or in the same A100 session. Only then upload the original `labeled_anomalies.csv`. The notebook checks the export seal before asking for labels.
5. Keep the score archive, label reports and execution provenance. The notebook downloads the compact report archive. If interrupted, rerun under the same identity. Release a stale lock only after terminating the old runtime; never launch concurrent runs.

Both execution toggles are False in the delivered notebook. The conditional request authorised auditing and building this runner; no evaluation was performed during the build.

## Audited input identities

| File | Bytes | SHA256 |
|---|---:|---|
| EXP012_PREPARATION_results.zip | 193,236,085 | `cf8a0887e5424f1e854a9114297130d5a68e5204991a5169bb7ff628e3e8ae45` |
| EXP012_ANOMALY_BENCHMARK_PREPARATION_V1_input_bundle.zip | 62,617,077 | `8e034443158e5862eae83efa695f570294fe4f9a5f89791604e41b1f195e2c84` |
| EXP012_execution_provenance.zip | 135,731 | `6078d268bdf519b35f32151c1eaa2b43ed63528ff92910905ebb319bd17eb9a3` |
| archive.zip (uploaded as 03-archive.zip) | 86,004,011 | `6084d3ee3906381f2c98aa3773b6b2d77c82413503faa78f962196582e873733` |

Pinned label CSV SHA256 from the existing contract: `057ce2d6c8875982bf4e5404aefea14efdcbce413d80826d2b737c95b59b7539`. Its contents have not been opened in this audit/build.

Verification boundary: saved nominal traces and checkpoint-state identity were independently audited on CPU. Full TEST inference and GPU end-to-end operation of the new exporter await the intended Colab execution; synthetic tests do not substitute for that runtime gate.
