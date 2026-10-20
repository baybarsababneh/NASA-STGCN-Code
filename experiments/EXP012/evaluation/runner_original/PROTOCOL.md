# EXP012 retrospective evaluation runner V1

This is the separate post-preparation runner. No frozen scientific source or preparation output is edited. The complete unchanged evaluation contract and immutable input identities are in EVALUATION_BINDING.json. The preparation contract's historical PREPARATION_ONLY status remains unchanged; the new runner's audit decision is separate. Delivery authorises no automatic execution; both notebook toggles default False.

## Input files

| Phase | File | Purpose |
|---|---|---|
| A and B | EXP012_RETROSPECTIVE_EVALUATION_V1_runner_bundle.zip | New source, binding, audit, tests |
| A | EXP012_ANOMALY_BENCHMARK_PREPARATION_V1_input_bundle.zip | Original frozen 486 GCN/removal archives, scalers and protocol |
| A | EXP012_PREPARATION_results.zip | Completed 243 LSTM weights, 243 historical TCN weights, all thresholds/source/freeze evidence |
| A | archive.zip | Original NASA archive, selected 81 TEST arrays only |
| B only | labeled_anomalies.csv | Original pinned labels, opened only after export seal |

Existing files are not re-exported or rebuilt. The original EXP005 sidecar is not required: its transferred weights are already in the completed full results. NASA archive copies must be byte-identical to the audited original; file names may differ. The roster excludes unlabelled T-10.

## Fixed execution boundaries

Phase A requires the actual EXP012 A100-SXM4-80GB/Python3.13.15/Torch2.11.0+cu130/NumPy2.1.3/pandas2.2.3/full resolved environment, with deterministic CUDA, one thread and no AMP/TF32/compile. It reads original TEST prefix for context; forecasting targets start at250 with context64. It never concatenates TRAIN to TEST. No new fit is performed. Every forecast is repeated exactly and selected states checked before and after. q99 is raw fit-standardized absolute residual; adaptive scoring uses the original inverse-transformed residual and unchanged frozen detector. Both arms and all targets are preserved.

Phase B checks the runner, all score bytes, full roster, channel supports and export seal before opening labels. The label hash must match the old contract. CPU is sufficient for Phase B (Python3.13.15, NumPy2.1.3). Corrected event scoring, strict point metrics, grouped-tie AP and actual-notification diagnostics are implemented as described in the audit. Both mission/seed-level records and seed means are provided. The primary metric is corrected event F0.5 and primary contrast GCN minus matched message removal under each detector. Sixteen fixed contrasts (two missions, two detectors, four controls), including10,000 channel-paired bootstrap draws each, are emitted; no detector or model is selected from the results.

## Resume and portability

Drive score and report directories are separate from preparation. Committed score streams are hash-verified and reused. A completed seal is never overwritten or recomputed. Labels are opened only for incomplete same-identity report scoring, after score verification. A stale lock is released only after the old runtime is terminated. Keep one writer. The archived score export excludes raw TEST cache but contains every sealed score/flag/issue export plus input/run freezes, so Phase B requires no GPU, models or NASA arrays. EVALUATION_RUN_FREEZE records the archive hash and full environment; TEST_INPUT_FREEZE records all selected arrays.

CLI after verified extraction (A100 environment for Phase A):

```bash
python src/export_scores.py --package /path/original-package --results /path/completed-results --nasa-archive /path/archive.zip --output /path/separate-scores --execute-test
python src/evaluate_labels.py --scores /path/separate-scores --labels /path/labeled_anomalies.csv --output /path/separate-reports --evaluate-locked-exports
```

Synthetic-only validation, on CPU:

```bash
python tests/test_runner.py
```

The tests generate their own disposable synthetic inputs, including a complete label-scoring fixture, and never access NASA TEST or labels. Runner source, binding, report and test receipts are covered by RUNNER_MANIFEST.json. Undefined final metrics are JSON null or blank CSV with denominators. Delay medians are conditional on an attributable notification; misses and late notifications are separate and must accompany them. No point adjustment, retuning, training extension or literature leaderboard claim is added.
