# NASA SMAP/MSL anomaly freeze v2

This directory implements the post-EXP005 frozen anomaly-evaluation protocol.

## Scope

- Principal GCN: EXP002.
- Main independently tuned competitor: EXP005.
- Core matched ablation: EXP003.
- EXP004 and EXP001 are supporting internal rows; persistence is deterministic baseline.
- Primary score: absolute standardized target residual.
- Primary calibration: per model/channel/seed empirical 99th percentile (`method="higher"`), strict `score > threshold`.
- Fixed sensitivities: EWMA span 10 and 99.5th percentile.
- Primary TEST region: target indices 250 onward.
- Secondary internal coverage check: index 64 onward for EXP002–EXP005.
- Primary outcome: corrected event-wise F0.5, mission results separately.
- Study status: locked retrospective NASA benchmark because TEST outcomes were historically exposed earlier in development.

## Required project layout

Place `anomaly_freeze_v2/` inside the existing `smap_msl_tcn_gcn_v2/` project, alongside `src/`, `configs/`, `checkpoints/`, `results/`.
The existing V2 loader expects the extracted NASA `archive/` directory and `labeled_anomalies.csv` to remain one directory above the project root, matching the current project convention.

## Order — do not skip

From the project root with `.venv` active:

```bash
python anomaly_freeze_v2/audit_readiness.py
python anomaly_freeze_v2/test_metrics.py
python anomaly_freeze_v2/calibrate.py
python anomaly_freeze_v2/freeze.py
```

At this point stop and inspect/send the readiness, calibration and freeze outputs before any TEST run.

Only after the pre-TEST freeze is accepted:

```bash
python anomaly_freeze_v2/export_test_scores.py
```

This script is label-blind. It opens TEST arrays but not anomaly labels.

Only after score export is complete:

```bash
python anomaly_freeze_v2/evaluate_locked.py
```

`evaluate_locked.py` is the sole anomaly-label reader in this package.

## Important outputs before TEST

- `checkpoint_manifest.csv`
- `readiness_report.json`
- `thresholds.csv`
- `calibration_diagnostics.csv`
- `calibration_report.json`
- `FREEZE_MANIFEST.json`

Any modification to a hashed frozen input after `freeze.py` creates a new protocol version.
