# Released-LSTM comparison

Compares frozen project forecasts with Hundman et al.'s author-released LSTM. No training is required. Results are in `outputs/mission_metrics.csv`; use `outputs/sensitivity_verified_seed_metrics.csv` for corrected mission-scoped C-2 sensitivity. C-2 belongs to MSL. The original no-op SMAP sensitivity remains an execution record.

From the repository root, stage an isolated workspace:

```bash
python tools/stage_workspace.py followups --destination /absolute/path/to/empty-workspace
```

First restore the companion artefacts. Staging supplies the historical directory names, NASA archive, labels, score export, and preparation source. The optional `--historical-report /path/to/original.pdf` is only for the exact original report input; it is not distributed. Do not recreate that report from current text.

Recorded dependencies: NumPy 2.3.5, pandas 2.2.3, h5py 3.16.0, SciPy 1.17.0, more-itertools 11.1.0, and threadpoolctl 3.6.0. See `outputs/ENVIRONMENT.json` for the original runtime.

In the staged parent directory, the historical sequence is:

The public copies currently fail the original full-tree package seals. These commands require the exact original sealed inputs for replay; the reviewed ZIP alone is insufficient.

The public copies currently fail the original full-tree package seals. These commands require the exact original sealed inputs for replay; the reviewed ZIP alone is insufficient.

The public copies currently fail the original full-tree package seals. These commands require the exact original sealed inputs for replay; the reviewed ZIP alone is insufficient.

```bash
python lstm_benchmark_audit/verify_sources.py
python lstm_benchmark_audit/analyse.py
python lstm_benchmark_audit/audit_historical_receipts.py
python lstm_benchmark_audit/verify_analysis.py
```

These scripts write outputs. Run them only in the working copy. The native detector has one documented integer-index compatibility fix; model weights and forecasting settings are unchanged.

`PACKAGE_MANIFEST.json` is the original scientific seal. Publication edits and omitted report material mean it does not describe the current public tree. Exact legacy gates require original inputs; the repository-wide public manifest is a separate packaging check. See the root `docs/REPRODUCTION.md` for limits. The source-history JSON records concern upstream Telemanom, not this project's private Git history.
