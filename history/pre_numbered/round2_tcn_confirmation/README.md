# Round 2: validation-only TCN-GCN confirmation

This folder is a new extension of the original SMAP/MSL research package. It does not replace original code, results, or uploaded files. No final anomaly evaluation is included. See `Round2_Summary.md` for conclusions, `Audit.md` for implementation details, and `configs/locked_tcn.json` for the final validation-selected configuration.

## Layout

- `inputs/`: immutable uploaded code/CSV snapshots, archived validation-only references and provenance hashes.
- `src/round2_models.py`: compatible, efficient TCN extension importing the uploaded models.
- `src/round2_train.py`: chronological fit/validation training, fixed graph and config-aware checkpointing.
- `src/plan_study.py`, `select_stages.py`: screening and promotion rules, all based on validation forecasts.
- `src/run_jobs.py`: resumable multi-process execution, one CPU thread per worker.
- `src/test_round2.py`: seven correctness/leakage/compatibility checks.
- `src/analyze.py`: rebuilds selection, mission/channel/family/seed/cost tables and plots.
- `src/calibrate_locked_later.py`: unexecuted next local experiment; TRAIN-only calibration after locking.
- `configs/`: exact study protocol, candidate and stage job lists, recorded decisions and final lock.
- `checkpoints/`: one `.pt` model, `.json` learning curve/metadata, and validation-only `.npz` per completed run.
- `results/`: CSV tables, audit test output, environment and run logs.
- `figures/`: exportable PNG figures.

## Reproduction

Use Python 3.11+ with NumPy, pandas, PyTorch and matplotlib. Exact executed versions are in `results/environment.json`; the executed CPU PyTorch version is 2.14.0+cpu. CPU arithmetic and backend differences can change near-tied early-stopping paths. No CUDA is required.

Place this folder beside `smap_msl_research/`, preserving its original `data/data/data/train/` directory. Alternatively set `SMAP_MSL_ROOT` to the original research root. This package deliberately excludes the large original dataset archive. It needs only TRAIN arrays for the executed study. `inputs/train_prefix_manifest.json` fingerprints every fit/validation prefix.

From this folder:

```bash
python src/test_round2.py
python src/run_jobs.py screen_executed_jobs.json --workers 3 --output screen_runs.csv
python src/select_stages.py screen
python src/run_jobs.py confirm_jobs.json --workers 6
python src/select_stages.py confirm
python src/run_jobs.py robustness_jobs.json --workers 6
python src/analyze.py
```

The recorded screen completed 81 of 84 planned fits. The three remaining SMAP RF255 fits were interrupted after an exact score bound proved that even maximally favorable missing scores could not win. `screen_pruning_note.json` records the proof. This computational pruning was added during execution, without changing the selection criterion. The default command reproduces the executed subset; the full planned `screen_jobs.json` is also retained. Existing compatible checkpoints are reused. For a genuinely fresh rerun, copy the folder to a separate directory and remove **only that copy's** `checkpoints/` and generated `results/*_runs.csv` before executing. Preserve input snapshots and recorded protocol. No optimizer state is retained; increasing a ceiling requires a new named configuration and retraining from the declared seed, not resuming the restored best epoch.

The incumbent seed-17 and matched fixed-GCN controls were computed in auxiliary queues while the expensive long-context screen completed; main-stage queues then reused those exact checkpoints. `all_runs.csv` deduplicates by configuration/channel/seed, so these are not counted twice. The always-retained SMAP incumbent also received its seed repeats in an auxiliary queue. MSL confirmation was run as an auxiliary queue once all 42 MSL screening fits finished, while the last SMAP screen fits continued. It applies the same full/nonscreen promotion rule; file locks prevent overlapping queues from writing or training the same checkpoint concurrently. Auxiliary queue files can be rerun but are unnecessary for reproduction through the commands above.

## Exact next local experiment (not executed here)

After accepting the configuration lock, run:

```bash
python src/calibrate_locked_later.py --output local_calibration
```

This loads each locked checkpoint at seeds 17, 29 and 43, evaluates only the reserved last 20% of TRAIN, and records per-channel thresholds under the inherited fixed policy (absolute error, EWMA span 10 with `adjust=True`, startup burn-in `min(10, n//4)`, empirical 0.995 quantile, and threshold floor `max(1e-8, fit_scale * 1e-6)`). It opens no TEST arrays or anomaly labels. It does not select a seed or refit models. It fails if the output directory already exists to protect earlier experiments. Review calibration drift/threshold stability without changing settings in response to NASA test outcomes. A later anomaly-label evaluation requires a separate, locked evaluation step; no such step is run or included here.

Calibration script syntax was checked, but execution is intentionally deferred to preserve the tuning-stage boundary. The quantitative conclusion here concerns one-step forecasting, not anomaly precision, recall or F-scores.
