# NASA SMAP/MSL anomaly detection

Code, configurations and saved results for experiments **EXP001–EXP015**. 

This project compares command-aware graph models with LSTM and other baselines for spacecraft telemetry anomaly detection. The graph models command-to-telemetry context, **not** a verified network of synchronised sensors.

## Find the code

- [`experiments/`](experiments/) — 15 numbered studies. Open an experiment's `README.md` for its entry points.
- [`followups/`](followups/) — released-LSTM and detector comparison code.
- [`results/`](results/) — links to main result tables.
- [`history/`](history/) — earlier source, controls and saved diagnostic outputs.
- [`tools/`](tools/) — verify files, restore large artifacts, and stage a workspace.
- [`EXPERIMENT_INVENTORY.md`](EXPERIMENT_INVENTORY.md) — one-line guide to every experiment.

## Quick start

Requires Python 3.11+ for the archive tools:

```bash
python tools/verify_archive.py
```

Large weights, telemetry arrays and some run records are **not in this Git repository**. Five separately held artifact ZIPs can be restored, where available:

```bash
python tools/restore_artifacts.py /path/to/NASA_SMAP_MSL_Artifacts_*.zip
python tools/verify_archive.py --artifacts
```

To stage an isolated experiment, run:

```bash
python tools/stage_workspace.py EXP006 --destination /path/to/empty-workspace
```

Read [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md) before launching a model. Historical runners may require their recorded environment and editing machine-specific input paths.

## Main results

The saved EXP012 event F0.5 scores for GCN / **project-trained LSTM** are **0.7694 / 0.7072 (MSL)** and **0.7156 / 0.8107 (SMAP)**. The author-released LSTM comparison uses a **different sample support**; see [`results/README.md`](results/README.md) before comparing scores.

## Limits

- These are preserved research runs, **not** newly executed reproducibility tests.
- EXP013 full trained-state backup is missing; EXP014–015 full replay depends on parts of it. Some EXP011 fit files and early CPU dependency pins are missing.
- The five artifact ZIPs are **not cleared for general redistribution**. Source/licence information: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and [`docs/DATA.md`](docs/DATA.md).
- Historical paths and some explanatory files were edited/omitted for a code-first public release. Frozen experiment hashes refer to the **original** archives, not edited release copies. See [`provenance/PUBLICATION_EDITS.csv`](provenance/PUBLICATION_EDITS.csv).
- No NASA affiliation or endorsement is implied.

Release review and safe GitHub update instructions: [`docs/PUBLIC_RELEASE_REVIEW.md`](docs/PUBLIC_RELEASE_REVIEW.md), [`docs/GITHUB_UPDATE.md`](docs/GITHUB_UPDATE.md). This is a reviewed archival candidate, not a verified end-to-end rerun.
