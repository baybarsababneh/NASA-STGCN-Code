# EXP016 — Run in Google Colab

1. Open **EXP016_Run_All.ipynb** in Colab.
2. Select **Runtime → Change runtime type → A100 GPU**, with **Runtime Version
   2026.07** recommended (40 GB is sufficient
   for the intended design; the actual full-batch GPU check must pass).
3. Choose **Runtime → Run all**. Authorise Google Drive when prompted, then
   upload **EXP016_Package.zip** at the upload prompt. Upload the ZIP intact.
4. Leave the notebook running. No code editing, separate dataset upload,
   Kaggle account or previous EXP012/013 checkpoint is required.

The package includes the supplied NASA archive, released LSTM checkpoints and
predictions, labels, scripts, configuration, dependency pins and tests.
**No LSTM is trained.** Internet access is needed only for Python dependencies.
The worker uses isolated Python 3.10–3.12 / PyTorch2.5.1+cu121; the notebook
kernel is not replaced and does not require a restart.

## What runs

Package validation → isolated dependency setup → automated correctness tests →
small synthetic end-to-end fixture → actual-GPU equivalence/recovery checks →
released-checkpoint verification → measured runtime estimate → 246 ST-GCN fits
(82 channels × 3 seeds) → frozen forecasts → shared detection → scoring/export.

The notebook stops the scientific session safely after about **5.5 hours** if
needed. This limit does not shorten training or reduce coverage. Initial setup
and preflight take additional time. The runtime estimate is measured on your
actual GPU; six-hour completion is a target, not a guarantee.

## Resume

If `STATUS.json` says `PAUSED_RESUMABLE`, or Colab disconnects, reconnect to an
**A100 with the same pinned software** and run all cells again. The package and
checkpoint mirror are recovered automatically from
`MyDrive/EXP016_V1/`. Already committed fits and predictions are verified and
reused. Up to approximately five minutes of unsynchronised work can be repeated
after abrupt VM loss. Do not run two sessions against this folder simultaneously.
If Colab says a cell is still running, reconnect to that runtime first.

The last cell displays status and offers **EXP016_run_results.zip** for download.
A durable copy is also written to `MyDrive/EXP016_V1/`. A partial archive is
labelled by its internal status and is not a completed scientific result.

## Read results

- `FINAL_AUDIT.json` / `STATUS.json`: completion, coverage and failures.
- `RESULTS.md`, `mission_mean_metrics.csv`, `mission_seed_metrics.csv`:
  detection comparison, every seed and both annotation/detector sensitivities.
- `channel_metrics.csv`, `event_notifications.csv`, `paired_bootstrap.json`:
  per-channel evidence, alarm timing and descriptive uncertainty.
- `fits/`, `predictions/`, `scores/`: states, optimizer/RNG recovery,
  ten-output GCN predictions, masks, thresholds and detector receipts.
- `RUNTIME_ESTIMATE.json`, `GPU_GATE.json`, `execution.log`, seals/manifests:
  runtime, correctness and provenance.

See **PROTOCOL.md** for scientific decisions. Primary detection removes the
historical C-2 label-informed exception; the native sensitivity retains it.
Chronological validation differs from the author's overlapping random split.
TEST-extrema scaling is inherited from the release and cannot be undone.
The comparison is retrospective and cannot isolate architecture perfectly.
T-10 is fitted/predicted but not scored because it has no supplied annotation.

The package has executed CPU correctness and integration checks. The included
validation report identifies exactly what was tested. **No full A100 experiment
has been run during package preparation.** The Colab notebook performs GPU gates
before fitting. GPU availability, driver compatibility, network access, browser
uploads and Drive authorisation remain environment-dependent.

The recommended Colab 2026.07 runtime uses Python 3.12.13: [official runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html). Dependency installation uses the [official PyTorch 2.5.1 wheel index](https://pytorch.org/get-started/previous-versions/).
