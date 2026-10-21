# EXP016 pre-delivery validation

**Engineering validation passed on CPU. No complete scientific GPU experiment
has been run. No new LSTM was trained.** Scientific ST-GCN results remain to be
produced in Colab.

## Executed checks

- **28 automated tests passed** in an isolated environment using Python 3.12.14,
  PyTorch 2.5.1+cpu, NumPy 1.26.4, pandas 2.2.3, h5py 3.12.1,
  more-itertools 10.5.0 and pytest 8.3.4.
- Both 25- and 55-feature model forward passes and all input/parameter gradients
  match the full causal-convolution reference in float64. Every one of the 250
  history positions and every input feature can affect the forecast.
- Actual full-size FP32 optimizer steps executed on real TRAIN windows for
  G-6 (25 features) and P-10 (55 features): three warm-ups and twelve measured
  batch 64 steps each, validation batch 64 and inference batch 256. These are
  discarded engineering fits, not scientific training or results.
- Chronological target alignment, exclusive author window endpoint, nine-window
  purge and shortest-channel feasibility checked. All 82 channels are feasible.
  D-12 has 32 fit windows and 11 validation windows, with no target overlap.
- Exact CPU mid-epoch interruption/resumption reproduces losses, final parameters
  and Adam state bit-for-bit. Corrupt-current-checkpoint fallback, both-generation
  corruption rejection, changed identity rejection, payload tampering rejection,
  mirror restoration and ZIP traversal rejection tested.
- Synthetic detector tests cover short/long streams, constant residuals, native
  loop parity, uniform C-2 treatment and availability timestamps. Inclusive
  endpoints, one alarm bridging two labelled events, strict point counts,
  AP ties/undefined support and refusal to bootstrap a reduced native roster
  are checked independently.
- A complete two-channel synthetic pipeline executes fitting, checkpoint commits,
  forecast export, both detectors, label gate, per-channel/mission metrics,
  bootstrap, manifest generation, final ZIP and mirror copy.
- **Every one of the seven delivered notebook code cells executes unchanged**
  in the explicit CPU-fixture mode, in a shared Python namespace. This exercises
  upload-path substitution, ZIP checking/extraction, dependency bootstrap,
  subprocess paths, tests, synthetic integration, benchmark, mirror and export.
  Normal notebook mode remains the default and requires CUDA.
- A clean isolated environment was actually installed using `bootstrap.py` with
  the CPU wheel variant. The worker is independent of notebook-kernel packages.
  The CUDA wheel installation and browser/Drive authorisation cannot be tested
  on this host.

## Independent author-reference verification

All 82 saved HDF5 models have two 80-unit LSTMs and a 10-output head. NumPy
inference at first/middle/last positions gives **246 passing forecasts**, with
maximum absolute discrepancy **8.35e-7** under predeclared atol 1e-5/rtol 1e-4.
This is inference-only verification, not training or complete prediction replay.

Recomputed cached smoothed errors match for all 81 labelled channels. T-10's
cache differs by 0.02599431; its forecasts pass model replay. It is unlabelled,
retained for training/prediction and excluded from labelled scores. Detection
uses recomputed residuals rather than trusting that cache.

The pinned detector reproduces **all 81 labelled channels' archived native alarm
intervals exactly**. This establishes source/interval parity, not reproduction
of the paper's printed scores. A full native parity pass took 69.23 seconds on
the preparation CPU. Source receipts are included under `evidence/`.

## Workload and runtime boundary

There are 139,577 fitting windows and 35,111 validation windows across 82 channels
per epoch. Full-roster batch counts are 2,222 fitting and 600 validation per seed
epoch. Three seeds and 35 epochs imply at most 233,310 optimizer steps, plus
validation, inference, detection and output work. No data subsampling is used.

Measured CPU batch 64 training medians were about 0.81 seconds (25 features) and
1.99 seconds (55 features). The corresponding full 35-epoch training projection
was 80.3 hours, before a 1.5× margin. This is a CPU extrapolation, **not an A100
estimate**. It demonstrates that CPU is unsuitable for the requested run.
No measured A100 speed or six-hour completion claim is available here.

The notebook measures actual GPU full-batch throughput before training and
exports 11/20/35-epoch projections. Reliable continuation replaces any temptation
to shorten history, epochs, channels or batches to meet a time limit. The 5.5-hour
scientific-session budget is not a six-hour whole-notebook guarantee.

## Remaining environment-dependent checks

This host has no CUDA device and prohibits local sockets, including Jupyter's
TCP and IPC transports. Therefore a live Jupyter kernel/Colab browser session
could not be exercised. Sequential execution of all actual cells passed as
stated above; no claim of browser/UI validation is made.

Colab performs dense/sparse CUDA forward/gradient checks, exact CUDA fixture
resumption, actual batch 64 training and batch 256 inference before science.
These gates must pass on the allocated GPU. A100 40 GB is the intended target,
not a measured memory certification. GPU availability, driver/wheel support,
network dependency downloads, browser uploads and Drive permissions remain
external dependencies. Unsupported Python/hardware fail with explicit messages.

The primary run also records any degenerate pruning-branch abstentions. Native
sensitivity failures retain explicit receipts and suppress incomplete-roster
summary scores rather than substituting another detector.

## Interpretation

A passing audit establishes implementation checks and recorded protocol gates.
It cannot establish ST-GCN superiority, an independent holdout, absence of
inherited TEST scaling, or perfectly controlled training against an author fit
whose original random split and individual training history are unavailable.
