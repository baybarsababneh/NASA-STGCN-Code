# EXP007_GPU_V1 — engineering correction E01

## Resume

1. Stop/close the failed notebook; do not run two notebooks together.
2. Open `EXP007_GPU_V1_E01_Resume.ipynb` in Colab with the same GPU model.
3. Run all, authorise Drive, and upload `EXP007_GPU_V1_E01_input_bundle.zip` once when prompted.
4. Subsequent resumes use the same E01 notebook and saved E01 bundle.

Do not manually edit the old worktree, regenerate its manifest, delete its freeze,
or copy its checkpoints into the new run.

## Exact correction

Add `import math` to `src/execution.py`. No other packaged Python source, configuration,
protocol, graph, split, seed, optimizer, stopping rule, promotion gate, or TRAIN array changes.
The package manifest is regenerated to describe the corrected source truthfully.
E01 is an engineering execution revision; the scientific experiment is still EXP007_GPU_V1.

The original error occurs in `clean_json(result)` **before** creation of the checkpoint
archive and completion marker. Although the first robustness fit completed training,
its weights were not durably committed by this code path. That fit restarts from its
original seed. The total planned scientific fit count remains 1,116; record the interrupted
training attempt separately as engineering overhead, not a second completed research fit.

## Provenance and safety

Original Drive directory: `MyDrive/NASA_EXP007_GPU_V1/results/` — untouched.
Corrected directory: `MyDrive/NASA_EXP007_GPU_V1/results_E01/`.
The original bundle and dependency lock must still be present alongside these directories.

The notebook verifies the original bundle/freeze, the exact one-line source difference,
unchanged scientific files, and **zero original completion markers**. If any original
committed fits exist it stops: those require a reviewed migration instead of silently
relabeling their run hashes or discarding them. It also stops on an active lease.
After an abandoned runtime, wait ten minutes rather than delete the lease.

Before training, a separate process imports the real corrected `execution` module,
checks finite/non-finite nested JSON serialization and package integrity, and requires
an exact match to the parent software/driver/GPU fingerprint. No tolerance is relaxed.
If Colab supplies a different fingerprint, the notebook stops; preserve the evidence
and resolve the environment mismatch rather than edit the freeze.

`results_E01/engineering_E01/` records the amendment, old/new hashes, parent freeze,
parent source, parent manifest, and parent effective configuration. The new run creates
its own normal `RUN_FREEZE.json`; existing commit/hash/resume checks remain unchanged.
Save this directory with the full result export. No NASA TEST or anomaly labels are used.

## Verification scope

Local checks verify the exact bundle diff, source/manifest checksums, notebook syntax,
serialization regression (old function fails; corrected function passes), and recovery
preconditions. The Colab notebook additionally verifies the real imported module and
CUDA environment before training. No GPU training was run to validate this import-only
correction here, and no completed scientific result is claimed by these engineering tests.
