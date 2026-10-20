# EXP013 E01 — startup and recovery guide

Engineering amendment to `EXP013_COMMAND_REFERENCE_V1`, 9 October 2026. The original release and EXP012 remain unchanged. Use this notebook and its matching E01 bundle together.

## First start

1. Open **EXP013_COMMAND_REFERENCE_V1_E01_Run_All.ipynb** in Google Colab. Select **A100, 40 GB** and a compatible Python 3.12/3.13 runtime. Keep the frozen dependency pins. Reserve at least **20 GiB free local disk and 20 GiB free Google Drive quota**; these are engineering minimums, not predictions of final archive size.
2. Run the first cell and authorise Google Drive. It creates or reuses `EXP013_COMMAND_REFERENCE_V1_E01_recovery` in My Drive. No Drive FUSE mount is used. For duplicate folder names, enter the intended folder ID in `DRIVE_FOLDER_ID`; do not change scientific settings.
3. Run setup and upload **only EXP013_COMMAND_REFERENCE_V1_E01_input_bundle.zip** when requested. It contains the original NASA arrays, frozen EXP012 reports and the exact pinned GDN/CATCH source ZIPs. No EXP005 sidecar, checkpoints.zip, EXP012 replay or separate NASA data upload is needed. The verified bundle is cached to Drive automatically for subsequent VM resets.
4. Run the engineering/performance cell **before committing to training**. It tests the actual A100 at all frozen batch sizes and shapes, exact GPU resume and source equivalence, local/Drive capacity and a real verified Drive transfer. Inspect the printed per-role timing, peak GPU memory, first-epoch and epoch-cap projections. No NASA research fit is performed in this cell. Its synthetic full-shape optimisation weights are discarded.
5. If the gates pass and the measured projection is acceptable, run the remaining cells or use Run all. Training, frozen selection, confirmation, TEST scoring, sealing and all required backups run automatically. Run all also executes the gates before any research fitting; it continues automatically after a successful preflight.
6. The notebook requests the historical **labeled_anomalies.csv** (or **02-labeled_anomalies.csv**) only after the TEST score seal has been verified. Required SHA-256: `057ce2d6c8875982bf4e5404aefea14efdcbce413d80826d2b737c95b59b7539`. This is the only scientific input that needs a later manual upload. An optional edit is not needed to enable backups. After this upload, the unmodified common evaluator, audit and persistent report export run automatically. The CSV is cached separately in the same private Drive folder, bound to the verified score seal, and is retrieved only after that seal passes on a later restart. It is excluded from training recovery archives and the supporting bundle.

## Recovery

| Interruption | Action | Behaviour |
|---|---|---|
| Browser disconnect, same Colab VM/kernel | Reconnect and inspect live output; do not start a second worker | Work may continue if Colab keeps the VM. A browser connection does not guarantee runtime survival. |
| Python kernel restart, `/content` survives | Rerun from the first cell | The owning-kernel guard terminates its worker. Verified local checkpoints and completed jobs are reused; dependencies and cached bundle are checked. |
| Complete VM reset, `/content` erased | Reopen the E01 notebook, select the same A100 environment, rerun from the first cell and authorise Drive if prompted | The bundle is retrieved from Drive; the most recent compatible, fully verified recovery generation is restored automatically. Completed fits are not retrained. |
| Backup upload interrupted | Rerun after connectivity/quota is restored | Pending uploads are ignored. The last verified generation is retained. A verified newer commit is recognised even if its acknowledgement was lost. |
| Newest archive corrupt | Read the explicit rejection message | Recovery can fall back to an older fully verified compatible generation. If none passes, execution stops. No verification is bypassed. |
| Source/runtime identity mismatch | Stop and retain all backups | Python patch version, Torch build, scientific package versions, GPU model/memory/driver and deterministic settings must match. A changed environment needs separately justified compatibility validation; do not edit identities or erase seals. |

For an ordinary local rerun, always start at the beginning so startup verification and the initial durable backup occur. Only an empty local work directory is automatically restored. A non-empty directory is never overwritten. If local state is demonstrably corrupt, retain it for diagnosis and use a fresh Colab VM to restore from Drive; do not selectively delete scientific seals or completed checkpoints.

## What the progress output means

The first MSL graph job contains **32,321 FIT windows / 127 FIT mini-batches**, plus **864 STOP windows / 4 batches**, at the unchanged batch size 256. The corresponding SMAP jobs contain **79,331 / 310 FIT batches** and **1,728 / 7 STOP batches**. `JOB n OF 22`, `FIT START`, batch/epoch progress, rolling phase ETA, elapsed seconds, GPU allocated/reserved/peak memory, completed-fit count and 30-second liveness heartbeats are unbuffered. ETA becomes meaningful after observed batches and includes no promise about later stages. `BACKUP_START` explicitly indicates that scientific computation is paused for a safe snapshot. `DURABLE BACKUP VERIFIED` identifies the recovery receipt and transfer duration.

## Recovery-loss target and storage

Safe batch/validation checkpoints are attempted when **600 seconds of subsequent execution** have elapsed since a verified backup; the first epoch has an initial checkpoint. Epoch and stage boundaries save state, and completed fits force a verified durable backup. Weights, both applicable optimisers, retained scheduler state, pending CATCH mask gradients, all RNG states, exact epoch sample indices, cursor, accumulators, best-selection state and scientific identities are included.

The intended training-compute loss window is approximately **10 minutes plus the currently executing batch**. This is **not a hard 10–15 minute wall-clock guarantee**: serialising a multi-GiB CATCH checkpoint, local hashing, a slow batch and Drive transfer/readback take time. Training pauses during transfer and stops if durability cannot be established. Actual checkpoint/upload duration is printed and retained. A reset during the first-ever upload has no prior checkpoint to restore. Full-shape GPU timing and large-checkpoint transfer behaviour cannot be certified by the local CPU tests.

Completed inference calls are cached with model/source/data/input identities and SHA verification. A long, interrupted inference call can still repeat; the current call is the scoring recovery unit, so no 15-minute scoring-loss guarantee is made. The preflight projection excludes selection, confirmation, TEST scoring, checkpointing, transfer and contention.

Recovery uses immutable archive objects and a **commit-last** receipt. Unchanged large completed checkpoint objects are reused, rather than re-uploading every model. Changed active checkpoints are uploaded only at safe periodic/milestone boundaries, never after every mini-batch. At least two fully verified generations are retained; an older generation is removed only after replacements pass verification. Unreferenced completed recovery objects are cleaned up. Incomplete upload remnants are ignored and may consume quota; remove them only after confirming two intact completed generations. Labels, release cache and final reports are separate objects and are not removed by recovery retention.

Final tables, figures, research report, scientific audit, execution audit, provenance and report checksum receipt are saved automatically to Drive. The notebook also offers immediate report ZIP/receipt downloads. The Drive recovery folder, not `/content`, is the durable state after a VM reset.

## Troubleshooting

- **No batch output yet:** look for `PREFLIGHT STEP START`, `FIT START` and heartbeat phase. Initial construction, source validation or snapshot upload may be running. Do not infer an epoch is complete from a live GPU process.
- **Drive permissions, timeout or quota failure:** execution stops and preserves prior good backups. Resolve the cause, then rerun from the first cell. No scientific fitting continues unprotected.
- **CUDA OOM / unsupported GPU / nonfinite new-model objective / incompatible source:** stop and retain logs. Do not lower batch size, model width, precision or detector settings to make it run. The original explicit published numerical-failure policy remains in force.
- **Packages replaced while already imported:** restart the Python session as instructed, then rerun from the beginning; installed pins and durable state are reused.
- **Native CATCH transfer takes longer than expected:** use the printed duration and actual receipt age. The 16 MiB Drive probe is a permissions/integrity check, not a guarantee of multi-GiB throughput.

No surviving EXP013 model checkpoint or external recovery archive from the lost Colab VM was found. In particular, the old `EXP013_work/models/MSL__graph__17/last.pt`, its selected checkpoint/history and external `EXP013_COMMAND_REFERENCE_V1_resume.zip` plus SHA receipt are unavailable. E01 starts those unfinished fits again; it does not invent recovered progress. The original successful A100 runtime gate/log is also unavailable as an inspectable file, so this release requires its own actual GPU gate.

Read `E01_ENGINEERING_AUDIT.md`, `E01_AMENDMENT_MANIFEST.json` and the unchanged `PROTOCOL.md` in the bundle for details. Engineering PASS is not evidence of improved anomaly detection. NASA results remain exploratory retrospective, with historical TEST exposure and inherited TEST-based scaling; the protocol's limitations are unchanged.
