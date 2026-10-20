# EXP013 E01 — technical engineering audit

**Release:** EXP013_COMMAND_REFERENCE_V1_E01. **Parent:** EXP013_COMMAND_REFERENCE_V1. **Date:** 9 October 2026.

## Scope and disposition

This amendment repairs execution, persistence and observability. It does not introduce EXP013 scientific results or claim a GPU speedup. The original notebook, input bundle, protocol, audit and guide were retrieved and SHA-verified before implementation; their original bytes were preserved. The original EXP012 historical reports embedded in the bundle remain unchanged and are reused without fitting or score replay.

The original release is preserved as history. The amended training file and regenerated package manifest are the only changed original package paths. All other original package members remain byte-identical, including architectures, data/synthetic fault definitions, complete scientific protocol, hyperparameter factory/job schedule, scoring, evaluator, metrics, official adapter/source pins, NASA array archive, TEST roster, EXP012 CSV/JSON exports, original CPU tests and original synthetic E2E fixture. Frozen-original copies of the old training file, manifest and build audits are supplied. The detailed amendment manifest lists both hashes for modified scientific-execution code and old hashes for every original member; the regenerated package manifest and outer release hashes bind the delivered bytes.

## Available state and missing evidence

The current project originals, supporting source/inputs, completed EXP012 reports and EXP013 preparation/build evidence were inspected. No completed EXP013 scientific result, valid model checkpoint or external EXP013 recovery archive was found. The reported Colab reset is consistent with no recoverable training. The prior runtime gate (`EXP013_work/ENGINEERING_GATE.json`), per-batch trace, `models/MSL__graph__17/last.pt`, `selected.pt`, `history.json`, and external `EXP013_COMMAND_REFERENCE_V1_resume.zip`/`.sha.json` are unavailable from that VM. Only the user's observed CPU/GPU utilisation and elapsed time are available. Therefore the precise cause of the 50-minute stall cannot be established retrospectively, and the prior GPU success is not reused to authorise E01.

## First-fit workload and slowdown investigation

Counts were derived from the original Windows construction and actual TRAIN metadata, without opening NASA TEST float arrays or annotations.

| Mission / role | FIT windows | FIT batches | STOP windows | STOP batches | Frozen batch | Epoch cap |
|---|---:|---:|---:|---:|---:|---:|
| MSL graph/removed/fusion | 32,321 | 127 | 864 | 4 | 256 | 25 |
| SMAP graph/removed/fusion | 79,331 | 310 | 1,728 | 7 | 256 | 25 |
| MSL GDN | 34,845 | 1,089 | 11,663 | 365 | 32 | 30 |
| SMAP GDN | 82,517 | 2,579 | 27,601 | 863 | 32 | 30 |
| MSL CATCH | 29,823 | 233 | 11,663 | 92 | 128 | 5 |
| SMAP CATCH | 72,477 | 567 | 27,597 | 216 | 128 | 10 |

The first job is MSL graph seed17. Its dataset is bounded by the unchanged per-channel FIT cap and weighted sampler. The original loader uses `num_workers=0`; no multiprocess DataLoader deadlock or Drive FUSE live-training directory is present in the source. Individual windows were sliced and tensorised in Python, then default-collated. The new path batches these identical slices with grouped NumPy indexing. Original DataLoader base-seed consumption, WeightedRandomSampler/RandomSampler order and their final generator-state draws are retained using the original pinned batch sampler. Only index plans, not a duplicated full training corpus, are materialised.

Actual TRAIN CPU collation measurements (single CPU thread, full B256) gave MSL median **3.268 ms → 0.718 ms** and SMAP **2.900 ms → 0.579 ms**, with tensor-byte parity. These are **collation-only ratios 4.55 and 5.01**, not end-to-end speedups. Their absolute scale is too small to explain the reported 50 minutes. No claim is made about A100 throughput from these measurements.

The original finite check and Python loss accumulator synchronised scalar GPU loss twice per batch. E01 performs one identical scalar conversion for both the immediate finite gate and the same accumulator before backward. Transfers remain synchronous, precision and optimiser arithmetic remain unchanged. Synthetic anomaly generation remains the original vectorised Torch operation. Clean and injected forward passes still independently calculate command encodings: an aggressive shared-forward optimisation could change gradient accumulation/rounding, so it was not introduced. CPU thread count, batch sizes, model dimensions, AMP/TF32/compile settings, sampler/caps and all training objectives are unchanged.

The original loop saved only at epoch boundaries and printed after an epoch. Thus `IDENTITY.json` alone and no visible progress were possible throughout a slow first epoch. E01 saves an initial batch-resumable checkpoint, emits unbuffered batch/epoch/fit telemetry and 30-second liveness output, and checkpoints safe FIT and STOP positions before a full epoch completes. The actual A100 preflight measures cold/steady full-shape FIT and STOP steps, transfer/forward/backward timing, peak memory, ten mission-role configurations, native first-MSL-graph arithmetic parity and an optional operator trace. It projects all22 epoch-cap compute costs with explicit omissions and no assumed early stopping.

## Minimum correctness correction

The original `train_job` constructed a mission-specific external scaler but passed the complete two-mission raw array dictionary into `transform`. A 55-column MSL centre/scale cannot broadcast onto a 25-column SMAP array. This would prevent native GDN/CATCH fits once reached, even though earlier single-mission fixtures passed. E01 passes only the selected mission's channels to the unchanged transform, consistent with the existing scoring path. It preserves every selected mission FIT statistic, value, window, loss and setting. A mixed-width MSL+SMAP regression test exercises both external roles. This repair is disclosed; it is not a baseline adaptation or hyperparameter change.

## Exact checkpoint design

The existing hash-prefixed Torch checkpoint envelope is retained, now streamed to local files rather than buffered in memory. Tests establish byte identity with the original serialiser. Payload hashes are checked before `weights_only=True` loading. Writes use local temporary files, fsync and atomic rename.

The new versioned checkpoint captures model state; main and applicable mask optimiser states; retained scheduler states; pending CATCH mask-generator gradients; Python, NumPy, CPU Torch and all CUDA RNG states; dedicated loader RNG; exact current epoch index plan/cursor; FIT accumulators; STOP plan/cursor/values and preserved pre-validation training RNG; epoch, best objective, patience counter, history and identities. Resume starts at the next committed safe operation. CATCH's frozen previous-gradient mask-step ordering and absolute batch index are preserved. Its original OneCycleLR constructors change initial LR but are never stepped; E01 retains their state **without introducing scheduler steps**. The original end-epoch LR policy is unchanged.

Completed jobs undergo identity, selected-file SHA and model-state hash checks before reuse. Frozen study/score/report outputs are verified and reused. Inference memoisation stores only completed calls from already hash-verified selected models, binds source/data/model/visible-input bytes and scoring arguments, and checks cache manifests before reuse. It preserves original scoring values and RNG effects. Future quarantined rows are not hashed by an earlier call. No new detector rule or prediction is introduced.

Workers run with the original `PYTHONHASHSEED=17` and unbuffered output. A Linux parent-death signal is armed before Torch/GPU import, with a race check. A kernel restart terminates its worker rather than leaving an orphan GPU process writing state. An OS file lock prevents two local experiment writers. Browser disconnect alone does not terminate a surviving kernel, and no mechanism circumvents Colab runtime restrictions.

## Durable Drive protocol

Google Drive is authorised at notebook startup. Scientific work and archive staging remain on `/content`; API transfers replace FUSE file-by-file verification. Each new object is uploaded under a unique pending name using resumable8MiB chunks and bounded retries. Server SHA-256 and byte count are checked; if Drive supplies no SHA-256, a full API download/readback SHA is mandatory. Metadata completion occurs only after verification.

Every archive includes member hashes, a manifest and safe paths. Files >=32MiB are separate immutable objects; smaller files are grouped. Unchanged large objects are referenced by later receipts. A new completion receipt contains the full local roster, outer/member hashes, source/protocol/runtime identities and object IDs. It is committed last, then the receipt and every referenced object are verified again. The only known-good state is never overwritten. Two verified generations are retained before superseded receipts/unreferenced completed recovery objects are removed. Pending remnants are ignored; they can consume quota and are not evidence of recovery. The supporting release, post-seal label cache and final report/receipt are separate persistent objects.

Startup lists completed receipts newest-first, checks compatibility and all hashes, downloads to local staging, verifies the complete restored roster, and atomically installs it only into an empty work directory. Corrupt/incomplete states are rejected visibly; an older fully verified compatible state can be used. Incompatible source/runtime identities stop execution, rather than weakening identity enforcement. Original runtime compatibility remains strict, including Python patch version, Torch build, GPU model/memory/driver and deterministic flags.

The target is a safe checkpoint after600seconds since the previous verified backup, plus a currently running batch. Scientific computation pauses while checkpointing/uploading; backup failure stops execution. This limits **unprotected training compute**, not wall-clock time. Multi-GiB CATCH checkpoint size, disk/hash time, slow batches, readback and Drive network make a universal10–15minute guarantee unjustified. First-ever incomplete upload has no earlier recovery state. A long incomplete inference call can repeat. Actual durations/receipt ages are logged. The small Drive probe cannot certify multi-GiB transfer time.

## Scientific gates and evaluation order

The execution wrapper retains original source/runtime checks and enforces:

`TRAIN staging → original actual-GPU engineering gate + E01 exact-resume/source-oracle tests → measured A100/Drive preflight → all22 FIT/STOP jobs → TRAIN-only calibration and selection → verified STUDY_FREEZE + durable backup → frozen synthetic CONF + durable backup → TEST scores → verified SCORE_SEAL + durable backup → annotation CSV request/read → unchanged common evaluation → original final audit → durable final report/audits`.

Study/score seal writes can be interrupted just before an ACCESS event append. E01 first verifies the existing seal, then appends a missing observed verification event at the current time before advancing; it never forges earlier access history. No labels are requested, retrieved, parsed or used in training, selection, preflight or score construction. The label CSV cache is accessed only after a successful verified seal and is bound to its SHA. Historical EXP012 reports contain label-derived material, so this is semantic workflow isolation, not a claim that prior exposure has been undone.

## Engineering evidence

- Retained original CPU science suite: **24 tests passed**.
- E01 execution suite: **18 tests passed**, including all-five-role source-oracle training/selected-checkpoint byte parity; FIT and STOP interruption/resume; optimiser, retained scheduler, RNG and CATCH gradient equality; mixed-mission scaling; kernel-parent death; interrupted object/receipt uploads; acknowledgement-loss recovery; corruption/fallback; full local/cache deletion and remote restore; completed-job reuse; large-object incremental reuse; retention; identity compatibility; score-cache parity; and pre-label/CONF/TEST ordering. The suite was exercised through `runpy` as used by the managed worker, preventing a notebook `__main__` discovery failure.
- Retained synthetic E2E fixture: **PASS**,22 fictional fits,44 fictional score streams, study/CONF/score sealing, common evaluation, final audit, verified reuse and tamper-blocked labels.
- The same E2E fixture with E01 incremental recovery and score-call caching enabled: **PASS**. No NASA research fits, NASA TEST array reads or annotation CSV reads were performed.
- Final release tests: **3 passed**, including unchanged job/hyperparameter AST checks, original archive integrity and automatic report/receipt persistence with tamper rejection. Notebook/static contract tests are recorded in `engineering_evidence/RELEASE_TEST_LOG.txt` and `E01_BUILD_VALIDATION.json`.
- Full test logs and CPU collation measurements accompany the bundle. Original build evidence remains byte-identical and is identified as historical evidence, not a new GPU result.

**Not executed here:** an actual A100 test, live Google Drive upload/authentication or full-sized CATCH durable checkpoint transfer. This environment has no NVIDIA GPU. Consequently E01 is built and CPU-audited, with **conditional GPU execution authorisation only after the notebook's actual gates pass**. The compulsory Colab gate runs the unchanged native full-batch tests plus all-five-role exact batch-resume/source-equivalence tests on CUDA. The preflight performs the outstanding live Drive and GPU timing checks before research fits. Successful engineering tests do not demonstrate anomaly-detection performance or guarantee uninterrupted Colab availability.

## Primary implementation references

- Pinned PyTorch v2.6 DataLoader base-seed consumption: https://github.com/pytorch/pytorch/blob/v2.6.0/torch/utils/data/dataloader.py
- Pinned PyTorch v2.6 RandomSampler/WeightedRandomSampler generator behaviour: https://github.com/pytorch/pytorch/blob/v2.6.0/torch/utils/data/sampler.py
- Official Colab authentication/Drive API example: https://colab.research.google.com/notebooks/io.ipynb
- Official Colab authentication implementation: https://github.com/googlecolab/colabtools/blob/main/google/colab/auth.py
- Drive resumable-upload protocol: https://developers.google.com/workspace/drive/api/guides/manage-uploads
- Drive binary checksum, size and metadata fields: https://developers.google.com/workspace/drive/api/reference/rest/v3/files

The scientific protocol, literature attribution, retrospective exposure limitations and original published-source fidelity constraints are unchanged. No EXP012 outputs were recomputed and no EXP013 scientific performance claim is made.
