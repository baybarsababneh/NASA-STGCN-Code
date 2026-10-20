# Integrity and recovery

## What is protected

Original EXP012/EXP013 paths are read-only inputs. The notebook verifies the standalone-pinned bundle manifest, all embedded inputs, the original EXP012 complete score seal and its explicit 486-stream subset, and the original completed report manifests. Recovered selected checkpoint/identity/threshold files match the hashes in the original STUDY_FREEZE. The original `scores/SCORE_SEAL.json` matches its SHA in FINAL_AUDIT, then pins every required GDN native reference file. A locally cached copy does not excuse a changed original.

The EXP014 `RUN_FREEZE.json` binds code, numerical environment, protocol/configuration and verified input content. Different inputs or code cannot resume an old run. Actual A100 model name/bytes, driver, CUDA, CuDNN, PyTorch, numerical libraries and determinism are recorded. Python 3.12/3.13 may start a new EXP014 run, but the exact recorded environment must match to resume it. Native reference replay additionally verifies scientific equivalence to the completed GDN score output.

## Commits and backups

Each channel is computed locally, written into an uncommitted staging directory, sealed with file hashes, verified, and published by one directory rename. Only then is the channel marked complete. Uncommitted staging is discarded on restart. Corrupt or stale completed units are rejected, not promoted or silently replaced.

The proven EXP013 `simple_backup.py` is reused as `src/backup.py` with only its import target and experiment version changed. It creates local archives, copies immutable content-addressed objects to Drive, reads them back for hash verification, and publishes a separate commit receipt last. Two committed generations are retained. Periodic backups run every 180 seconds at safe batch/unit boundaries, and forced backups occur at startup, each mission's memory/restore gate, mission completion, score sealing, final reports and failures. A failed copy cannot create a successful receipt.

A channel is the smallest resumable scientific unit. A disconnect may lose the current channel and locally committed channels since the last verified periodic backup. They are recomputed with the same frozen model/settings; no training hours are at risk in this inference-only stage. The `BACKUP VERIFIED` message distinguishes durable state from a local `COMPLETE` message.

## Resume

Reconnect A100 40 GB, reopen the same notebook and **Run all** with unchanged `RUN_SLOT`, input bundle and original source. The bundle is retrieved from Drive. An empty local work directory triggers automatic restoration of the newest valid backup. Every recovered byte is verified; if its newest object is corrupted, the previous valid generation may be restored, with an explicit `REJECTED BACKUP` diagnostic. A mismatched runtime stops immediately rather than silently using an older/incompatible run.

Verified completed channels are skipped. When the score seal is present and valid, the inference stage is skipped. Existing sealed reports are reused. The original input archive is never replaced and no old E01/E02/native-convolution fit is resumed.

If a live local committed unit is corrupt, stop and retain diagnostics. In a fresh Colab runtime the automatic backup restoration supplies the last valid durable state. Do not edit a unit's receipt or copy incomplete output into it. If both backups are invalid or runtime compatibility fails, supply the diagnostic to investigate the exact mismatch. Do not weaken a gate.

## Final export and traceability

The final ZIP is built and rehashed locally, then copied and rehashed through Drive. `exports/EXP014_FINAL_RESULTS_<hash>.zip` is immutable. `LATEST_EXPORT.json` points to the newest verified export; the first `FINAL_RESULTS.zip` alias remains unchanged. Verified resumes can add log entries and produce a new archive identity while keeping scientific score/table bytes unchanged.

Key evidence: `INPUTS.json`, `PROTOCOL_FREEZE.json`, `RUN_FREEZE.json`, `MEMORY_GATE.json`, `SCORE_SEAL.json`, channel `COMMIT.json` and `record.json`, `reports/COMPLETE.json`, `FINAL_AUDIT.json`, `PROGRESS.jsonl` and `SNAPSHOT_MANIFEST.json`. The log records score verification before annotation parsing. Historical exposure is still disclosed; the gate does not assert physical label isolation from earlier reports.

The notebook uses a local isolated venv and fresh worker processes. The Colab kernel can keep its own numerical packages; no kernel restart is needed. Exact pins are checked before scientific execution. The bundle and notebook are paired by ZIP/manifest hashes.

Use one active Colab runtime per Drive run folder. Local file locking detects duplicate workers in one VM; it is not a cross-VM Drive lease. Running the same folder from two VMs is unsupported.
