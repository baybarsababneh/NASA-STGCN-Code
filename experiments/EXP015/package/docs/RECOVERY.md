# Restart and recovery

Use one active Colab runtime per EXP015 run slot. Reopen the paired notebook, select the same A100 capacity and unchanged numerical environment, and choose **Run all** with the same `RUN_SLOT`. The notebook retrieves its hash-verified input bundle from the new EXP015 Drive folder. It reads the original EXP013 archive without writing to it.

Live computation occurs under `/content`. Each channel/arm/seed is written to an uncommitted directory, sealed and verified, then published by one directory rename. Incomplete staging is discarded. Completed units with a wrong hash, support or run identity stop the run. They are never silently promoted, replaced or treated as missing.

The EXP014 backup mechanism retains two committed, content-addressed Drive generations. It copies and reads back state objects, verifies every hash, and publishes a commit receipt last. Backups occur every 180 seconds at safe inference batch/channel boundaries and are forced at startup, full-shape gates, fit-job completion, complete score sealing, report completion and failures. `BACKUP VERIFIED` identifies a durable commit; `COMPLETE` alone identifies local completion.

On a fresh VM, an empty work folder is automatically restored from the newest compatible valid receipt. A corrupt newest object is explicitly rejected; the previous valid generation can be restored. All restored bytes and the numerical/runtime identity are checked. Original checkpoint and native-output constituents are independently reverified from the immutable source on every start. No model is retrained.

The channel/arm/seed is the smallest resumable scientific unit. A disconnect may repeat the currently unfinished channel and locally completed units since the last durable backup. Verified completed units are skipped. A valid score seal skips all inference and detection; sealed reports are reused. Logs and archive identities may grow on resume, while scientific score/table bytes remain unchanged.

A100 40 GB and 80 GB can each start a new run, subject to the same full-shape and bitwise native-output checks. Exact environment/device identity is required to resume a slot. Do not mix capacity, Python, driver or dependency changes into an existing slot or weaken a failed gate. The file lock detects duplicate workers within one VM; it is not a cross-VM lease.

If a gate fails, retain the diagnostic and verified backups. Missing immutable inputs are enumerated in `cache/MISSING_OR_INVALID_INPUTS.json`; the required original file is `MyDrive/EXP013_COMMAND_REFERENCE_V1_E03_CONV/main/FINAL_RESULTS.zip`. An input bundle, reports ZIP or backup object directory is not the completed experiment root. No score transformation, altered model, reduced batch or retraining is an authorised recovery path.

The immutable final export is `exports/EXP015_FINAL_RESULTS_<hash>.zip`. `LATEST_EXPORT.json` supplies its full SHA-256. The first `FINAL_RESULTS.zip` alias remains unchanged. Send the content-addressed final archive back for independent analysis.
