# Experiment runtime evidence (technical records)

Contains retained timer evidence and a standard-library audit script for EXP001–EXP015. These are scientific execution records, **not** the report appendix or its PDF preview.

- `audit_runtimes.py` reconstructs recorded timers from saved logs and archives without running scientific jobs.
- `Runtime_Evidence_Notes.md` explains evidence quality, missing stages and hardware differences.
- `timing_summary.csv`, `timing_evidence.json`, `early_fit_timers.csv`, and `epoch_timers_EXP013.csv` contain the available recorded durations.
- `README_EXECUTION_RECORDS.md` explains extracted execution-timing metadata.

Summed fit times are not complete wall-clock experiment runtimes. No unrecorded stage duration should be inferred from file timestamps.
