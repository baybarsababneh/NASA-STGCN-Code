# EXP001–EXP015: experiment and runtime audit

Prepared for Baybars Ababneh's NASA SMAP/MSL report, 20 October 2026.

The appendix table is suitable for inclusion, provided its last column remains
**Recorded runtime**, with the measurement scope retained. There is useful
timing evidence for 13 experiments. However, most of it concerns fits or
individual stages, rather than complete experiment wall time. EXP010 and
EXP011 have insufficient runtime evidence. These records provide computational
context, not a fair speed comparison between architectures or experiments.

The purposes and outcomes were reconciled with the current concise report,
the original detailed report, the early saved-model records, and the final
experiment summaries. No models were trained, detection scores recomputed, or
research results changed for this audit. The subsequent
author-released LSTM benchmark is not relabelled as part of EXP012.

## What the timings mean

- A **summed fit time** adds the recorded elapsed durations of individual fits.
  It includes overlapping CPU workers and is neither whole-study wall time nor
  process CPU time. It must not be divided by a guessed worker count.
- **Stage elapsed time** comes from an explicit run timer or an executed
  notebook cell. A resumed run can reuse earlier work, so its timer does not
  recover the original launch's duration.
- **Scoring timers** concern frozen-model inference, detector processing, or
  cached score generation as stated below. They are not training durations.
- The table rounds measured values. It contains no hardware-scaled estimates,
  durations inferred from file dates, or assumed costs for missing stages.

## Reconstructed evidence

| Experiment | Exact recorded quantity | Scope and primary evidence |
|---|---:|---|
| EXP001 | 2,086.032 s = 34.77 min | Sum of `result.elapsed_seconds` from 243 CPU checkpoints in `checkpoints.zip`, under `v2_exp001_tcn_baseline`. |
| EXP002 | 31,066.867 s = 8.63 h | Same field for 243 checkpoints under `v2_exp002_tcn_gcn_fixed`. The independently supplied `all_runs(1).csv` agrees. |
| EXP003 | 28,054.699 s = 7.79 h | Same field for 243 checkpoints under `v2_exp003_tcn_no_graph`. |
| EXP004 | 35,731.239 s = 9.93 h | Same field for 243 checkpoints under `v2_exp004_tcn_gcn_learned_edges`; agrees with `all_runs.csv` and the mission totals in `full_parallel_run.log`. |
| EXP005 | 11,086.681 s = 3.08 h | Sum over all 486 retained confirmation fits, including both candidates confirmed within each mission, rather than only the final 243 selected fits. The confirmation summary in `full_tuning_run.log` agrees. Screening is additional. |
| EXP006 | 8,061.459 s = 2.24 h | Sum of synchronized `fit_wall_seconds` for 1,209 unique jobs in the full final GPU results table. |
| EXP007 | 5,621.862 s = 1.56 h | Sum of the same field for 1,116 unique jobs in the full final GPU results table. Repeated log output and resumed sessions are not counted as additional fits. |
| EXP008 | 10.969 s | Sum of `elapsed_seconds` for 534 unique saved-weight replays in `tables/replay_results.csv`. These are two-pass inference/check timers, not the complete readiness study. |
| EXP009 | 187.306 s = 3.12 min | `results/STUDY_CLOSE.json.elapsed_seconds` in the completed E01 export. The pre-amendment inventory contains 936 reused units; the final study contains 4,194 units. The earlier failed launch is not included in this timer. |
| EXP010 | Unavailable | The closure, run log, response tables, and all 3,078 committed unit records lack a computational duration. Synthetic-event `duration` and response delays describe signal samples, not execution time. |
| EXP011 | Unavailable | The summary records 440 classifier fits and zero forecasting fits. The closure, exported JSON records, and log provide no runtime timer. Individual classifier-fit payloads are not in this summary export. |
| EXP012 | 1,557.744 s = 25.96 min | The successfully executed preparation cell's `executionInfo.elapsed`, stored in milliseconds in `EXP012_ACTUALLY_EXECUTED.ipynb`, zero-based cell 5, within `EXP012_execution_provenance.zip`. This is not the later NASA TEST export/evaluation duration. |
| EXP013 | 8,913.470 s in epoch progress + 14,371.096 s in scoring calls | The final accumulated epoch heartbeat for each of 254 `(job, epoch)` pairs, across 22 mission-level fits, plus 49,781 `SCORE_CALL_COMMITTED.seconds` timers. Rounded separately to 2.5 h and 4.0 h in the table. |
| EXP014 | 103.912 s = 1.73 min | Sum of 81 unique `UNIT_COMMITTED.seconds` entries in the final `PROGRESS.jsonl`. These time scoring-unit processing, including commit/verification, rather than GDN training. |
| EXP015 | 158.456 s inference + 143.545 s detector = 302.001 s | Sum of the two timer fields over 729 unique committed scoring units in the final `PROGRESS.jsonl`, rounded to 5.0 min. Model loading, restoration/replay checks, reporting, and export work are additional. |

### Important scope checks

**Early CPU fits.** The original `src.zip` training functions put the model on
the CPU. Their elapsed timer surrounds training and validation, with data
preparation and final saving outside it. The processor model is not recorded
in the timing evidence. Smoke checkpoints were excluded, and every early
experiment was checked for the expected 81 channels and seeds 17, 29, and 43.
The checkpoint metadata were read without importing Torch, loading tensor
storage, or executing checkpoint-defined classes.

**EXP004 parallel execution.** `full_parallel_run.log` explicitly identifies
EXP004. It reports 18,003.23 s wall time for a resumed launch that completed
239 new fits and reused four existing ones. Its summed elapsed fit time is
35,336.27 s for those new fits, consistent with two workers. The appendix's
9.9 h instead covers all 243 fit timers. Neither number is a measured elapsed
duration for the complete 243-fit experiment.

**EXP005 screening.** The log reports 288 screening jobs, with 37 already
complete and 251 newly completed. Their printed one-decimal times sum to
4,597.1 s for the 251 new jobs. This partial, rounded display cannot recover
the complete screening cost. It is therefore not silently added as though it
were a complete measured stage. The 3.1 h table entry is explicitly the full
486-fit confirmation total, with screening additional.

**EXP006–EXP007 final GPU studies.** Their `fit_wall_seconds` timers bracket
the training function and synchronize CUDA before stopping. They include more
of each fit than the inner `elapsed_seconds` loop timers. The two fields are
not added together. The full result ZIPs are contained in the supplied
`exports-20261004T114850Z-1-001.zip` and
`exports-20261004T114932Z-1-001.zip`. The staged totals are:

| Experiment | Stage | Fits | Summed fit wall time (s) |
|---|---|---:|---:|
| EXP006 | Reference | 243 | 733.428 |
| EXP006 | Screening | 288 | 1,794.773 |
| EXP006 | Confirmation | 486 | 4,888.419 |
| EXP006 | Earlier-origin robustness | 192 | 644.839 |
| EXP007 | Canonical comparisons | 729 | 4,140.651 |
| EXP007 | Earlier-origin robustness | 144 | 411.787 |
| EXP007 | Final message-removal ablation | 243 | 1,069.424 |

These totals do not recover earlier CPU/CUDA development attempts, failed
engineering checks, paused sessions, setup, backups, or all export costs.
The appendix therefore identifies them as final GPU fits rather than the
complete lifetime of an experiment identifier.

**EXP008 inference boundary.** The replay timer starts after checkpoint
loading, model construction, transfer, and raw-array loading. It covers two
prediction passes and numerical/state checks. The very short 11 s total must
not be presented as the time to run the complete study or load 534 models.
Nominal-alarm calculations are outside this timer.

**EXP009 resume boundary.** The original source computes elapsed time from
the start of the current detector invocation to its closure. E01 preserves
936 pre-existing commits. Its 187 s is a measured resumed-stage duration,
not an estimate of a fresh 4,194-unit run. The detector uses CPU NumPy
operations, with no recorded CPU model.

**EXP012 preparation boundary.** The notebook's successful preparation cell
runs the complete preparation command and verifies its closure. The freeze
and provisioning record identify an **A100-SXM4-80GB**, not 40 GB. Upload/Drive
interaction and environment installation occur in other cells and are excluded.
Within the preparation, 243 controlled LSTM fit timers sum to 1,221.204 s.
The much shorter GCN/control elapsed values describe imported-weight replay,
not new GCN training. The later score-export and retrospective-evaluation
freezes do not supply a complete additional runtime.

**EXP013 accumulated stages.** The progress logger records elapsed time
since the current epoch began, including its saved offset after an exact
resume. Summing every heartbeat would overcount. The reconstruction instead
takes the maximum per job and epoch, verifies that all 254 terminal records
are the final STOP batch, and sums those terminal elapsed values. It counts
22 fits, not 254 fits. Score-call timers include inference, writing the
cached scores, and sealing the cache. They exclude other pipeline work.
Per-batch engineering projections and historical implementation speedups
are not treated as executed experiment durations. Gaps between logged UTC
timestamps are not added as computational time.

**EXP014–EXP015 frozen weights.** Their final audits record zero new scientific
fits. Inspection of the original runner sources confirms the timer boundaries.
EXP014's progress timer covers per-channel inference, native replay,
residual/detector processing, and unit commit/verification. EXP015 times
`predict(...)` and `detect(...)` separately; its native-output replay check
and other work lie outside those fields. Its `reports/timing_summary.csv`
concerns anomaly-notification delays, not computational runtime, and was not
used for runtime reconstruction.
Both runner source manifests were checked against the code-manifest hashes
in their final run identities; the inspected timer code matches those sources.

## Hardware

The final EXP006, EXP007, EXP008, EXP010, EXP013, EXP014, and EXP015 records
identify NVIDIA A100-SXM4-40GB hardware. EXP012's timed preparation instead
identifies an A100-SXM4-80GB. The final EXP006–EXP008 stack records
Torch 2.11.0 with CUDA 13.0, while EXP013–EXP015 record Torch 2.6.0 with CUDA
12.6. These are historical measurements under different workloads and
software stacks. No throughput normalisation or cross-hardware extrapolation
is justified by the available records.

## Use in Overleaf

Upload `Experiment_Reference_Appendix.tex`, add `\usepackage{longtable}` to
the existing preamble, and insert `\input{Experiment_Reference_Appendix}`
before `\end{document}`, after the current appendix content. The report
already loads `array`, `booktabs`, `setspace`, and the other required packages.
The fragment does not issue a second `\appendix` command. Labels and table
numbers are assigned by the report.

The two-page preview uses the existing report's typography. The fragment was
also compiled in a temporary copy of the complete report. The report source
and findings were left unchanged.

## Reproduce the timing arithmetic

`audit_runtimes.py` uses only the Python standard library. Place the original
source files in one or more folders and run:

```bash
python audit_runtimes.py --sources /path/to/original/files --output reconstructed
```

The script names every required original file in its `source(...)` or
archive-reading calls and stops if one is missing. For EXP013, the standalone
`EXECUTION_PROGRESS.jsonl` must be the audited E03 run, not another experiment's
log. The included `timing_evidence.json` records SHA-256 hashes of the exact
inputs used. `early_fit_timers.csv`, `epoch_timers_EXP013.csv`, and
`timing_summary.csv` preserve the extracted numbers and measurement scopes.

The supporting files accompany the appendix audit only. They are not the
complete project archive intended for GitHub.
