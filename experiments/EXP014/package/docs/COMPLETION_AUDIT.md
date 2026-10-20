# EXP014 delivery and completion audit

## Delivery status

**Executable comparison package delivered; real NASA GDN–Hundman results not yet generated.** Development validation passed 17 CPU regressions. The original EXP013 checkpoint/score archive is still required; no A100 was available here. A CPU fixture is not a scientific NASA experiment or GPU PASS.

The final bounded decision is two frozen GDN mission-checkpoint inference jobs, one exact additional Hundman detector,81 new target-residual streams and **zero new fits**. Original EXP012 and all 22 completed EXP013 fits, checkpoints, thresholds, scores, conclusions and source packages are immutable. No historical fit is continued from an incompatible checkpoint.

## Evidence inspected and preserved

The supplied Overleaf report, scientific/source-to-claim audit, frozen exports, protocol, source packages, engineering amendments, checkpoint/runtime freezes and completed notebook receipts were inspected. All 173 members in the report delivery manifest and 101 members of its original source-package manifest matched their recorded hashes. This verifies supplied evidence bytes, not the absent raw models/score files or every underlying scientific claim.

The bundle includes 2001 byte-verified immutable input files. It carries the NASA arrays/annotations,486 exact EXP012 GCN/LSTM score streams and their original seals/subset bindings, the frozen EXP012/EXP013 report exports, original freezes/protocol/report sources, and pinned native GDN code. `INPUT_INVENTORY.json` and `audit/INPUT_INVENTORY.csv` identify their hashes and intended Drive containers. The original 1053-stream EXP012 complete seal is retained unchanged; only the486-stream subset needed here is included/reverified.

The missing contract is exactly252 EXP013 constituents: two selected checkpoints, their completion/identity records, two calibration files, the original score seal, and81 native GDN score/record/commit triples. Nine file hashes are independently known from original freezes. The remaining243 hashes must be read from the original hash-pinned score seal; they are not fabricated. The expected completed archive is `/content/drive/MyDrive/EXP013_COMMAND_REFERENCE_V1_E03_CONV/main/FINAL_RESULTS.zip`. An equivalent 252-file ZIP/directory is sufficient. No CATCH checkpoint is required.

## Scientific decisions fixed before new annotation evaluation

| Evidence panel | Models | Source | Interpretation |
|---|---|---|---|
| Common detector | EXP012 GCN/LSTM; frozen EXP013 GDN + exact EXP012 detector | Old results reused; GDN column 0 forecast inverted through both scalers | Common-detector forecasting evidence with input/lookback/fitting/seed differences |
| Complete NASA pipelines | EXP012 GCN/LSTM; native-score EXP013 GDN/CATCH adaptations | Frozen reported count/timing tables | Controlled NASA adaptations under different frozen detectors |
| Paper context | Original published methods/results | Original report and primary methodological sources | Different protocols; no combined leaderboard |

CATCH–Hundman is omitted because a reconstruction-score hybrid adds no essential forecast-residual comparison. STGLR has a bounded source/checkpoint fidelity gap; OmniAnomaly has an unvalidated legacy-environment/port requirement. Neither exclusion is evidence of poor anomaly performance. No new hyperparameter tuning, model selection, calibration, training or architecture/numerical correction is introduced.

`docs/SCIENTIFIC_DECISION.md` records the method, inverse scaling, support and reasons. `PROTOCOL_FREEZE.json` is committed before new score generation. The exact original Hundman implementation and evaluator are byte-preserved. Native GDN inference retains t=5/batch 32 boundaries and full smoothing history; only afterward is native replay cropped at 250. The separate Hundman stream starts at 250. Each native array/flag/issue reference must reproduce bitwise. Checkpoints, FP32 outputs, states and RNG must restore exactly.

The first P-2 label row,81 unique channels,104 original events, common support, short channels, strict point metrics without adjustment, nominal alarms and actual notification timing are retained. Scores are committed/sealed/verified before new annotation parsing. Historical label/TEST exposure and released TEST-extrema preprocessing remain explicit. Uncertainty is descriptive paired-channel resampling conditional on fixed saved fits; it does not resolve shared-model/channel dependence or one versus three initialisations.

## Engineering status and remaining production gates

The runner reuses the original tagged checkpoint reader and proven two-generation hash-verified backup routine. New channel units and reports are published atomically. Code/config/environment/input identity binds recovery; incomplete, stale or corrupt outputs stop. Periodic 180-second backups and forced stage/failure backups are automatic. Progress, measured channel timings, ETA and memory receipts are visible. Original experiment folders are never written.

Full-shape native CPU checks, exact restoration, gradients against the existing dense oracle, six historical detector reference streams, all 486 historical counts/four mission results, and miniature source-ZIP→inference→seal→evaluation→figure→archive execution passed. The miniature run simulated interruption and VM loss, restored/skipped committed work, rejected corruption, and matched uninterrupted numerical output exactly. Details and honest FP32 differences are in `docs/TEST_REPORT.md`; machine-readable evidence is in `audit/TEST_REPORT.json` and its log.

**Unexecuted:** Google authentication/mounted-Drive performance, CUDA-wheel installation in Colab, actual A100 40 GB memory/throughput, recovery/loading of both real checkpoints,81 bitwise native NASA replays and81 new NASA detector streams. A conservative working-set estimate is not a measured GPU validation. The notebook must pass the real gates before `FINAL_AUDIT.json` can claim A100 validation.

## Handoff after actual completion

Use `LATEST_EXPORT.json` in the new EXP014 Drive folder to identify the verified content-addressed final archive. The archive contains the input/run/protocol freezes, both measured native full-shape receipts, per-channel residual/score/flag/issue arrays and commits, score seal, progress log, report seal and final audit.

Supply `reports/REPORT_READY_SUMMARY.md`, `mission_metrics.csv`, `mission_seed_metrics.csv`, the separate common/native channel tables, event notifications/outside-support file, `paired_channel_bootstrap.json`, `table_rows.tex`, PNG/PDF figure, `FINAL_AUDIT.json` and this decision/audit to the report-writing task. `src/export.py` copies verified report-ready assets without rescoring or overwriting the source.

The report writer must preserve EXP012's 0.769420/0.715577 GCN and 0.707182/0.810694 LSTM results, EXP013's negative synthetic conclusion and its disclosed auxiliary model-seed export defect. That auxiliary defect is not imported into NASA aggregation. New retrospective comparisons cannot erase historical TEST exposure or support independent generalisation, architecture-only causality, equivalence or operational deployment claims. Failed replay/feasibility conditions and negative results must remain visible. Stop after these exports; this package schedules no further experiment stage.
