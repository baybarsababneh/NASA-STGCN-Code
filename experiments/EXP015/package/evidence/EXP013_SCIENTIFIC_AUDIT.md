# Independent scientific audit — NASA SMAP/MSL through EXP013

Prepared for the research team, the university. 11 October 2026.

**Overall assessment: report arithmetic verified, with a genuine synthetic-reporting defect and explicit evidence limitations.** The experiment completed its execution requirements. It did not establish the prespecified graph advantage or synthetic confirmation. An execution `PASS` is not a scientific-success certificate.

This is a read-only audit of supplied frozen source, records, tables and event/case exports. No model was trained, score generated, threshold selected, detector changed or frozen result overwritten. The report preserves useful architectural interpretation and metric distinctions from the supplied earlier LaTeX draft, but replaces unfinished sections and does not recycle its old raw-quantile results as EXP013 findings.

## 1. Evidence and reproducible scope

The principal inputs are the supplied E03 input bundle, `reports.zip`, DATA/RUN/STUDY freezes, ENGINEERING_GATE, FINAL_AUDIT, execution-amendment records and notebook outputs. The historical EXP012 evaluation archive is included in the frozen bundle. Literature was checked against papers, publishers and official repositories; the supplied Hundman PDF is the original-paper source. The annotation CSV was inspected only for post-seal roster verification.

The independent script implements count-based metrics and the bootstrap directly. It does not call the experiment's metric functions. `audit/numerical_audit.json` records **1,412 passing checks**; `scripts/audit_metrics.py` provides the implementation. Check counts are assertions, not 1,412 independent statistical tests.

| Scope | Independently checked | Boundary |
|---|---|---|
| Frozen package | All 101 manifest-member SHA-256 values | Establishes supplied package integrity, not GPU execution |
| Reports | All 20 COMPLETE-member SHA-256 values | Establishes supplied report integrity |
| Freeze links | DATA/RUN/ENGINEERING hashes in STUDY; STUDY and COMPLETE hashes in FINAL; matching score-seal references | SCORE_SEAL bytes absent |
| Count metrics | All 891 channel rows, 22 mission-seed records and ten mission/role summaries | Counts/flags cannot be reconstructed from missing score arrays |
| Events | Hits, misses, notification rates, delays and timing identities | No visual replay of original score trajectories |
| Statistics | All ten NASA contrast intervals, associated point/alarm intervals and eight synthetic contrast intervals | Resampling is descriptive, conditional on the fitted models |
| Synthetic exports | All 60 family-summary rows, invalid/background counts, CONF gates and frozen objective arithmetic | Model-seed identity is lost in the flattened case export |
| Engineering | Ten role receipts, corrected exact replay and disclosed P01 gate/errors | No new GPU run or element-level Adam audit |

Important identifiers:

| Item | SHA-256 | Verification |
|---|---|---|
| STUDY_FREEZE.json | `a8cacfa66a92eafe8c03eb09b18e7b8d063bcb8b7e6edf613ea2dcdf80712fa8` | Bytes rehashed; matches FINAL and COMPLETE |
| reports/COMPLETE.json | `b31b4c446ebdf822c01057e30d4af38a75f4a00454b7315552695d376d154684` | Bytes rehashed; matches FINAL |
| SCORE_SEAL.json reference | `b9f76f695947aa0675cf9864da839eae700e74fa4f224e351547ce9548387bc6` | Matching receipt references only; file absent |
| Annotation CSV | `057ce2d6c8875982bf4e5404aefea14efdcbce413d80826d2b737c95b59b7539` | Supplied bytes rehashed |

The identifier list is also retained in machine-readable form in `source_to_claim.csv`.

## 2. Metric consistency and primary scientific result

There are 27 MSL and 54 SMAP unique supported channels, and 36/68 original events. The annotation CSV has 82 rows but 81 distinct channel identifiers. The frozen first P-2 row is retained; T-10 has no supplied label row. All model roles cover all supported channels. Every new event record is fully inside supported scoring; no left-censored or post-event notifications are reported.

Common scoring begins at index 250. Pooled nominal/anomalous supports are 59,213/7,766 MSL samples and 366,404/55,922 SMAP samples. Confusion totals, nominal counts and event counts agree across channel and mission-seed exports.

The evaluator uses hit-event count H, missed-event count M and wholly nominal alarm-run count U. Event precision is H/(H+U), event recall H/(H+M), corrected precision multiplies event precision by (1−FPt/N0), and corrected event F0.5 applies the standard F formula to that corrected precision and recall. Undefined domains and zero-hit cases follow the frozen contract. Strict point metrics use original flagged samples, without point adjustment. Mission ratios pool counts within each seed and then average seeds. AP is instead a channel-macro, then seed-macro metric.

| Mission | GCN corrected event F0.5 | Removal | Fusion | GDN | CATCH |
|---|---:|---:|---:|---:|---:|
| MSL | 0.0265 | 0.0333 | 0.0322 | 0.0674 | 0.0567 |
| SMAP | 0.0277 | 0.0369 | 0.0248 | 0.0599 | 0.0512 |

The GCN's SMAP strict point F1, 0.4026, is the largest descriptive mean in this panel. Graph-minus-removal and graph-minus-fusion point-F1 exploratory 95% intervals include zero. The four primary corrected-event graph-control intervals, each 98.75%, also include zero. These findings do not establish improvement, equivalence or absence of any heterogeneous graph effect.

High event recall and poor corrected event scores are arithmetically consistent. The GCN hits 33/36, 33/36 and 35/36 MSL events and 65/68, 57/68 and 59/68 SMAP events, but produces 929–1,366 and 1,562–3,806 wholly nominal runs per seed. Mean nominal occupancy is 23.68%/14.60%, starts are 20.62/6.92 per 1,000 nominal samples, and corrected precision is about 0.02. Event overlap gives credit for any within-event flag; point recall remains 0.3234/0.4932. Fragment merging alone could not remove this nominal occupancy.

GCN median first-notification delays among detected events are 10/15/14 samples MSL and 24/29/21 SMAP. Approximately 30–33%/29–34% of detections occur after one quarter of the labelled event duration. Median fragmentation is 2–3/2–4 runs among detected events. These are descriptive audit summaries, exclude misses and do not use operational minutes. `audit/event_timing_audit.csv` retains every mission/role/seed value.

The evaluator's notification is the first **issued flag within the labelled event**, including a run already active before onset. It is not necessarily a new operational alarm start. New issue indices equal target indices, so an overlap hit cannot produce a post-event notification under this contract. The absence of such records is consequently partly structural, not evidence that an operational alerting system has avoided late notices. Missing raw flags prevent quantifying pre-existing runs here.

## 3. Statistical interpretation

All supplied paired-channel intervals were reproduced using 10,000 draws and seed 20261006. Channels are paired across arms; all model seeds stay grouped; count metrics are recomputed inside every draw before seed means are taken. Historical LSTM contrasts use the preserved EXP012 counts and retain their complete-pipeline qualification.

The four primary NASA contrasts use 98.75% percentile intervals, a nominal Bonferroni allocation of alpha=0.05/4. Exploratory point/alarm and baseline intervals use 95% without multiplicity adjustment. The synthetic primary contrasts likewise use 98.75% intervals. These conventions are accurately labelled; they do not prove familywise coverage under correlated channels, shared context and shared mission-model fitting. Seeds are not resampled, models are not refitted, and the single GDN/CATCH initialisation omits optimisation uncertainty. No independent NASA confirmation or equivalence test is claimed.

Exploratory SMAP graph-minus-GDN and graph-minus-CATCH corrected-event intervals lie below zero. This is descriptive statistical support for worse GCN values within the adapted panel, conditional on these fits and dependence assumptions. It is not independent published-method superiority.

## 4. Calibration and confirmation

The frozen code selects among six EWMA/quantile candidates using TRAIN CAL and seed 17, with mean-channel occupancy ≤1% and starts ≤1/1,000. All exported EXP013 mission/role statuses are `SELECTED_TRAIN_SYNTHETIC`. **Do not describe this run as a CAL `NO_FEASIBLE_CANDIDATE` failure.** Selection-grid and threshold bytes are absent, so feasibility of the actual selected candidates cannot be independently replayed.

All six new-model mission combinations fail both frozen CONF nominal-alarm limits. Graph CONF occupancy is 10.94% MSL and 9.06% SMAP; its starts are 12.18/7.16 per 1,000. These CONF values are channel/seed means, unlike NASA pooled-count occupancy. The difference in aggregation is preserved in captions and traceability.

CAL length is only 44–431 MSL and 31–288 SMAP samples. Tail-quantile estimation and nominal-state transfer are plausible limitations, but no raw-score evidence here isolates a causal explanation. Hypotheses such as score saturation, level adaptation, state shift or unsuitable graph support must remain hypotheses.

Synthetic families contain invalid unchanged cases and clean-background overlap. Graph stuck cases include 108/324 invalid MSL and 186/648 invalid SMAP injections. Valid incremental stuck recall is 0.1019/0.0952. Long held-out variance/oscillation cases have about 30.25%/29.32% clean-background overlap for the graph. Invalid injections are excluded; unconditional hits cannot substitute for incremental evidence.

Every primary held-out synthetic interval includes zero, and the independent alarm gates fail. The frozen scientific decision is **NOT_ESTABLISHED_ON_BOTH_MISSIONS**.

## 5. Genuine defect: synthetic seed export and all-family weighting

**Severity: material auxiliary reporting/traceability defect; no evidence of a training or NASA-score change.** In `src/pipeline.py`, confirmation case flattening constructs a dictionary with the model seed and then expands `**item`. Each item already contains the injection seed. Python's later key assignment overwrites the model seed. The exported 19,602 case rows therefore have 1,782 injection seeds and zero rows labelled with model seeds 17/29/43.

The synthetic objective groups by seed and family assuming seed is the model initialisation. In the flattened export, groups instead follow deterministic injection seeds, generally tied to one family. The frozen all-family objective thus weights valid case groups, rather than the declared equal family weighting within each model seed. Spike has two settings, most families four, and unchanged stuck cases vary by channel. This is a real implementation mismatch, not a rounding difference.

The audit first reproduces the frozen calculation and its bootstrap exactly. It then performs a separately labelled equal-family arithmetic diagnostic, without selection or modifying any frozen file:

| Mission / contrast | Frozen all-family difference | Equal-family diagnostic | Change |
|---|---:|---:|---:|
| MSL graph − removal | −0.030145 | −0.026698 | +0.003448 |
| MSL graph − fusion | +0.057507 | +0.054767 | −0.002740 |
| SMAP graph − removal | +0.004338 | +0.003389 | −0.000949 |
| SMAP graph − fusion | +0.029028 | +0.028710 | −0.000318 |

This diagnostic does not restore lost model-seed identity and is not a replacement frozen inferential result. Variance and oscillation have balanced valid cases across channels/model seeds; their equal-family contrasts match frozen held-out results to less than 1e−12. Thus the primary numerical contrasts and the independently failed CONF alarm gates preserve the negative decision. Original per-fit confirmation JSONs are required for a complete model-seed-resolved correction. Any correction should be a separately identified analysis retaining the original export.

## 6. Other source/report corrections and engineering boundaries

The textual `published_short_FIT_support` field calls D-12 MSL, but DATA_FREEZE identifies it as **SMAP**, FIT length 187. CATCH's length-192 FIT windows omit it from training, while the shared SMAP model scores it. The report corrects the text without changing data or computation.

GDN score normalisation actually uses per-entity **CAL** median/IQR, with 0.01 denominator and causal four-point smoothing. A broad protocol description calling this FIT-only is insufficiently precise. The report states CAL correctly; neither stage uses NASA TEST for this transformation.

ENGINEERING_GATE reports all ten role receipts passing and exact corrected checkpoint replay on an A100-SXM4-40GB (42,405,855,232 bytes), deterministic FP32 with AMP and TF32 disabled. The command/telemetry dilated-convolution correction expresses unchanged equations and weights as lagged-window GEMMs. Native/corrected FP32 reduction order is not bitwise equal. Old native checkpoints are not certified bitwise continuations.

The P01 amendment applies only to SMAP fusion `root.discriminator.0.weight`: parameter atol 7.5e−6, rtol 2e−5; the other parameter gate remains atol 5e−6. Reported maximum differences after updates 1/2 are 6.02286309e−6 and 6.02309592e−6, with relative L2 about 1.84e−6. Corrected replay is exact. The receipt references a detailed numerical audit with SHA-256 `03a14486fe6bd59ecfd10b7a979b286457d6e1dfd96955877c6bc4100a40f369`, but its bytes were not supplied. This report does not independently inspect its failing elements, gradients or Adam amplification, and makes no new GPU-validation claim.

## 7. Comparison fidelity and literature

Three categories stay separate: common-protocol EXP013, historical EXP012 complete pipelines and each paper's original results. EXP013 changes GDN/CATCH splits, pooling, calibration and causal score export while retaining pinned native models/objectives and disclosed compatibility adapters. It is not a leaderboard reproduction. Removal reduces effective capacity; fusion adds capacity; baseline seed counts differ.

The historical EXP012 Hundman-adapted LSTM gives corrected-event F0.5 0.7072 MSL and 0.8107 SMAP; its detector, architecture and historical timing differ. Historical GCN values 0.7694/0.7156 do not identify the causal effect of EXP013's redesign. A separate count audit reproduces corrected event F0.5, point F1 and nominal occupancy for all six displayed historical rows (maximum difference 8.33e−17; `audit/historical_aggregation_check.csv`). No historical fit was rerun.

Published-paper metadata and specific results are recorded in `references.bib` and `source_to_claim.csv`. Hundman's printed Table 2 F0.5 values are inconsistent with its printed precision/recall under the standard formula. Both the printed values and the arithmetic discrepancy are disclosed; the paper's values are not silently corrected. GDN's quoted results are SWaT/WADI, not NASA. CATCH's MSL figures are taken from ICLR-era arXiv v2, not later revisions. MTAD-GAT and NCAD expand point hits to labelled segments; NCAD explicitly reports an optimistic best-TEST threshold. Their high F1 values cannot be ranked against strict-point TRAIN-calibrated EXP013 scores.

OmniAnomaly was not executed because of its legacy official environment; STGLR lacks sufficient located implementation evidence in this build. These are fidelity/execution gaps, not evidence those methods are ineffective or unavailable in principle. Anomaly Transformer and NCAD are conceptual comparators, not evaluated baselines. ESA-ADB is cited as a verified preprint and possible future independent telemetry resource, not assigned an unverified venue or treated as performed validation.

## 8. Leakage, missing evidence and claims that must remain limited

Historical experiments had exposed NASA TEST results and final TRAIN observations. The frozen package includes historical label-derived reports. The new CSV access gate protects the declared new execution sequence, not physical isolation from all historical labels or human knowledge.

The official NASA arrays inherit telemetry scaling using TEST extrema. FIT-only added standardisation cannot undo it. The study is retrospective and cannot be called leakage-free. Anonymised, anomaly-centred streams also do not establish long-horizon operational alert rates or spacecraft-wide aligned causal relationships.

Missing from the supplied completed-result subset:

- Raw per-stream scores, thresholded flags and `SCORE_SEAL.json` bytes.
- The 22 selected checkpoint files and per-fit completion/confirmation records.
- Detector CAL candidate grids, chosen threshold files and their detailed feasibility evidence.
- The full P01 element/gradient/Adam/RNG audit referenced above.
- Original time alignment/command semantics and raw telemetry needed to remove inherited TEST scaling.

Consequently AP is accepted as reported (its mission macro aggregation is checked), but not recomputed from continuous scores. The original score seal and checkpoint hashes cannot be rehashed here. Resolving these evidence gaps requires archive recovery, not retraining or retuning. Operational superiority, SOTA, independent NASA confirmation and deployment readiness remain unsupported.

## 9. Disposition

The report is suitable for technical review **with these disclosures**, not as a certificate of deployment capability. It answers the original question directly and preserves the negative confirmation. The next priority is complete archive preservation and a separate synthetic-metadata correction, followed by independent chronological validation and prespecified calibration/mechanism experiments on separate data. No observed NASA TEST ranking is used to justify retrospective model selection.
