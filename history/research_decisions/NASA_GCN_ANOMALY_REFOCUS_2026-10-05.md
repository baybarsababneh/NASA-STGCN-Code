# NASA SMAP/MSL: anomaly-centred research decision

Assessment date: 5 October 2026. This is a new research recommendation, not a change to a frozen experiment or permission to run NASA TEST evaluation.

## Decision

Keep the final forecasting architectures and weights frozen. Stop the unsuccessful synthetic-detector selection branch. Prepare a bounded, controlled comparison of complete anomaly-detection pipelines that directly answers the original LSTM-to-GCN brief. Do not require a positive result before completing the study.

The current evidence establishes useful command-message access for forecasting within the matched architecture. It has not established useful anomaly discrimination for the final GCN models. The historical EXP002 detector had very high event recall and severe false-alarm burden. EXP005 demonstrated that better forecasting does not guarantee better anomaly detection. EXP008–EXP011 have not selected a detector.

## Evidence checked and limits

This reassessment used the supplied original brief and Hundman paper, current model implementations, the architecture audit/report, EXP011's frozen protocol, and the newly recovered EXP011 summary dated 20261005T070729Z. The earlier reviewer snapshot predates detector development and is not the current status authority.

EXP011's STUDY_CLOSE records 440 detector fits, zero forecasting fits, no NASA TEST or final-calibration access, and STOP_REQUIRE_NEW_INFORMATION for both missions. Every arm abstained in every outer fold because no inner threshold passed. Zero outer alarm/hit outputs are abstention bookkeeping, not measured performance of an operating detector. The supplied summary entries and embedded source bundle were hash-checked. Fit payloads were not supplied in that summary, so a complete 440-fit byte audit remains separate.

The final TRAIN calibration block was used in historical detector work. It was protected from the recent detector experiments, but must not be described as globally virgin or independently unseen. NASA TEST is historically exposed and any new benchmark is explicitly retrospective.

## Immediate stage: EXP012_ANOMALY_BENCHMARK_PREPARATION_V1

Question: with a fixed decision rule and common scoring population, do the final GCN forecasts support better anomaly detection than the matched message-removal model, a command-aware LSTM, a direct-fusion TCN, and persistence?

This stage prepares and freezes the benchmark; it does not itself read NASA TEST arrays or labels. The existing EXP011 STOP decision remains in force for further automatic synthetic-based detector selection.

1. Obtain the independent review and resolve substantive correctness issues. Complete the EXP011 fit-level audit when its full export is available.
2. Prepare a pinned, reviewed implementation of Hundman-style nonparametric dynamic thresholding and pruning as one publication-derived detector. It is not guaranteed to satisfy our earlier deployment gates.
3. Exclude channel-specific code exceptions motivated by known anomalies, including the official C-2 startup exception. Document every scientific departure from upstream and give the implementation its own version name.
4. Apply the same startup, scale conversion, batching, smoothing, threshold objective, padding and pruning rules to every forecasting model. Preserve original upstream defaults unless a correctness issue requires a documented uniform change; do not select among variations using NASA TEST.
5. Keep absolute-residual TRAIN-calibrated q=.99, method='higher', strict exceedance as a reference arm. This is a continuity/control rule, not a newly validated detector.
6. Use the same eligible 81 channels and seeds 17,29,43. Include final GCN and final matched message-removal checkpoints. Reuse available direct-fusion TCN checkpoints with training-budget/backend differences explicitly disclosed. Add a controlled command-aware LSTM with published architecture defaults and the existing chronological TRAIN protocol. Do not overwrite any historical experiment.
7. Freeze the full evaluation specification, source/config/checkpoint hashes and adaptations before generating new NASA TEST outputs. Keep score generation and label evaluation in separate programs.

Adaptive statistics calculated from incoming evaluation telemetry by a predeclared detector are inference, not permission to inspect labels or redesign the algorithm. Batch rules may use observations later than the points they flag. Record the actual issue time and do not describe backdated flags as immediate causal alarms. Scale-sensitive guards require predictions and observations in the same supplied telemetry units, reversing the project's additional fit standardisation where needed.

The primary scientific contrast is GCN versus final matched message removal under the publication-derived common detector. LSTM and equally informed direct fusion are essential secondary contrasts. The raw-q=.99 arm is a prespecified reference. Show all arms; do not choose a winning detector after NASA evaluation.

## Baseline budget

For the shortest study, prioritise the LSTM and the existing internal controls. For a stronger external comparison, add at most three fixed model families: GDN, OmniAnomaly, and CATCH. MTAD-GAT is a useful alternative if a faithful implementation is obtainable; the commonly used ML4ITS repository explicitly implements a modified architecture.

Use published configurations, the same chronological TRAIN roles, fit-only scaling, and nominal validation early stopping with each method's appropriate training objective. Publish the unequal historical development budgets. Do not tune reconstruction/probability methods by forecasting MSE merely to manufacture a common selection objective. Any additional baseline tuning must be declared before final inference and cannot use NASA anomaly labels.

Reproduction wrappers must disable test-label-optimal threshold routines and record whether score normalisation uses calibration data or evaluation covariates. Distinguish native complete pipelines from controlled common rescoring. Separate reported literature numbers from our reproduction results.

## Evaluation freeze requirements

- Retrospective status and prior NASA exposure stated prominently.
- Common target mask from index 250, with method-specific unsupported history handled as unavailable output or explicit latency, never silent exclusion of difficult events.
- Exact label-file hash, duplicate P-2 policy, channel roster, interval convention and boundary truncation counts.
- Corrected event F0.5 primary, with event precision/recall, false alarms, misses, nominal alarm occupancy and alarm starts always visible.
- Hundman sequence-overlap F0.5 secondary; strict pointwise P/R/F1/F0.5 and tie-aware average precision secondary. No point adjustment for primary comparisons.
- Alarm fragmentation, actual issue-time delay, and miss/censoring rates reported. No claims in physical time without verified sample timing.
- Mission counts aggregated within each seed before nonlinear metrics; mean across fixed seeds. Paired 10,000-channel bootstrap resamples the same channels for both methods and preserves seed structure. Report family/channel concentration and leave-one-channel-out sensitivity. Cross-channel dependence limits confidence intervals.
- Historical AP stays unchanged and marked affected; future grouped-tie AP is versioned and verified against a reference implementation.
- No metric, smoothing, alarm merging, threshold, checkpoint or seed changes after final outputs are examined. Failures remain results.

## Stop rule and interpretation

After this fixed comparison and the bounded external baseline panel, finish the report. A negative result answers the brief: a tuned graph forecaster may be predictively useful without improving the complete anomaly detector. Do not launch another automated search to obtain a positive NASA result.

Only independent new anomaly-development information, expert-reviewed nominal transitions, substantially longer development exposure, or a demonstrated validity defect justifies a new detector/model-development round. Supervised downstream models are scientifically allowed with proper independent development labels; the already exposed NASA TEST labels must not quietly become that development set.

If new information becomes available, the most plausible next mechanisms are context-versus-suspect window discrimination, command/regime-conditioned uncertainty, or masked/denoising graph representations. These address a different objective from one-step MSE. They are hypotheses, not selected methods or promised improvements. More graph layers, static edge-weight retuning, and recursive use of a one-step model are not justified by current evidence.

## Primary reading and implementation register

1. Hundman et al., KDD 2018: https://arxiv.org/abs/1802.04431 ; official code https://github.com/khundman/telemanom
2. Deng and Hooi, GDN, AAAI 2021: https://ojs.aaai.org/index.php/AAAI/article/view/16523 ; https://github.com/d-ailin/GDN
3. Zhao et al., MTAD-GAT, ICDM 2020: https://arxiv.org/abs/2009.02040 ; modified third-party implementation https://github.com/ML4ITS/mtad-gat-pytorch
4. Su et al., OmniAnomaly, KDD 2019: https://github.com/NetManAIOps/OmniAnomaly
5. Dai and Chen, GANF, ICLR 2022: https://arxiv.org/abs/2202.07857 ; https://github.com/EnyanDai/GANF
6. Campos et al., NCAD, IJCAI 2022: https://www.ijcai.org/proceedings/2022/0394.pdf
7. Cuellar et al., Engineering Applications of Artificial Intelligence 2024: https://doi.org/10.1016/j.engappai.2024.108083
8. Wu et al., CATCH, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b25c39788e5cf11d3541de433ebf4c0-Abstract-Conference.html ; https://github.com/decisionintelligence/CATCH
9. Lai et al., STGLR, Sensors 2025: https://doi.org/10.3390/s25020310 ; Chen et al., PSTG, Entropy 2026: https://pmc.ncbi.nlm.nih.gov/articles/PMC13114900/
10. ESA-ADB benchmark: https://arxiv.org/abs/2406.17826 ; https://github.com/kplabs-pl/ESA-ADB

Evaluation context: Kim et al., AAAI 2022 https://ojs.aaai.org/index.php/AAAI/article/view/20680 ; TSB-AD, NeurIPS 2024 https://github.com/TheDatumOrg/TSB-AD ; TAB, PVLDB 2025 https://www.vldb.org/pvldb/vol18/p2775-hu.pdf ; TimeSeAD, TMLR 2023 https://github.com/wagner-d/TimeSeAD .

This assessment performed no new model training, detector development run, or NASA TEST/anomaly evaluation. Frozen outputs and the reviewer package were not modified.
