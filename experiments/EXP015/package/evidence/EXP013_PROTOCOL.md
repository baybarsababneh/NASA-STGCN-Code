# EXP013_COMMAND_REFERENCE_V1 — scientific protocol

This release is a new experiment. EXP012, its selected checkpoints, detectors, score exports, metrics and completed evaluation remain unchanged. The objective is anomaly discrimination and useful alarms on **both** MSL and SMAP. Improvement is a hypothesis, not an expected outcome or a prerequisite for a valid result.

## Why this experiment

The audited final EXP012 primary Hundman-adapt results were:

| Mission | Model | Corrected event F0.5 | Strict point precision | Strict point recall | Strict point F1 | Nominal alarm occupancy |
|---|---|---:|---:|---:|---:|---:|
| MSL | GCN | 0.769420 | 0.463082 | 0.271440 | 0.342249 | 0.041275 |
| MSL | Matched graph removal | 0.715623 | 0.422322 | 0.261696 | 0.322679 | 0.047152 |
| MSL | Controlled LSTM | 0.707182 | 0.419893 | 0.315220 | 0.360076 | 0.057234 |
| SMAP | GCN | 0.715577 | 0.542005 | 0.108377 | 0.180478 | 0.014064 |
| SMAP | Matched graph removal | 0.796265 | 0.608021 | 0.117688 | 0.197175 | 0.011621 |
| SMAP | Controlled LSTM | 0.810694 | 0.644126 | 0.219675 | 0.327593 | 0.018497 |

The byte-preserved original CSV is authoritative; this table is a reading aid. MSL's graph benefit is limited: the corrected event contrast against removal is +0.05380 with descriptive 95% interval [0.00080, 0.11538], but its contrast against LSTM includes zero, and point F1 is lower than LSTM. SMAP graph-minus-removal is −0.08069, interval [−0.14384, −0.02601]. A better forecasting average is insufficient evidence of better anomaly discrimination.

There is one telemetry target and command-module features in each supplied stream. Different entity streams are not time-aligned. A graph joining separately collected telemetry files would invent sensor relations. The new graph instead learns directed **within-file command → telemetry messages**. These are statistical associations over anonymized command-module slots, not physical causal topology. Mission-specific shared slots follow the encoding described by Hundman; actual module names are unavailable. Entity embeddings distinguish target channels.

## Evidence and exposure limits

NASA TEST labels, TEST results and the final TRAIN portions have been used in earlier research. Architecture choices in this release are explicitly informed by that history. The new TEST comparison is **exploratory retrospective** evidence, never an independent confirmation, irrespective of bootstrap intervals. The last 10% of TRAIN is quarantined only within EXP013; it is not previously unseen real spacecraft data.

The [official NASA/telemanom README](https://github.com/khundman/telemanom) states that released telemetry was pre-scaled using TEST minima/maxima. No raw unscaled replacement is available in the recovered project. New transformations use FIT only, but cannot remove this inherited preprocessing or certify raw-data leakage freedom. The numerical scale floor below is in supplied normalized units, not physical sensor resolution. All new comparisons share the same supplied arrays; the limitation must accompany any publication.

The benchmark consists of anonymized, event-centred snippets rather than continuous full-mission operations. False-alarm occupancy and starts per 1,000 nominal samples are benchmark quantities, not validated operational rates per hour. The canonical anomaly-enriched roster and expert interval labels do not establish performance over all spacecraft channels or unlabelled failure types.

The NASA annotation CSV and historical event-interval reports were not parsed while building this release; completed EXP012 performance tables were inspected and reused. TEST NPY bytes were copied unchanged without loading their float arrays. The preserved historical report archive contains label-derived outcomes and event metadata, so the bundle is **not physically free of historical label information**. New training, detector selection and scoring ignore those archived event intervals. The later gate controls parsing of the original annotation CSV for the new evaluation; it cannot undo prior exposure. No EXP012 fit or model replay is required.

## Data and chronology

The frozen roster is 81 streams: 27 MSL and 54 SMAP. Preserve the first P-2 label row and all 104 original inclusive intervals, converted to half-open only for evaluation. Scores are evaluated on the original EXP012 common target support beginning at index 250.

For each TRAIN file, floor chronological boundaries at 60%, 80% and 90%:

| Segment | Purpose | Permitted selection |
|---|---|---|
| 0–60% FIT | New model fitting and synthetic training faults | Parameter fitting |
| 60–80% STOP | Checkpoint/early stopping | Synthetic balanced loss for new models; native reconstruction/forecast loss for published models |
| 80–90% CAL/SELECT | Clean nominal threshold calibration and fixed synthetic detector selection | Six prespecified candidates; seed 17 only selects the procedure |
| 90–100% CONF | Frozen mechanism and nominal-alarm assessment | No fitting, retuning or winner selection |
| NASA TEST | One new frozen score export and retrospective comparison | Evaluation only; labels requested after the score seal verifies |

Windows remain within one file. Context before a split boundary may be observed; no target beyond the segment is used. No windows concatenate files, and no graph joins entities. FIT telemetry statistics are float64 per entity, standard deviation floored at 0.001; commands remain unchanged. There are 20 channels below this floor in the recovered FIT data. Published models additionally apply a pooled FIT-only StandardScaler over all dimensions. Neither new TEST moments nor TEST normalization is fitted.

## New anomaly detector and controls

One shared model per mission, seeds **17, 29, 43**. Channel-balanced window sampling with at most 2,048 evenly spaced eligible FIT windows/channel/epoch; at most 32 STOP windows/channel. Batch 256, Adam 0.001, maximum 25 epochs, patience 5. Five causal convolutions, width 32, dilations 1/2/4/8/16, receptive field 63; channel embedding 8.

The graph computes a directed top-four command attention message with an optional null message. Command eligibility is determined separately per entity from variation in FIT only, stored in a model buffer. Both graph and fusion zero inactive command slots; inactive slots cannot enter graph top-k, and a null message handles no varying commands. Commands are available only through t−1. A normal telemetry reference uses commands and channel identity, **no telemetry**, preventing the reference itself from following a sustained faulty telemetry level. The discriminator sees the observed telemetry history through t, the command message, channel identity and eight residual/dynamics summaries. It may ignore the graph; the controls test that possibility.

| Arm | Difference | Interpretation |
|---|---|---|
| `graph` | Learned sparse command message + null | Proposed command-reference GCN |
| `removed` | Every command message path zeroed; identical parameter initialization/state layout | Exact command-message ablation; effective trainable capacity is smaller |
| `fusion` | Direct flatten-and-linear command fusion | Strong non-graph context control; its larger parameter count is disclosed |

Each fit minimizes clean-reference Huber loss plus balanced binary cross entropy for clean and injected windows. Checkpoint selection uses balanced synthetic BCE + 0.1 clean Huber, not forecasting MSE alone. Synthetic training includes spike, offset, ramp, stuck and contextual telemetry swaps from another FIT batch window while keeping commands unchanged. Signed amplitudes are uniform 0.5–4 FIT standard deviations, durations 1/8/32. Unchanged stuck/swap samples receive the normal label. This is inspired by outlier exposure, not a faithful NCAD reproduction.

An observed value at t is permitted when issuing an anomaly score at t. This is causal anomaly monitoring, not a t−1 forecast of the future fault. No current command enters the new reference. The one-hop graph is deliberately small; no learned cross-entity adjacency, VAE ensemble or architecture sweep is added.

## Published comparison panel

There are **18 new-control fits + 4 published fits = 22 fits**. GDN and CATCH each use one shared model/mission, seed 17. Individual within-entity windows are pooled; streams are not concatenated at boundaries. Native published architectures/losses are used, but these are **controlled NASA adaptations**, not reproductions of published leaderboard numbers. The common split, pooled training, causal support and TRAIN-only thresholds are protocol changes. Published models have one initialization; their seed uncertainty is not measured.

**GDN:** [official source](https://github.com/d-ailin/GDN), commit `9853899da860682669a134e4af315d036aab4eca`; ZIP SHA-256 `8e4fa40b492f07a2d010fa26776cd2e78f318931dfa2cc63092d8ba10a848e9e`. Official run.sh settings: dimension 64, lookback 5, top-k 5, batch 32, maximum 30 epochs, patience 15, learning rate 0.001. Preserve original graph equations and forecast loss. Two PyG API edits explicitly retain the old equations: third softmax argument becomes `num_nodes=`, and the old node axis becomes `node_dim=0`. An independent dense forward/gradient oracle checks these edits. The released error-score helper fits median/IQR on TEST; this runner does **not** call it. Use per-entity CAL error median/IQR plus original 0.01 denominator, causal four-point mean and max-dimension error, then the common detector. This scoring normalization adaptation is disclosed.

**CATCH:** [official source](https://github.com/decisionintelligence/CATCH), commit `3647c69be5eb56649b072596cf89098e689e20c3`; ZIP SHA-256 `d464a528bd312ec747dc2ad45c5ee7d5313f0d542eb778478f0fa443392be4c5`. Use the exact pinned MSL/SMAP detect_score script overrides: sequence 192, batch 128, 5 MSL/10 SMAP epochs and native mission model dimensions. Preserve forward, RevIN, frequency reconstruction objective, mask generator and time+frequency scoring. Preserve the official unusual mask-optimizer ordering, carried mask gradients and learning-rate schedule. Short-loader modulo-zero is guarded by a minimum mask interval of one; this does not affect the normal pooled NASA loader. Native stochastic Gumbel masks remain; deterministically seed each channel/case, and use the same mask seed for clean/injected pairs. Emit the native **right-edge** window score at t, avoiding future context used by offline block scoring. That causal adapter is an evaluation change. MSL has 210,881,152 parameters; SMAP 53,464,208. Parameter count and GPU memory/timing are reported. D-12 has only 187 FIT points and contributes zero native 192-point CATCH training windows. The shared MSL model is learned from other entities and applied to D-12; no shorter sequence or per-entity substitute is introduced. Every job reports per-channel FIT/STOP window counts and zero-window channels. Actual full-batch GPU validation must pass before fitting; no smaller architecture or batch is substituted on failure.

Source archives are fetched by immutable commit and content hash, about 4 MB total; cached files are checked against the original archives and exact allowed edits on every reuse. No unpublished CATCH source is redistributed in the bundle. See the upstream repositories for licenses/terms before redistributing their code.

**STGLR:** the [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11769452/) is reviewed, but no official source implementation was located. Graph/variational/decoder/attention and detector details do not justify a claimed faithful implementation from the available description; its point-adjusted reported numbers are also incompatible with our metrics. Record a fidelity gap, not an invented baseline.

**OmniAnomaly:** the [official source](https://github.com/NetManAIOps/OmniAnomaly) depends on a legacy Python 3.5/3.6 TensorFlow/TFSnippet environment. A modern Torch approximation would not be faithful. CATCH supplies the agreed CATCH/OmniAnomaly alternative. STGLR and OmniAnomaly are not reported as successfully reproduced.

**Historical LSTM:** reuse the original controlled Hundman-derived EXP012 LSTM reports, without training or inference replay. Common entity/support definitions and unchanged metric implementation permit a complete-pipeline contrast. Its original adaptive detector and batched issue timing differ from EXP013; the contrast cannot isolate architecture. The full original EXP012 table also retains GCN, removal, TCN and persistence for context.

## Detector, selection and freeze

Exactly six candidates: causal EWMA span {1, 8} × clean-CAL higher quantile {0.99, 0.995, 0.999}; strict `score > threshold`. Per-entity/model/seed thresholds; no TEST adaptive threshold, score normalization, merging, interval expansion, point adjustment or best-TEST threshold. The nominal calibration gate is mean channel occupancy ≤ 1% and mean alarm starts ≤ 1 per 1,000 samples. Occupancy on assumed-normal TRAIN is not verified real false-positive rate.

Seed 17 selects each role/mission procedure by equal-family mean of 0.5 incremental event recall + 0.5 strict point F0.5. Selection faults: spike length 1; offset/ramp/stuck lengths 16/64, amplitudes 1.5/3. Other seeds retain the selected span/quantile and calibrate their own clean thresholds. Ties prefer lower nominal occupancy, shorter smoothing and higher quantile. If no candidate passes, freeze q=0.999/span=1 **diagnostically** and record `NO_FEASIBLE_CANDIDATE`; no certification or NASA-driven rescue.

All selected weights, data identities, source bytes, environment, decisions and thresholds are sealed before CONF and before opening new NASA TEST float arrays. CONF faults use amplitudes 2/4, spike length 1 and other lengths 48/96; add held-out variance and oscillation families. Amplitudes lie within the training amplitude range and are not claimed to be unseen fault magnitudes. Requested length is bounded by `min(requested, max(1, (segment_length−16)//2))`. Report requested/effective lengths and unchanged invalid cases. Fault location is deterministic near the middle of the reserved segment; this limited placement is not exhaustive real fault coverage.

Background overlap, unconditional injected hits, hits conditioned on a clean background window and incremental hits are separate. An existing normal-window alarm cannot count as an incremental detection. Repeated deterministic cases are not independent real spacecraft events.

Prespecified held-out-family graph-minus-removal/fusion comparisons use paired channel bootstraps, seed-grouped equal-family objectives, 10,000 draws, seed 20261006 and 98.75% descriptive intervals for four contrasts across both missions. Synthetic support requires all four lower bounds >0 and CAL/CONF nominal gates for all three new arms. Its conclusion applies only to injected mechanisms, under shared-model/channel dependence.

## NASA evaluation and interpretation

Only after complete score-seal verification may the notebook request the NASA annotation CSV, exact SHA-256 `057ce2d6c8875982bf4e5404aefea14efdcbce413d80826d2b737c95b59b7539`. This CSV is excluded from the bundle; archived label-derived reports are retained as declared historical evidence. All eligible jobs must export the whole entity roster; with all published models finite, there are **891** score streams (11 models × 81 entities). Published numerical failures are explicit records or pre-freeze exclusions, never replaced by invented scores. Integrity, OOM and source errors abort rather than becoming numerical exclusions.

Use the **byte-identical EXP012 metric module**: corrected event precision/recall/F0.5, strict point precision/recall/F0.5/F1, tied-score per-channel AP with undefined-support counts, nominal occupancy, alarm starts per 1,000 nominal points and actual event notification timing. No point adjustment. Original inclusive events and original support are preserved. New issue time is t; historical notification delays remain as recorded. Missing or nonfinite published coverage prevents a full-roster superiority claim.

Four graph-minus-removal/fusion NASA corrected event F0.5 contrasts use 98.75% descriptive channel-bootstrap intervals. LSTM/GDN/CATCH contrasts use exploratory 95% intervals; published comparisons do not estimate between-fit variability. Channel bootstrap resamples channel groups and keeps all seeds together, but cannot resolve shared model/command dependence or historical TEST adaptation. These are not independent significance tests.

Point precision, recall, F0.5/F1 and nominal occupancy differences also receive exploratory 95% descriptive intervals from the same paired channel draws. These additional metric intervals are not multiplicity-adjusted and do not authorize confirmatory significance claims or TEST-driven metric selection. Undefined draws and coverage are retained.

`DECISION.json` reports synthetic gates and whether graph point F0.5, point F1 and corrected event F0.5 all exceed each control **while both alarm-burden measures do not increase**. This is a prespecified descriptive outcome, not model selection. Do not promote an event-only improvement while point discrimination deteriorates. Both missions must improve to support the research objective; inconsistent or negative results remain final for V1. No post-result hyperparameter or seed changes are permitted. Independent real-anomaly data and operational false-alarm validation remain necessary for generalization/deployment/SOTA claims.

## Runtime, resume and integrity

Scientific fitting requires CUDA and a successful full-batch GPU gate; CPU is reserved for staging, engineering fixtures and label evaluation. Recommended runtime: **A100 40 GB**, high host RAM if available. 80 GB is not a scientific requirement. The full-batch gate on the actual GPU must pass with deterministic FP32, no AMP/TF32/compile. CPU engineering tests passed during build; full A100 batches could not be tested here because this workspace has no GPU. Hardware success is therefore enforced at execution, not asserted in advance. Fixed published dimensions/batches are not silently reduced.

Use `/content` for active files and verification. Checkpoints atomically contain a SHA-protected payload; selected weights, stage manifests and source archives are hashed. Epoch resumes restore optimizer, sampler, all RNG states and CATCH mask gradients. Completed jobs/stages are reused only after verification; changed code, environment or data aborts. Snapshot ZIPs have a complete file manifest. Same exact software/hardware identity is required for scientific fit/score continuation.

An epoch checkpoint survives a kernel restart in the same VM. A downloaded or copied snapshot survives a VM reset. The notebook builds a local resume archive after every completed model and at freeze; it does not automatically make `/content` durable. Download/copy that archive at useful milestones. A lost unfinished job may repeat work since its last durable snapshot; never claim otherwise. A single archive can be copied to Drive, but no active tiny-file reads, writes or seal verification use Drive FUSE.

The final audit checks package/source/data identities, study and score seals, report integrity, full roster, label-access order and exposure declarations. Audit PASS is an engineering/protocol outcome, not proof of scientific improvement.
