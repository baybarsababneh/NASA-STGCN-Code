# SMAP/MSL source and protocol audit

## Authoritative inputs

All seven supplied files were inspected. Byte hashes are recorded in
`audit/source_manifest.json`. The email asks for tuning a spatiotemporal GCN on
these data and reporting performance; it does not prescribe graph topology,
validation method or a target score. No Kaggle/account content was changed.

The supplied paper is Hundman et al., arXiv:1802.04431v3. Relevant sections are
3.1 (prediction and command encoding), 3.2-3.3 (thresholding), 4.1-4.2 (scoring
and settings), and Tables 1-4. Table 2 was also checked in a rendered page.
The public authors' Telemanom code was read and downloaded locally, because the
attached notebooks do not implement the paper faithfully. The downloaded
`vendor/telemanom` code, config, README and Apache licence are retained.

## Raw data and labels

The ZIP contains 412 files: 82 training arrays, 82 test arrays, 82 saved model
files, 82 forecast arrays, 82 smoothed-error arrays, a run log and a label CSV.
The archive CSV is byte-identical to the separately supplied label CSV.
All labelled arrays are finite. Each row is a time step; column 0 is a single
continuous telemetry target; all remaining columns are binary command-context
indicators. The paper describes module and send/receive command encoding.
The public data do not supply a physical connectivity map or identifying timing
metadata. Simultaneous activity can occur in multiple command columns.

The CSV contains 82 rows, but only **81 unique labelled channel IDs**:

| Policy | SMAP streams | MSL streams | SMAP train rows | MSL train rows | SMAP test rows | MSL test rows |
|---|---:|---:|---:|---:|---:|---:|
| Original label rows, counting duplicate | 55 | 27 | 140825 | 58317 | 444035 | 73729 |
| Unique labelled files | 54 | 27 | 138004 | 58317 | 435826 | 73729 |

`P-2` appears twice, with [5350,6575] and [5300,6420]. We preserve both source
rows and use their union [5300,6575] for the main unique-stream evaluation.
This gives 68 SMAP and 36 MSL labelled events. First-row and last-row policies
are evaluated separately as sensitivity analyses. The unlabelled file `T-10`
has 425 training and 670 test rows, with 55 inputs; it is excluded rather than
silently treated as normal.

The paper's 429735 SMAP and 66709 MSL evaluated values are reproduced exactly
by subtracting 260 rows from each of the 82 CSV rows, **including the duplicate**.
Our unique-file common comparison covers 421786 SMAP and 66709 MSL points.
All label indices are valid and inclusive. Very short training streams exist:
`D-12` has only 312 training rows. Many command columns are constant on training.

The upstream README states that targets were originally scaled using test-set
extrema. This is a provenance warning rather than a verified uniform policy:
11 MSL test arrays actually exceed [-1,1], including M-6 with maximum 258.108.
All SMAP test targets lie within the nominal range. `audit/test_ranges.csv`
records the observed extrema; preprocessing cannot be reconstructed conclusively
without original units and scalers. The release may contain test-dependent scaling.
For new model fitting we re-standardize the target using only the first 60% of
training. For nonconstant data this cancels a prior positive affine min-max
scaling algebraically. No new test-dependent scaling is fitted. Constant
training targets use scale 1; that exception and the raw-unit thresholds in the
compatibility track mean we do not claim the released benchmark is fully free
of upstream test dependence. Original physical units cannot be reconstructed.

## Attached notebook assessment

| Notebook | Useful material | Why its outputs are not the paper benchmark |
|---|---|---|
| `smap-msl-let-s-get-started` | Paths, CSV inspection | Only five exploratory cells; no model or evaluation |
| `lstmcode`, first cell | PyTorch/Keras workflow ideas | Stacks unrelated 55-input files; windows cross file boundaries; discards real labels; injects an artificial anomaly; reconstruction across all input columns; includes unlabelled T-10 |
| `lstmcode`, second cell | Transformer example | Same concatenation and synthetic-label problems; explicitly maximizes F1 over test labels; this is an oracle threshold, not held-out performance |
| `nasa-anomaly-unsupervised` | Per-file loading, train-fitted scaler | Single P-1 autoencoder; averages error over commands and time; fits threshold to whole test errors; no quantitative real-label evaluation; attaches window [i,i+99] to i+100 rather than its last observed point |

The 98-99% accuracies shown in the first notebook primarily reflect class
imbalance in a different, synthetic task. They are not reused as evidence.
Whole-test unsupervised thresholding is distinct from test-label leakage, but
it is retrospective and cannot be described as a fixed training-only threshold.

## Original LSTM and evaluation

Paper: one model per telemetry stream, two LSTM layers of 80 units, dropout .3,
lookback 250, Adam, MSE, batch 64, at most 35 epochs with early stopping.
The paper describes one-step prediction. The saved models and config instead
have **10 output horizons**. Upstream shaping omits the final possible window,
producing N-250-10 outputs, and prediction aggregation uses the first horizon.
The saved model uses legacy Keras hard-sigmoid gates, not modern PyTorch sigmoid
gates. We independently ported inference for eight windows each in four channels;
`audit/archive_weight_verification.json` records agreement with saved predictions.

The code shuffles overlapping training windows before the Keras validation split;
that is not a chronological validation holdout. We do not repeat this practice.
The archive provides test forecasts and weights, but no trustworthy independent
calibration predictions; its main comparison therefore uses its own detector.

Original error processing uses absolute target prediction error, EWMA span 105,
70-point batches, nominal 2100-point history, threshold search and pruning p=.13,
and expansion by 100 points. The current upstream code additionally checks
inverted errors, searches z from 2.5 through 11.5 (paper specifies through 10),
and imposes absolute/relative error guardrails. Initial error values are replaced
by an average using future errors, with a hard-coded exception for C-2. Its first
history window is evaluated retrospectively. These details are retained only
in the explicitly labelled `telemanom_compatibility` analysis.

The code and paper also differ in interval scoring: the paper credits each
labelled interval hit by any prediction; the code credits only the first label
hit by a predicted interval. We use the paper's stated definition in new results
and preserve a literal-code replay as an audit. Current upstream code includes
the documented +250 correction from error-vector to raw-array coordinates;
historical logged results can consequently differ. No coordinate shift is
chosen by maximizing score.

**Event precision** = hit labelled events / (hit labelled events + unmatched
predicted intervals). **Event recall** = hit labelled events / all labelled
events. This convention can reward extremely broad predictions and ignores
duplicate alarms within a label. Therefore we also report unadjusted point
precision/recall/F1/F0.5, alarm coverage, per-channel AP, false events per 1000
steps and detected-event delay. No point adjustment is used.

For beta=.5 we use F = 1.25 TP / (1.25 TP + FP + .25 FN).
Table 2 prints SMAP P=85.5%, R=85.5%, F0.5=.71, which cannot satisfy the
standard definition; its recalculated F0.5 is .855. MSL P=92.6%, R=69.4%
implies approximately .868, rather than printed .69. Both printed and
recalculated values are retained in the report, without guessing the cause.
The archived log itself has MSL TP=25, FP=1, FN=11 (precision .962), differing
from the paper's MSL precision .926. A replay is not labelled an exact paper
reproduction merely because it reuses archived predictions.

## Introduced graph and model

A cross-file sensor graph would require verified alignment. We instead use a
separate typed graph within each stream: one telemetry node and one node for
each nonconstant **command indicator** in the fit segment. They are different
variable types, not interchangeable sensors or proven physical subsystems.
Directed command-to-target edges encode a modelling hypothesis that past
command context can help forecast telemetry. No causal or physical discovery
is claimed. Constant command indicators are removed using fit data only.

A shared length-L temporal convolution embeds each node's past; per-node gains
and biases retain indicator identity. A relational graph convolution has a
separate self transformation and mean-aggregated incoming-command transformation,
followed by a telemetry-only readout. A persistence residual connection predicts
the next value. This is a compact temporal-then-graph model, not a reproduction
of a particular published traffic STGCN architecture. The lag-axis linear map
is exactly a valid temporal convolution with kernel width L at the forecast time.

Focused configurations vary lookback 32/64/128, width 16/32 and all-command
star versus four fit-only lag-associated command neighbours. Top-four ranking
uses absolute correlations of past command activity with target changes at
lags 1,2,4,8,16. Its edges are associations, not causal links. Controls include
persistence, a flat lag MLP, a compact LSTM and an exactly matched GCN with
command edges removed. The latter is essential to test whether the graph adds
anything beyond its temporal component.

## Clean experiment protocol

Within each official training stream: 60% fit, 20% chronological validation
for early stopping and graph configuration selection, 20% untouched calibration.
Validation/calibration windows may use preceding observed context, but targets
are disjoint and no future observation is an input. Streams are never joined.
Model weights, active nodes and top-four graph statistics use only fit data.
The best checkpoint is restored. Graph selection minimizes median per-channel
standardized validation MSE separately by mission; anomaly labels are never
used to choose models, graphs, thresholds or seeds. Forecast validation is a
proxy, not a guarantee of good anomaly detection.

All neural controls receive the same fitting/validation budget. Seeds 17,29,43
are used for the selected graph and its matched edge-free control. Seed 17 is
the declared primary run, not the best seed on test performance. AdamW uses
lr=.001, weight decay=.0001, batch 256, at most 40 epochs, patience 6 and
gradient clipping at norm 1. Configuration is frozen before new test evaluation.

The primary detector uses causal EWMA span 10 and the 99.5th percentile of
calibration errors, with a numerical floor. Short smoothing startup is removed
from calibration. Test smoothing starts at the common evaluated index 250;
the last 10 points are excluded for an identical horizon with archived LSTM.
Predetermined q=.99/.999 and span=30 sensitivities are reported without
selecting a winner. No test-derived threshold is promoted to the primary result.
The percentile is an operational choice, not a calibrated false-alarm guarantee:
serial dependence and short calibration sets, especially D-12, are limitations.

## External provenance

- Authors' repository: https://github.com/khundman/telemanom
- Dataset description supplied via notebooks: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl
- Paper: https://arxiv.org/abs/1802.04431

The downloaded repository snapshot is retained byte-for-byte and hashed locally;
it was read from the master branch, so it is not assumed identical to 2018 code.

## Completed run and interpretation

The completed experiment contains 648 initial fits (five GCN configurations and
three controls across 81 streams) plus 405 repeats/matched ablations: 1053 fits.
There are 810 retained test prediction files and 3402 per-channel metric rows.
1053 fits do not mean 1053 hyperparameter trials: only five graph settings were
compared, with the remainder coming from per-stream models, controls and seeds.

The chosen graph does not improve primary event F0.5 over its matched edge-free
control in either mission, across the three tested seeds. Its SMAP point F1
exceeds the compact LSTM's, but the event precision is very low. Thresholding
and alarm fragmentation dominate the primary event result. For example, GCN
seed 17 gives A-7 304 unmatched intervals despite point precision .762; A-5
has 220 unmatched intervals and point precision .014, demonstrating both alarm
fragmentation and substantial false-positive coverage. The compatibility
detector increases event precision through smoothing, pruning and buffering.
That is a separate protocol, not evidence that the fixed-calibration result
has improved.

About 22.7% of initial fits selected epoch 40, the imposed maximum; thus the
training budget may constrain some configurations. No extension or parameter
change was chosen after seeing test outcomes. Constant-target streams and the
persistence residual warrant an absolute-level forecasting follow-up under a
new predeclared evaluation design. The present outcome is an initial negative
or mixed result about this specific graph/model/detector combination, not a
general result that GCNs cannot work on spacecraft telemetry.
