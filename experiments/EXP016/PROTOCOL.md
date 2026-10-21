# EXP016: author-release methodology, ST-GCN forecasting substitution

This is a new, retrospective experiment. EXP001–EXP015 are untouched. The
comparison is against the supplied author-released LSTM predictions, verified
against its saved weights by inference only. No LSTM is trained.

## Frozen design and scientific claim

One independent command-aware ST-GCN per supplied channel, seeds 17, 29 and 43.
All 82 channels are fitted and predicted, including unlabelled T-10. Detection
endpoints use 81 unique labelled channels (27 MSL, 54 SMAP), the first supplied
P-2 annotation, and a separate alternative-P-2 sensitivity. No best seed, model,
detector, label variant or threshold is selected using NASA TEST performance.
The benchmark has already informed earlier research. This is not a fresh holdout.

The primary outcome is standard event-overlap F0.5, with strict point F1/F0.5,
precision/recall, nominal occupancy, alarm starts, raw/EWMA within-channel AP,
and notification timing mandatory alongside it. These prevent a high overlap
score alone from establishing useful detection. All seeds are reported. Mean
seed scores pool channel counts within a seed before averaging. The fixed author
fit is not treated as three independent LSTM runs. Paired-channel bootstrap
(10,000 draws) keeps all GCN seeds together and the author fit fixed. Intervals
are descriptive, conditional on these fits, and do not resolve shared-incident
or between-LSTM-fit variability. No superiority/equivalence decision is automated.

## Fidelity decisions

| Item | EXP016 treatment | Comparison limit |
|---|---|---|
| History | Exactly 250 actual prior observations per prediction | Seven temporal layers cover RF255, truncated by causal padding to 250 actual observations |
| Inputs | All 25 SMAP / 55 MSL released dimensions; no command pruning | Graph nodes are one telemetry series plus anonymised commands, not synchronised sensors |
| Objective | Mean squared error over 10 future telemetry outputs; scalar detector uses output 0 | Paper prose says one step; release log, source, saved Dense heads and forecast lengths establish 10-output training |
| Window endpoints | Starts `0 .. N-261`; target index `250 .. N-11` | Preserves author's exclusive-loop endpoint, omitting last 10 first-horizon targets |
| Scaling | Use supplied arrays directly, no additional scaler | Author README states TEST-extrema pre-scaling; unavailable raw units prevent removing this inherited exposure |
| Validation | Last 20% of available window starts; purge 9 crossing training starts | Author shuffled overlapping windows before splitting; this is deliberately not reproduced |
| Optimiser | Adam lr .001, beta .9/.999, epsilon 1e-7, no decay | Broad settings match saved Keras metadata; PyTorch and historical Keras numerical update details are not identical |
| Training | Batch64, all eligible windows each epoch, max35, patience10, min_delta .0003 | Chronological validation differs; original individual seeds/history are unavailable |
| Dropout | .3 after temporal embedding and graph hidden readout | Location is necessarily architecture-dependent |
| Selection | Save/use termination weights, as author code; retain best-validation state diagnostically | Not a best-epoch restoration, not a TEST-selected checkpoint |
| Detector | Pinned telemanom v2 source, identical rules for both forecasters | Primary removes label-informed C-2 exception; native sensitivity retains it |
| Labels/scoring | Inclusive intervals, all overlapping true events counted once | Literal upstream first-matched-event bug exported separately; printed paper F0.5 arithmetic not used as a target |

Primary validation prevents forecast-target leakage: with N training rows,
`W=N-260`, `cut=floor(.8W)`, fitting starts `[0,cut-9)` and validation starts
`[cut,W)`. No fitting target reaches the first validation target. Validation
histories may contain prior observed training targets, as in rolling forecasting.
They overlap in context, so validation errors are dependent; no claim of fully
independent raw-window validation is made. On D-12, N312 gives 32 fitting windows,
9 purged starts and 11 validation windows. First validation target291 exceeds
last fitting target290. This is feasible but statistically weak early-stopping
evidence; disclose the small support. The former 60% FIT prefix would have only
187 rows and cannot sustain a 250-history model. No padding of missing real
observations, concatenation of channels, TEST fitting, or shorter-history
fallback is permitted.

## Architecture and implementation efficiency

This extends EXP002/006's shared causal node encoder, node-specific affine
transform, command-to-telemetry graph message, and persistence-residual readout.
A fixed star includes every command node, with row-normalised weights
A[telemetry,command]=1/(features-1). No learned/test-derived graph, cross-file
alignment, or top-four selection is used. Seven kernel3 temporal convolutions
with dilations 1,2,4,8,16,32,64 and 64 hidden channels cover all 250 inputs.
Same-width temporal layers have residual skips for optimisation stability.
There is no batch normalisation, scaling fit, mixed precision, gradient clipping,
window subsampling or reduced minibatch. A linear 10-output residual head starts
at persistence. Architecture capacity and inductive bias are not matched to the
LSTM, which is the intended changed component. The full-history extension is
prespecified, not claimed to be empirically optimal or newly anomaly-tuned.

Production computes only the exact temporal ancestors of the final readout.
The full dense causal-convolution implementation remains available and is tested
against the optimised forward, parameter gradients and input gradients. No
observed timestep or feature is discarded. This is an implementation optimisation,
not an approximation. Full batches run in FP32 with deterministic algorithms,
TF32/AMP disabled. Input windows are indexed on the selected device rather than
materialised for the entire dataset. All training windows are visited each epoch.

## Detector, timing and evaluation

Source pinned to telemanom commit 2e6c5b6c3558e7835601519b7bdef37c649bdbdc.
Compatibility change: cast np.delete row indices to integer. Parameters remain
batch70, trailing30 batches, EWMA span105 (adjust=True), buffer100, p=.13,
threshold grid2.5:0.5:11.5, inverse branch and original 10-target scale guards.
Primary applies the same startup smoothing to C-2 as all other channels.
Zero-SD searches skip a mathematically inert NaN search. If primary pruning
has no nominal reference, that branch explicitly abstains and records a
`degeneracies` receipt. Native sensitivity retains original semantics apart
from integer compatibility and reports failures without substituting a detector
or silently scoring a reduced roster. Primary abstentions must be inspected.

This is a retrospective batched detector, not a strictly causal streaming alarm.
The first error window can use 2100 residuals; startup averaging uses up to500.
Ten-target scale guards see nine observations beyond the first-horizon target.
Issue-time exports charge all this availability, plus buffered retrospective
labels. Detector flags may overlap an event yet become available after its end.
The comparison does not quietly claim timely detection from overlap recall.

Forecast/decision streams and all trained model commits are sealed before the
production evaluator reads labels. Adaptive thresholds use unlabelled TEST
residuals exactly as in the released approach; they are not TRAIN-calibrated.
Original labels already existed in the project and influenced historical method
development. Physical packaging of labels does not establish blindness.

## Sources inspected and reused

- Attached arXiv1802.04431v3, all nine pages; supplied source notebooks and email.
- telemanom source/config/history and release archive model metadata/logs.
- NASA-STGCN-Code commit fc9554c6463cf882acfdd90b001e0a248680aa68:
  experiment inventory, model/graph implementations, protocols and outcomes
  across EXP001–015, and released-LSTM/native-detector followup source/audits.
- Existing followup's inference-only NumPy HDF5 replay function adapted here.
- Existing trained GCN checkpoints are not required: their shorter receptive
  fields/features/objectives do not implement this new protocol. Historical
  missing EXP013 checkpoints therefore do not block EXP016.

The existing audits establish why forecast MSE alone, the narrow project LSTM,
or detector substitution alone cannot answer this question. EXP016 changes the
forecasting architecture and approaches released experimental conditions while
explicitly retaining the validation/preprocessing/fit-variability limitations.

## Reproducibility and continuation

The input archive and extracted files are SHA256-bound. The package manifest,
config, exact environment and hardware name are frozen before training. Changed
identities fail clearly; do not edit config or install different versions to
resume a scientific directory. Use the same GPU type and dependency environment.
Epoch and periodic minibatch checkpoints retain model, Adam, early-stop state,
window permutation/cursor, accumulated epoch loss and Python/NumPy/Torch/CUDA RNGs.
Each checkpoint is atomic, internally hashed and keeps a previous generation.
Only a completed optimizer step is resumable. An interruption during a step
rolls back to a previous checkpoint and deterministically repeats work.
Bitwise resume is tested on CPU; CUDA is gated in Colab and cross-hardware
bitwise reproducibility is not claimed.

Active work is local; a user-authorised Drive mirror saves changes at least every
five minutes when a training checkpoint is reached, at every completed fit,
and at phase boundaries. Abrupt VM loss can repeat work since the last durable
copy. Checkpoint mirror copies are atomic; commit receipts follow payloads.
Only one Colab session may write a given Drive run folder at once. Do not run
parallel sessions against it. A 5.5-hour session budget pauses safely without
reducing the experiment. Rerun the notebook to continue automatically. Final
and partial ZIPs retain states, source, logs, arrays, metrics and checksums.

Preflight measures full-size forward/backward/Adam steps on actual TRAIN windows
for both feature widths and uses exact full-roster minibatch counts. It reports
11/20/35-epoch training projections plus a 1.5x training safety margin. These
are estimates, excluding detector, export, installation and Drive overhead;
no A100 runtime is invented from CPU measurements. If six hours is unrealistic,
the unchanged design continues across sessions. All interim engineering fits
are discarded and never entered into scientific results.
