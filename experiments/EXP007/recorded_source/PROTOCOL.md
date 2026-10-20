# EXP007_GPU_V1: conditional total-message gating

## Question and scope

Does one small, time-dependent scalar gate improve the final EXP006 forecasting
architecture by modulating the total command message? Test both missions. MSL's
failure to benefit from a capacity/RF search does not test this distinct hypothesis.
SMAP's successful RF31 baseline is retained. No anomaly outcome motivates channel,
seed, architecture, threshold or stopping choices.

The uploaded completed EXP006 summary was audited: all 1,209 fit rows are unique,
match the frozen job plans and execution hash, reproduce the screening/promotion
decisions, and use source/data hashes matching the supplied package. The upload
does not contain checkpoint archives, so this is a summary/source audit rather
than an independent recheck of every checkpoint byte. Original experiments and
the locked retrospective NASA anomaly benchmark remain unchanged.

## Baselines locked from EXP006

| Mission | Temporal RF | Dilations | Hidden/message width | Learning rate |
|---|---:|---|---:|---:|
| MSL: retained EXP002 architecture | 7 | 1, 2 | 32 | 0.001 |
| SMAP: selected g09 | 31 | 1, 2, 4, 8 | 64 | 0.001 |

Lookback 64, kernel 3, one fixed graph layer, original node retention and graph
support, fit-only scaling and command selection, no new normalisation/dropout,
AdamW weight decay 0.0001, batch 256, clipping norm 1, maximum 80 epochs, patience 6,
min_delta 1e-7. Preserve the completed EXP006 trainer's min_delta checkpoint
selection behavior exactly. No epoch extension or learning-rate adjustment.

All roles are fresh from scratch on the same current GPU execution path. Prior
EXP006 scores are provenance, not the numerical comparator for EXP007 promotion.
Fresh reference fits avoid dependence on unavailable prior checkpoints and any
environment change. This does not replace or rerun the earlier research studies.

## Exact gate

Let z_i(t) be the existing node-specific, rectified causal temporal representation,
a_0(t) = sum_j A[0,j] z_j(t) the existing equally normalised command aggregate,
and m_0(t) = W_message a_0(t). The telemetry update becomes

    g(t) = 2 sigmoid(w^T [z_0(t); a_0(t)] + b)
    h_0(t) = ReLU(W_self z_0(t) + b_self + g(t) m_0(t))
    prediction(t) = previous_telemetry(t) + head(h_0(t)).

The gate is one scalar per prediction time, shared across message dimensions.
Its inputs contain only the existing causal history ending at t-1. This is a
conditional multiplier, not a learned per-timestamp variable, new temporal
encoder, attention over edges, or graph-support change. All edge ratios remain
fixed. Gate range is (0,2), allowing suppression and bounded amplification.
Set w=0 and b=0, giving g=1 exactly. The persistence-residual head retains its
original zero initialisation. There is no gate MLP or separate gate learning rate.
Gate parameters receive the same AdamW decay as the rest of the model.

Additional dynamic parameters: 2h+1 = 65 for MSL and 129 for SMAP. Direct zero
parameter creation consumes no RNG draws. All roles have identical backbone
initial weights for each channel/seed. RF7 is checked against the original
EXP002 implementation after mapping its convolution names.

Three fixed roles, no hyperparameter search:

1. **fixed:** g=1, the selected EXP006 architecture.
2. **static:** g=2 sigmoid(b), b=0 initially, one learned scalar per fitted model.
   This is an explanatory control for global message rescaling. EXP004's
   normalised edge softmax could not change total incoming weight this way.
3. **dynamic:** the sole architecture candidate defined above.

The static scalar is not a second promotion candidate. Its purpose is to prevent
crediting conditional gating for a benefit obtainable by a constant multiplier.
If static dominates, report that finding and reject the dynamic extension rather
than launch a new search. Dynamic versus static is not parameter-count matched.
This experiment evaluates the conditional parameterisation as a whole, not a
capacity-independent causal effect of time dependence.

## Stages and exact budget

There is no separate architecture screening stage: only one candidate is tested.
Use the original mechanical 12-channel roster for earlier-origin robustness:
MSL C-1, P-10, M-7, P-14, T-13, D-16; SMAP P-3, A-8, G-6, E-1, G-4, A-5.

| Stage | Specification | Fits |
|---|---|---:|
| Earlier-origin robustness | 3 roles x 12 channels x seeds 17/29 x 2 folds | 144 |
| Canonical confirmation | 3 roles x 81 channels x seeds 17/29/43 | 729 |
| Final matched message removal | selected architecture x 81 channels x 3 seeds | 243 |
| Total research budget | no extra candidates or seed selection | 1,116 |

R1: fit [0,40%), early stop [40,50%), assessment [50,60%).
R2: fit [0,50%), early stop [50,60%), assessment [60,70%).
Canonical: fit [0,60%), validation/early stop [60,80%), calibration [80,100%).
Use floor boundaries and the same past-window convention as EXP006. Recompute
scalers and graph support from each fitting block. Assessment targets never
guide that fit's early stopping. Calibration is never scored for this study.
The original canonical loader constructs calibration windows as before, but
these do not enter training, diagnostics or decisions.

Complete both missions and all prespecified roles regardless of intermediate
performance. There is no performance-based screening stop, dropped channel or
adaptive fit budget. Stop on engineering failure and preserve completed work.
Do not change batch size after OOM or substitute a seed to complete the table.

These TRAIN validation/assessment intervals have already supported earlier project
development, including EXP006. "Confirmation" names an internal stage, not a new
independent holdout. Their repeated use limits generalisation claims, even though
NASA TEST is excluded and each individual fitting/early-stopping split is causal.

## Frozen mission-specific promotion rule

Mean scores weight channels equally and average all three canonical seeds.
Promote dynamic gating for a mission only if every condition holds:

1. Its full-channel mean validation MSE is at least 2% below the fresh fixed baseline.
2. Its mission mean MSE is strictly lower than fixed for each of 17, 29 and 43.
3. Its mean R1/R2 relative mission improvement versus fixed is nonnegative.
4. Its canonical mission mean MSE is strictly lower than the static scalar control.
5. Its mean R1/R2 relative mission improvement versus static is nonnegative.

For condition 3 or 5 use arithmetic mean over folds of
100*(reference_fold_mean - dynamic_fold_mean)/reference_fold_mean, using both
robustness seeds and equal channel weights within a fold. A zero/nonfinite
reference denominator is an audit error, not a revised metric.
An exact canonical tie with static rejects the dynamic extension.

Conditions 1–3 carry forward EXP006's practical development standard. Conditions
4–5 justify the extra conditional parameters against the explanatory scalar
control. These are development gates, not significance tests. Bootstrap intervals
are diagnostics and never introduce additional promotion or stopping criteria.
Failure of any gate retains the EXP006-selected fixed architecture for that mission.

Write and hash ARCHITECTURE_FREEZE.json immediately after these 873 fits, before
starting the final matched control. Never choose a best seed or a per-channel model.

## Final matched message-removal control

After the architecture freeze, refit the selected mission architecture with the
command aggregate zeroed BEFORE both the message transformation and any gate
input. Keep telemetry self path, temporal backbone, head, widths, graph metadata,
initialisation, nominal parameter set, learning settings and all seeds unchanged.
Shared backbone weights are initialised identically to the corresponding selected
canonical fit. The selected canonical checkpoints already provide the graph side
of each pair, so another 243 winner refits are unnecessary.

This removes the entire command-message pathway without a gate-based command
bypass. Parameters exclusive to the disabled pathway are inactive; equal nominal
parameter count does not mean equal active capacity. Label this a matched pathway
ablation, not an independently tuned non-graph competitor. EXP005 remains the
independently tuned command-aware competitor from its own completed study.

The ablation explains the final model and cannot reopen selection. If removal
forecasts better, report that result without forcing a graph-benefit claim.
This does not measure anomaly-detection performance.

## Diagnostics and evidence

Retain all fit scores, actual parameter counts, gate counts, epochs, stopping and
ceiling indicators, training/wall time, peak GPU memory, full epoch curves, best
checkpoints, fit-only scalers, feature/edge lists and gate activation traces.
Gate summaries use canonical validation or robustness assessment only. Gate
mean/std, 5th/95th percentiles and fractions below 0.1/above 1.9 are explanatory,
not promotion criteria. With zero incoming edges the gate is unidentifiable.
Do not interpret latent-space gate coefficients as physical command importance.

Report mission and seed means, channel and family effects, absolute MSE changes
and guarded percentages (relative percentages undefined below reference MSE
1e-8). No selection based on favourable families or near-zero-error percentages.
For dynamic vs fixed, dynamic vs static, and final removal vs selected model,
average three seeds within each channel and bootstrap paired channels 10,000
times using reporting RNG seed 1007. Report 95% percentile intervals of mean
absolute MSE improvement. Seeds are not 243 independent samples. These intervals
are conditional development diagnostics; related channels and overlapping rolling
origins limit independence. No multiplicity-adjusted SOTA claims or new holdout
claims follow from these tables.

## Execution contract and data exclusion

The ready-made bundle contains only the same 81 official TRAIN arrays and hashes,
with T-10 excluded by the prior study roster. It contains no NASA TEST arrays,
anomaly-label files, TEST residuals or anomaly metric tables. The running code has
no data-download function. An additional file-access guard rejects out-of-bundle
NPY and anomaly-label access. No anomaly evaluator is included.

GPU-native from the first research fit. Prefer an A100 in Colab. Another CUDA GPU
may start the study, but its model/software fingerprint is then locked for every
stage and resume. Use one active notebook and one worker. No CPU fallback, AMP,
TF32, torch.compile, or performance-dependent hardware selection.

The input includes the exact completed EXP006 dependency version list, including
PyTorch 2.11.0+cu130, NumPy 2.1.3 and pandas 2.2.3. Install into a separate virtual
environment. Record and lock the actual Python patch version, packages, driver,
CUDA/cuDNN, GPU model/capability/memory, build information, code/config/data hashes
and flags before fits. The first EXP007 environment need not reproduce EXP006
trajectories because all controls are refitted. Resume may not silently change it.
GPU UUID is provenance, not a requirement to obtain the same physical unit.

Use deterministic algorithms, deterministic cuDNN, benchmark off, CUBLAS
:4096:8, float32, explicit AdamW foreach=False/fused=False and single CPU threading.
Preflight verifies data/scaler/graph equality to completed EXP006, gate identity,
causality/RF, exact command-removal invariance with nonzero gate weights, gradient
flow, zero-edge behavior, legacy RF7 equivalence, checkpoint reload and exact
within-GPU repeats of tiny synthetic production-loop fits. These are engineering
checks, not extra research candidates. No cross-backend tolerance is relaxed.

## Durability and stop rules

After each fit, save a ZIP containing checkpoint (including gate trace), result,
epoch history and job specification. Copy to Drive, verify SHA256 by read-back,
and write the completion marker last. Resume verifies package/environment, plan,
job/run identity and archive integrity. A CSV alone never proves fit completion.
An interrupted, uncommitted fit restarts from its original seed, not partial
weights. No completed fit is knowingly duplicated. Corrupt committed files stop
for investigation. A single-run heartbeat lease expires ten minutes after a dead
session; do not intentionally run concurrent notebooks on Drive.

The notebook automatically executes every stage, freezes the selection, runs the
matched control, produces summaries/checkpoint indexes, and exports summary/full
ZIPs. Colab termination can still require reconnecting and pressing Run all.
Drive authorisation may recur. There are no keep-alive or resource-limit bypasses.

After this bounded study and its matched control, STOP GCN architecture
optimisation whether gating succeeds or fails. Begin detector development only
after auditing the architecture freeze and outputs. Detector code/planning may
be prepared earlier, but do not optimise a detector against moving forecasters.
Freeze a separate detector protocol before accessing its reserved calibration
block for selection/calibration. No detector training/evaluation starts here.
Final NASA evaluation remains explicitly retrospective, since historical TEST
outcomes have already been observed.

Execution references:
https://docs.pytorch.org/docs/2.11/notes/randomness.html
https://research.google.com/colaboratory/faq.html
