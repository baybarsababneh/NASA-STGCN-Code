# EXP006 — Independently Tuned Fixed-Edge GCN

## Purpose
EXP006 closes the tuning-budget gap identified after EXP005 while staying directly aligned with the research team's request to tune a GCN on the SMAP/MSL data.

The scientific question is deliberately narrow:

> Can EXP002's fixed command→telemetry GCN forecast normal telemetry better when its temporal receptive field, capacity and learning rate receive the same bounded independent tuning opportunity given to EXP005?

EXP006 does **not** use NASA TEST arrays, TEST residuals, anomaly labels, prior TEST rankings, false-alarm channels or anomaly sensitivity winners for candidate selection.

## Fixed architecture contract
EXP002's graph hypothesis remains unchanged:
- one graph-message layer;
- command→telemetry direction;
- SMAP: fit-varying command star support;
- MSL: fit-only top-4 incoming commands selected at lags 1,2,4,8,16;
- equal fixed graph normalisation;
- shared causal node temporal encoder;
- node-specific gain/bias;
- persistence + learned residual prediction;
- no dropout, batch normalisation, gate, attention or new residual design.

Only receptive field, hidden width and learning rate are searched.

## Candidate grid
Lookback is fixed at 64. Kernel size is fixed at 3.

- RF 7: dilations [1,2]
- RF 15: dilations [1,2,4]
- RF 31: dilations [1,2,4,8]
- widths: MSL {32,64}; SMAP {64,128}
- learning rate: {1e-3,3e-4}

This produces 12 candidates per mission. Message width is tied to temporal width.

## Stage A — freeze and preflight
Before training:
- freeze this candidate list, screening roster, selection criteria and code hashes;
- verify RF calculations against the implementation;
- verify all 81 TRAIN channels are available;
- verify no TEST or anomaly-label path is used by EXP006 selection code.

## Stage B — screening
Reuse EXP005's exact screening roster:
- MSL: C-1, P-10, M-7, P-14, T-13, D-16
- SMAP: P-3, A-8, G-6, E-1, G-4, A-5
- seeds: 17,29

Run 12 candidates × 12 channels × 2 seeds = **288 fits**.

Rank within each mission by arithmetic mean validation MSE over the six channels and two seeds. Each channel receives equal weight. Select the best two candidates per mission. Exact ties are resolved by fewer mean parameters, then candidate ID.

Split remains:
- [0%,60%): fit
- [60%,80%): validation / early stopping
- [80%,100%): untouched calibration

## Stage C — chronological robustness
Before full confirmation, compare each mission's two shortlisted GCNs against:
- historical EXP002 architecture refit from scratch;
- selected EXP005 direct-fusion TCN refit from scratch.

Use the same screening channels and seeds 17,29 under:
- R1: fit [0,40%), early stop [40,50%), assess [50,60%)
- R2: fit [0,50%), early stop [50,60%), assess [60,70%)

Scaling and command support are recomputed from each fold's fit block. The same resulting incoming command sources feed the GCN support and the direct-fusion comparator. Assessment targets do not contribute to fitting or early stopping. Earlier history may be used causally as input context.

Total: 4 roles × 12 channels × 2 seeds × 2 folds = **192 fits**.

## Stage D — full-channel confirmation
Confirm the two shortlisted candidates per mission on all 81 study channels with seeds 17,29,43 under the original 60/20/20 split.

Total maximum: **486 fits**.

## Promotion rule
Among candidates that pass all criteria, promote the one with the lowest full-channel mission mean validation MSE.

A candidate passes only if:
1. mission mean validation MSE is at least 2% lower than historical EXP002;
2. mission mean validation MSE is lower than EXP002 for each seed 17,29,43;
3. the mean of its R1 and R2 relative mission improvements versus the refit EXP002 reference is non-negative.

The 2% rule is a practical development threshold, not a significance test.

If neither candidate passes for a mission, retain historical EXP002 for that mission and **do not open another replacement search**.

## Training common settings
- AdamW
- weight decay 1e-4
- batch size 256
- max epochs 80
- patience 6
- min delta 1e-7
- gradient clipping norm 1.0
- minimum-validation-MSE checkpoint
- record epoch-ceiling hits

## What happens after EXP006
Per the frozen Astra plan:
1. EXP007: one identity-initialised time-dependent total-message gate.
2. One matched message-removal ablation for the final GCN.
3. Separate detector development.
4. Controlled external baselines.
5. Final locked retrospective NASA evaluation and report to the research team.

No architecture search should continue merely because the GCN does not beat EXP005 or a NASA anomaly result.
