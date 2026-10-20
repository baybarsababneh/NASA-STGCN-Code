# V2 EXP005 — tuned command-aware non-graph TCN

## Scientific question
Does graph aggregation in EXP002 add forecasting value beyond simply giving a strong non-graph temporal model access to the same command histories?

EXP003 remains the matched ablation. EXP005 is a separately tuned competitor.

## Input contract
For every channel, EXP005 receives telemetry column 0 plus exactly the original command columns that send messages into telemetry node 0 in EXP002:
- SMAP: all fit-varying command columns retained by the star support.
- MSL: the fit-only top-4 command columns selected by EXP002.

The model fuses these features directly in the first causal temporal convolution. No graph adjacency or graph message passing is used.

The all-command MSL variant is deliberately deferred so that the primary comparison isolates graph aggregation rather than command-set choice.

## Data contract
Official TRAIN only during development:
- first 60%: fitting, scaling, command selection;
- next 20%: validation, early stopping, hyperparameter selection;
- final 20%: calibration, untouched during EXP005 tuning.

NASA TEST arrays and anomaly labels are not used by EXP005.

## Screening channels
The 12 channels were fixed before candidate results. Within each mission, channels with a constant fit target or zero selected commands were excluded from screening only because they cannot discriminate command-aware candidates. The remaining channels were represented in rank-transformed fit-only space using sequence length, target variability, and selected-command activity; deterministic farthest-point coverage was then used to choose six channels per mission. All 81 channels return in confirmation.

See `configs/v2_exp005_screening_manifest.csv`.

## Candidate space
12 predeclared configurations:
- receptive field: 7, 15, or 31 samples (kernel 3; causal dilations `[1,2]`, `[1,2,4]`, `[1,2,4,8]`);
- capacity: baseline (`MSL=32`, `SMAP=64`) or large (`MSL=64`, `SMAP=128`);
- learning rate: `1e-3` or `3e-4`;
- AdamW weight decay `1e-4`, batch 256, max 80 epochs, patience 6, min-delta `1e-7`, gradient clipping 1.0.

`c01_rf7_base_lr1e3` is the documented temporal default adapted to the EXP005 input contract.

## Staged tuning allocation
Screening:
- 12 candidates × 12 channels × seeds 17 and 29 = 288 fits.
- Rank candidates separately by mission using mean validation MSE.
- Keep best two per mission.

Confirmation:
- best two per mission × all mission channels × seeds 17,29,43 = 486 fits.
- Select one candidate per mission using mean validation MSE across all mission channels and all three seeds.

Total planned fits: 774.

## Reporting
Keep all trial rows, failures, runtime, parameter counts, selected epochs and ceiling hits. The final selected model is compared with EXP002 on channel-level paired validation results; anomaly performance is evaluated only later under the frozen common anomaly protocol.
