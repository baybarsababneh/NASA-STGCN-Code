# EXP006_GPU_V1 completed

1209/1209 fits verified. TRAIN-only forecasting development; no anomaly evaluation.

Promotion uses same-backend EXP002 architecture refits. CPU EXP006 and original EXP001–EXP005 are unchanged.

- MSL: retain_exp002_architecture; exp002_reference; mission mean validation MSE 0.202299289.
- SMAP: promote_gpu_exp006; g09_rf31_base_lr1e3; mission mean validation MSE 0.102449889.

See tables/promotion_checks.csv for each prespecified gate and tables/paired_channel_effects.csv for concentration of gains.
This does not establish improved anomaly detection or a new independent holdout. Do not open another replacement search if promotion fails.

Checkpoints are inside fits/<stage>/<job-id>.zip; each includes preprocessing metadata, graph features, specification, history and hashes.