# EXP007_GPU_V1 completed

1116/1116 fits verified. Architecture selection used TRAIN only and was frozen before the final message-removal control.

- MSL: retain_EXP006_selected_architecture. Gate=fixed, RF=7, width=32. Mean validation MSE=0.202299289.
- SMAP: retain_EXP006_selected_architecture. Gate=fixed, RF=31, width=64. Mean validation MSE=0.102449889.

The static scalar is a mechanism control. Only the prespecified dynamic gate is eligible for promotion.
Message-removal and bootstrap results are explanatory diagnostics and cannot reopen architecture selection.
Positive improvement in message_removed_vs_* means removal forecasted better. It does not establish an anomaly-detection result.
Stop GCN architecture optimisation. Audit these outputs, then freeze a separate detector-development protocol before using calibration data.
NASA TEST remains retrospective. No NASA TEST scores or labels were read.