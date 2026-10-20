# TCN-GCN validation confirmation — Round 2

Recommendation: retain the simple two-layer causal TCN and lock the mission-specific widths below. All choices used TRAIN forecasting validation only. Calibration and NASA test anomaly evaluation have not been performed in this round.

| Mission | Channels | TCN_MSE | Fixed_GCN_MSE | Macro_reduction_pct | Median_paired_reduction_pct | W_L_T |
| --- | --- | --- | --- | --- | --- | --- |
| MSL | 27 | 0.20056 | 0.26586 | 24.563 | 19.409 | 19/2/6 |
| SMAP | 54 | 0.11047 | 0.12376 | 10.736 | 13.118 | 28/11/15 |

MSE is the channel-macro mean of fit-standardized validation error, averaged over seeds 17, 29 and 43. The fixed GCN received the same 80-epoch ceiling. W/L/T counts channel improvements, regressions and ties.

**Configuration to lock:**

| Mission | Hidden | Lookback | Kernel | Dilations | Graph | Epoch_ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| MSL | 32 | 64 | 3 | [1, 2] | top4 | 80 |
| SMAP | 64 | 64 | 3 | [1, 2] | star | 80 |

Both use kernel 3, dilations 1/2, effective receptive field 7, AdamW learning rate 0.001, weight decay 0.0001, patience 6, minimum improvement 1e-7, minibatch target 256, gradient clipping 1, ReLU, no dropout and a persistence residual. Keep all three seeds; restore the best validation epoch for each model.

**What was tested:** 81 of 84 planned screening fits across seven configurations per mission and six channels; three uncompetitive fits were excluded by an exact score bound. The screen spanned widths 16/32/64 and receptive fields 7/13/63/127/255. Promising alternatives had to improve on channels outside the screen before three-seed confirmation. Graph support, data splits and preprocessing stayed fixed. The uploaded TCN is causal and correctly oriented; its nominal 64-step window only uses the final seven observations. Seven correctness tests pass, including checkpoint/output and gradient equivalence checks.

**MSL robustness:** stabilized paired MSE ratios versus fixed GCN range from 0.699 to 0.724 across the three seeds (below 1 favors TCN). 18/27 channels exceed the near-zero-error cutoff. 3/81 selected-model runs still have their best epoch at 80.

**SMAP robustness:** stabilized paired MSE ratios versus fixed GCN range from 0.725 to 0.760 across the three seeds (below 1 favors TCN). 38/54 channels exceed the near-zero-error cutoff. 12/162 selected-model runs still have their best epoch at 80.

**Limits:** One chronological validation interval, repeated validation-based selection, small family groups and some ceiling-limited fits. Width changes alter model capacity. Archived LSTM/MLP/learned-edge references retain their smaller 40-epoch budget. Better forecasting does not guarantee better anomaly detection.

**Next local experiment:** run `python src/calibrate_locked_later.py --output local_calibration` from the Round 2 folder. This uses only the reserved final 20% of TRAIN with the frozen models and inherited threshold policy; it does not open NASA TEST or labels. The script is supplied but intentionally unexecuted.

The full audit, baseline/family/seed tables, costs, exact lock, code, checkpoints and reproduction instructions are retained in the accompanying research package. See `Round2_Report.md`, `Audit.md`, `README.md` and `results/`.
