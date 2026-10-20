# EXP007 GPU-native gate study

1. Open **EXP007_GPU_V1_Run_All.ipynb** in Colab.
2. Select a GPU runtime, preferably **A100**, and choose **Run all**.
3. Authorise Google Drive and upload **EXP007_GPU_V1_input_bundle.zip** when prompted.

The bundle already contains the audited source, frozen protocol, dependencies
and all 81 TRAIN arrays. No local assembly or earlier checkpoints are required.
No NASA TEST arrays, labels or anomaly evaluation are included.

Everything after setup is automatic: preflight → earlier-origin robustness →
canonical confirmation → architecture freeze → matched message removal → export.

## What is being tested

- MSL baseline: EXP002 architecture, RF7, hidden 32, learning rate 0.001.
- SMAP baseline: selected EXP006 g09, RF31, hidden 64, learning rate 0.001.
- One new candidate: a causal scalar gate on the existing total graph message,
  initialised exactly to one, with 65/129 extra parameters respectively.
- Controls: the fixed baseline and one learned constant scalar multiplier.
- No architecture grid, extra seeds, epoch extensions or learning-rate search.

All models are fresh same-environment GPU fits. Earlier experiment results are
preserved and do not serve as the numerical promotion comparator.

| Stage | Fits |
|---|---:|
| Three roles, 12 original screening channels, two seeds, two earlier origins | 144 |
| Three roles, all 81 channels, seeds 17/29/43 | 729 |
| Selected architecture with messages removed, all 81 channels, three seeds | 243 |
| **Total** | **1,116** |

Dynamic gating is promoted separately by mission only if it achieves ≥2% lower
canonical mean MSE than fixed, improves each seed versus fixed, passes mean
earlier-origin robustness versus fixed, beats static in canonical mean MSE,
and passes mean earlier-origin robustness versus static. Otherwise retain the
EXP006-selected fixed architecture. The static scalar is an explanatory control,
not another promotion candidate. Full definitions are in PROTOCOL.md in the bundle.

Selection is saved before the message-removal fits. Those fits remove the command
aggregate before both the gate and message branches, preserving the remaining
architecture and settings. They explain the chosen GCN and cannot reopen selection.

## Saving and resume

Results: `My Drive/NASA_EXP007_GPU_V1/results/`.
Use **one active notebook** only. Each completed fit is durably archived and
checksum-verified before it receives a completion marker. A disconnected browser
may leave the runtime running. Reconnect to that session first.

If the runtime has ended, obtain the same GPU model and Run all again. The saved
bundle is reused and completed fits are verified/skipped. An unfinished fit
restarts from its seed. Allow ten minutes for a crashed session's lease to expire.
Driver/software/GPU mismatches and corrupt completed files stop safely. Do not
delete the freeze or change settings to bypass a stop.

Colab may terminate sessions. The notebook preserves completed work but cannot
reconnect itself or guarantee a single uninterrupted run. Keep several GB free
for dependencies, checkpoint archives and exports.

## Outputs and next step

- `SUMMARY.md`, `selected_models.json`, `tables/promotion_checks.csv`.
- `ARCHITECTURE_FREEZE.json`, written before final ablation results exist.
- Mission/seed/channel/family summaries, paired bootstrap intervals, gate
  diagnostics, epoch ceilings, timing and parameter counts.
- Checkpoints and gate traces in `fits/`, indexed by `final_checkpoint_index.json`.
- Full provenance and two final ZIPs in `exports/`. The small summary ZIP is
  offered for download automatically.

After auditing completion, **stop GCN architecture optimisation**. Begin a
separately frozen detector-development stage with the fixed final forecasters.
Forecasting gains do not establish anomaly-detection gains. NASA TEST remains
retrospective, and is not accessed by this workflow.

Local engineering checks pass, including the full synthetic orchestration and
resume exercise. Actual CUDA/Colab execution is not available in this workspace.
The notebook therefore runs mandatory CUDA checks before any research fit.
