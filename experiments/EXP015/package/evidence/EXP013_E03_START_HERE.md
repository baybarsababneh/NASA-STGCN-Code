# EXP013 E03 CONV — begin the corrected run

1. Open **EXP013_COMMAND_REFERENCE_V1_E03_CONV_Run_All.ipynb** in Colab. Select an **A100**, preferably high host RAM. Mount Drive and upload the matching **EXP013_COMMAND_REFERENCE_V1_E03_CONV_input_bundle.zip** when requested.
2. Run cells in order. Section 3 performs and retains the first real MSL graph epoch; section 4 checks all corrected roles against native convolution and measures the remaining full batches. Section 5 continues all **22 fits**, selects using TRAIN, freezes, confirms synthetically and seals TEST scores.
3. Supply the original annotation CSV **only when section 6 requests it after verifying the score seal**. Evaluation, audit, Drive backups and `FINAL_RESULTS.zip` are automatic.

For a disconnect, reconnect an A100 with the same pinned runtime and rerun the same notebook with `RUN_SLOT='main'`. Verified E03 backups restore automatically. Keep one active VM per slot. Do not copy E01/P01/E02 checkpoints or gates into E03. All E03 fits start from their original seeds; native-convolution states are not bitwise-compatible continuations.

The graph correction has your reported A100 validation and **501.39× computational speedup**. This build performed CPU regressions, not another A100 run. The removed/fusion A100 comparisons and the four baseline capacity/timing checks are enforced in section 4, before the remaining fits. Failure stops; it never reduces dimensions, batch sizes or scientific gates.

At the approximate original 56 s/step, 501.39× implies **0.112 s/step of measured compute**. Applying that to all three roles and allowing SMAP to cost either 24/54 as much or the same as MSL gives **1.85–3.05 hours for the 18 new-model FIT compute loops at the full 25-epoch caps**. This is an extrapolation. GDN/CATCH, STOP, data handling, selection, confirmation, TEST scoring and Drive transfers are additional. Section 4 replaces extrapolation with measured per-role training/STOP projections for all 22 jobs. No defensible unconditional completion time for tonight exists before the baseline checks.

Details, honest FP32 differences and scientific limitations: `E03_AMENDMENT_AUDIT.md`. Tests are included; no lengthy test sweep is required to start Colab.
