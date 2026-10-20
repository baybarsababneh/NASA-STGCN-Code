# Run EXP015 in Google Colab

1. Open the paired `EXP015_EXP013_GCN_HUNDMAN_SUPPLEMENT_V1_Run_All.ipynb` in Colab. Select an NVIDIA A100 40 GB runtime. A100 80 GB is also accepted only after the same numerical checks pass.
2. Keep the completed original archive at **`MyDrive/EXP013_COMMAND_REFERENCE_V1_E03_CONV/main/FINAL_RESULTS.zip`**. It must be the full completed result archive, not the E03 input bundle or reports ZIP. If it is elsewhere, edit only `SOURCE_PATH` in the first code cell. No original folder will be modified.
3. Choose **Runtime → Run all**. Authorise Drive and upload the matching EXP015 input bundle once when prompted. Later runs recover the bundle from `MyDrive/EXP015_EXP013_GCN_HUNDMAN_SUPPLEMENT_V1/main/` automatically. The prior EXP014 Drive directory need not be uploaded: its frozen detector, evaluator, reports and required evidence are included.
4. The notebook installs an isolated numerical environment, runs CPU regression checks, then verifies the original weights/native outputs and performs frozen inference plus residual detection. Full-batch GPU checks and bitwise original-output replay are mandatory. Failures stop with exact diagnostics.
5. Progress reports distinguish inference, detector processing and backup commits. ETA is measured from completed batches/channels after work starts; no unmeasured NASA runtime is promised. Setup/source verification, scoring and export report their own measured durations.
6. At completion the notebook shows the report tables/figures and the hash-verified final archive. Send **`MyDrive/EXP015_EXP013_GCN_HUNDMAN_SUPPLEMENT_V1/main/exports/EXP015_FINAL_RESULTS_<hash>.zip`** back for analysis.

For interruption recovery use the same runtime capacity, run slot, inputs and notebook, then **Run all** again. Read `docs/RECOVERY.md`. One channel/arm/seed is the recovery unit; verified completed units are skipped.

This is a small inference-only supplement: **zero new fits**, 18 selected checkpoint loads, 729 residual detector streams and complete replay of the old 729 native streams. The exact original checkpoint archive is not available in this development workspace. Real checkpoint/native replay and A100 execution must pass in Colab; CPU/synthetic verification is not labelled as GPU or NASA scientific validation.

Included outputs: mission and seed tables, point/event/nominal alarm measures, notification timing and misses, paired descriptive intervals, figures, score arrays, numerical gates, source files, audit and an immutable archive. The separately labelled discriminator hybrid is excluded for the frozen scientific reasons in `docs/SCIENTIFIC_DECISION.md`. Existing Overleaf files and EXP012–EXP014 results remain unchanged.
