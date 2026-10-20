# Start here

This is a bounded **synthetic-supervision feasibility pilot**, not a selected detector and not NASA evaluation. It uses the frozen EXP010 predictions and never loads a forecasting model.

1. Open `EXP011_FUSION_PILOT_V1_Run_All.ipynb` in Colab. A CPU runtime is sufficient; an A100 brings no benefit.
2. Run All, authorise Drive, and upload `EXP011_FUSION_PILOT_V1_bundle.zip` plus your existing `EXP010_MECHANISMS_V1_results.zip` when requested. Previously saved inputs are reused.
3. The notebook verifies inputs/source, installs an isolated environment, runs engineering tests, performs at most 440 unique small detector fits, and exports the results to Drive. Results are checkpointed after every fit.

Keep `RELEASE_STALE_LOCK=False` normally. After a hard runtime disconnect, confirm that the old runtime is terminated, set it to True once, and Run All to resume. This archives the stale lock and reuses verified fits. Do not run two copies concurrently. Source/environment/input mismatches stop safely.

Read `STUDY_CLOSE.json` and `arm_decisions.json` first. A pass means only eligibility for a separately frozen all-channel TRAIN-only confirmation, not permission to evaluate NASA TEST. A failure stops this detector branch; do not modify settings to obtain a pass.

Inputs are hash-bound to the supplied EXP010 full export. Do not upload NASA datasets, anomaly CSVs, TEST arrays, or final TRAIN calibration arrays. The source bundle already contains the required parent metadata.

Locally: install requirements in a clean Python environment; set OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1; run `python src/test_pilot.py`; then `python src/pilot.py --input /path/to/EXP010_MECHANISMS_V1_results.zip --output /durable/EXP011_FUSION_PILOT_V1`.

The scientific run has not been executed in preparing this delivery. Local engineering tests and full EXP010 input/feature loading passed. Read PROTOCOL.md before committing compute.
