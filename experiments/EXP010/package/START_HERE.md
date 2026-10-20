# EXP010 — start here

Open `EXP010_MECHANISMS_V1_Run_All.ipynb` in Colab with the same A100 GPU environment used for EXP007/EXP008. Run All, mount Drive and upload `EXP010_MECHANISMS_V1_bundle.zip` once. The focused bundle contains the required 114 frozen checkpoint archives and 19 original TRAIN prefixes. No other project upload is needed.

Results go into a new folder: `MyDrive/NASA_GCN_EXP010_MECHANISMS_V1/results`. Previous experiments remain unchanged. The notebook verifies source/data hashes, installs the pinned isolated environment, runs synthetic checks, performs exact nominal checkpoint replay and then runs the frozen perturbation bank. No training or NASA TEST evaluation exists in this workflow.

Interrupted runs resume from verified committed units. Use one active notebook only. The inherited Drive lease expires ten minutes after the last heartbeat following a lost runtime. Do not delete the lease, alter hashes, switch numerical environments or loosen replay checks to force continuation.

Expected completion: `COMPLETE_DIAGNOSTIC_NO_SELECTION`, 3,078 units. Download both the compact summary and full evidence ZIPs, and save the executed notebook with its outputs.

Main outputs:
- `nominal_context.csv`: residual structure by causal incoming-command activity.
- `response_streams.csv`: actual forecasting response, residual-only response magnitude, recovery and null interventions.
- `policy_diagnostics.csv`: all original scorer policies evaluated descriptively on actual versus residual-only perturbations.
- `descriptive_response_summary.csv`: compact descriptive responses. Always separate mechanical and post-hoc cohorts. Do not treat row counts as independent event samples.
- `RUN_FREEZE.json`, unit archives/markers, `OUTPUT_MANIFEST.json`, `STUDY_CLOSE.json`, source/protocol copies and session logs.

CPU tests, source/input checks and commit/resume/corruption tests were run locally. GPU/Colab execution has not been performed here. The first real run must satisfy inherited exact-replay gates before its checkpoint's perturbations. An environment mismatch stops safely.

No detector is selected, and no alarm budget changes. After audit, interpret whether the evidence supports one new bounded detector formulation or instead requires more operational context/labelled development data. Final calibration and retrospective NASA TEST evaluation are not part of EXP010.
