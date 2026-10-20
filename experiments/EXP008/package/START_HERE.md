# START HERE — EXP008 readiness study

## Decision

Run EXP008_READINESS_V1: saved-weight numerical replay, corrected-metric engineering checks and a bounded TRAIN-only nominal-calibration study. This is not detector selection or NASA TEST evaluation. Architecture optimisation remains closed.

## Run on Colab

1. Open EXP008_READINESS_V1_Run_All.ipynb in Google Colab.
2. Select an A100 GPU runtime. The workflow requires the original A100-SXM4-40GB/software fingerprint and stops if Colab supplies a different environment.
3. Run all. Mount Drive and supply EXP008_READINESS_V1_input_bundle.zip once when prompted. No old results exports or separate data uploads are needed.
4. Leave it running. It installs a pinned isolated environment using virtualenv, verifies input/provenance, replays 534 existing checkpoints, then generates the fixed descriptive tables. No model is trained.

Results are saved incrementally under MyDrive/NASA_GCN_EXP008_READINESS_V1/results. All inputs are copied to a separate durable input location. EXP001–EXP007 directories are never modified.

## If interrupted

Use the same notebook and bundle. Ensure the earlier runtime is stopped, wait ten minutes after its last heartbeat, then Run all. Verified committed jobs are reused. Partial uncommitted jobs restart inference only. Do not delete a lease or edit hashes/configurations to force progress. A missing/corrupt committed file, environment difference or numerical mismatch stops the study with evidence preserved.

The notebook automatically resumes with the same environment lock and run identity. It does not bypass failures or silently switch GPU/CPU backends. Colab availability cannot be guaranteed. If the required environment is unavailable, share the mismatch record before adopting a separately versioned inference-portability amendment.

## Completion

Expected terminal status: COMPLETE_NO_DETECTOR_SELECTED.

- 534/534 checkpoint replay jobs, including 486 final graph/control and 48 earlier-origin GCN checkpoints.
- 288 nominal diagnostic rows across the fixed four span/quantile policies, both missions, GCN and non-duplicated persistence.
-Zero new training fits, zero final-calibration values, zero NASA TEST or anomaly labels.
-exports/EXP008_summary.zip and EXP008_results.zip, each with SHA256.

The small summary is offered for download. Both exports remain on Drive. Save the actually executed notebook after completion for provenance. Send the summary, full results and executed notebook for audit before starting another scientific stage.

## What is inside the input bundle

Only first 80% TRAIN prefixes,534 unchanged selected EXP007 fit archives and commit records, original model/environment source and provenance, the frozen protocol, a four-policy diagnostic runner and synthetic-tested metrics. The final 20% TRAIN block and all NASA TEST/label/score payloads are physically absent.

## Testing status at delivery

Python source/notebook syntax, full input hashes/roster, tensor metadata provenance, synthetic metric/scorer cases,288-row diagnostic aggregation, transactional resume/corruption checks and 200 tied-score AP cases against scikit-learn were tested locally. The CUDA inference workflow has NOT been executed in this environment. Its534 exact numerical replay gates are the first scientific execution on Colab. No successful EXP008 outcome is claimed in advance.

## After the run

Preserve the existing reviewer package. Add EXP008 as a separately hashed addendum. The study cannot choose a detector from quietness alone, establish real-anomaly recall, or certify a1% false-alarm rate. The next decision belongs in an explicitly frozen downstream protocol informed by the independent review, not a silent automatic continuation to TEST.
