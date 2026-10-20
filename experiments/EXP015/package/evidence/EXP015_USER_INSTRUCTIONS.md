# EXP015 — Frozen EXP013 GCN with Hundman Anomaly Detection

## Objective

I want you to design, implement, rigorously test and package one final **small, supplementary anomaly-detection experiment** for my NASA SMAP/MSL spacecraft research project.

The experiment must investigate:

**Would the frozen command-reference GCN developed in EXP013 perform substantially better at detecting real NASA spacecraft anomalies if we replaced its original anomaly-scoring/detection pipeline with the Hundman-adapted detector successfully used in EXP012 and EXP014?**

We discovered in EXP014 that a frozen GDN model achieved dramatically different anomaly-detection results depending on the scoring and detection procedure:

| Mission | GDN with EXP013 pipeline | GDN with EXP012 Hundman detector |
|---|---:|---:|
| MSL | 0.067379 | 0.747087 |
| SMAP | 0.059906 | 0.793519 |

These are corrected event F0.5 scores.

This motivates investigating our own EXP013 command-reference GCN, whose original results were:

- MSL corrected event F0.5: 0.0265
- SMAP corrected event F0.5: 0.0277

**These values are not targets to beat. An unchanged or worse result is equally important scientific evidence.**

## 1. Inspect the existing work first

Review the existing research materials, especially:

- EXP013 scientific protocol, source code and completed frozen results.
- EXP013 graph, graph-removal and direct-fusion architectures.
- EXP013 trained checkpoints and output representations.
- EXP012 exact `HUNDMAN_ADAPT_V1` detector, including numerical safeguards.
- EXP014 source, validation tests, exact Hundman implementation, A100 compatibility work and final results.
- The completed scientific report and audit documentation.

The existing EXP013 final archive is located in Google Drive at:

`MyDrive/EXP013_COMMAND_REFERENCE_V1_E03_CONV/main/FINAL_RESULTS.zip`

The completed EXP014 comparison and frozen Hundman implementation are in:

`MyDrive/EXP014_LITERATURE_COMPARISON_V1/main/`

If you cannot directly access these Drive files, identify the exact files required. Do not retrain or substitute missing checkpoints.

Determine whether the frozen EXP013 command-reference outputs or necessary score arrays have already been saved and can be reused without GPU inference.

**Do not rebuild the previous experiments.**

## 2. Independently establish the scientifically correct detector adaptation

EXP013 is architecturally different from GDN and EXP012.

Its proposed model combines:

- A command-conditioned normal telemetry reference.
- Learned command-to-telemetry graph messages.
- A discriminator trained using synthetic anomalies.
- A clean-reference Huber objective and synthetic binary classification loss.

Therefore, simply replacing a numerical threshold on its existing discriminator output may not reproduce a Hundman-style forecast-error detector.

I want you to investigate two possible detector adaptations.

### A. Primary experiment: Command-reference residual + Hundman

Take the frozen EXP013 model's normal telemetry reference and calculate the absolute difference between that reference and the observed telemetry.

Determine the correct causal alignment, target representation, inverse scaling and original-unit residual construction.

Apply the **exact frozen EXP012 `HUNDMAN_ADAPT_V1` detector** to those residuals.

This is the primary experiment because it uses a residual signal compatible in form with Hundman's detector.

However, explicitly acknowledge that a command-only normal reference is not equivalent to GDN's telemetry-history forecasting model. The result is a comparison of detection pipelines, not a perfectly controlled architecture-only ablation.

### B. Secondary experiment: Discriminator score + Hundman-style thresholding

Independently determine whether applying Hundman-style dynamic thresholding to EXP013's existing learned discriminator scores is scientifically meaningful.

If it is:

- Define the score transformation, orientation, smoothing and treatment of calibration clearly.
- Label the resulting method a **hybrid discriminator–Hundman detector**.
- Do not claim it reproduces the original Hundman method.
- Freeze the procedure before examining new evaluation outcomes.

If it is not scientifically meaningful, exclude it and explain why.

**Do not force a detector onto an incompatible score simply to obtain another result.**

The choice between these experiments should be justified by the mathematics and actual model implementation, not by whichever approach achieves a better NASA TEST score.

## 3. Include the appropriate EXP013 controls

I want to know whether any improvement is genuinely associated with the graph-message pathway.

Therefore, evaluate the same compatible detector on:

1. **EXP013 graph model:** The proposed command-reference GCN.
2. **EXP013 matched graph-removal control:** The same architecture without effective command-message influence.
3. **EXP013 direct-fusion control:** Uses command information without the graph pathway.

Use the existing verified frozen checkpoints and available seeds 17, 29 and 43 where appropriate.

Preserve their distinct architectures, trained parameters and original roles.

Do not retune one arm differently from the others.

If identical score constructions are not valid across these arms, explicitly document the obstacle and make the fairest bounded comparison possible. Do not fabricate a controlled comparison.

The main research question remains whether **the graph itself adds anomaly-detection value**, not merely whether changing the detector improves all three models.

## 4. No retraining or retrospective optimisation

These requirements are strict:

- No new neural-network fitting.
- No checkpoint modification.
- No new hyperparameter searches.
- No additional synthetic-fault training.
- No modification of EXP012, EXP013 or EXP014 scientific outputs.
- No detector selection or parameter tuning based on NASA TEST anomaly labels.
- No changes to the existing event evaluator.
- No replacing failed results with favourable alternatives.

Use the previously frozen Hundman algorithm wherever it is directly applicable.

Preserve its smoothing, dynamic thresholding, pruning, timing, startup and score-support conventions.

If a genuinely new compatibility decision is required, document and freeze it before evaluating against anomaly labels.

Recognise that historical NASA TEST exposure makes these additional analyses retrospective and exploratory.

Name the new work something like:

`EXP015_EXP013_GCN_HUNDMAN_SUPPLEMENT_V1`

Keep all files separate from previous experiments.

## 5. Evaluation requirements

Report the complete anomaly-detection picture, not just event F0.5.

For both MSL and SMAP, include:

- Corrected event F0.5.
- Event precision and recall.
- Strict point precision, recall and F1.
- Nominal alarm occupancy.
- Nominal alarm starts per 1,000 samples.
- Actual notification timing and missed events.
- Relevant descriptive paired uncertainty.
- Differences between graph, graph removal and direct fusion.

Reuse the exact frozen common NASA evaluation domain, labels, aggregation rules and scoring implementation from EXP012/EXP014.

Report alongside, but do not modify:

- EXP012 GCN and controlled LSTM results.
- EXP013 original graph/control results.
- EXP014 GDN + Hundman results.

Keep detector adaptations and complete pipelines clearly labelled.

If the new command-reference residual improves performance but the graph-removal control improves equally or more, say so explicitly.

If performance remains poor, explain what the evidence allows us to conclude without inventing a mechanism.

## 6. Google Colab engineering requirements

Build a fully runnable Google Colab notebook for an **NVIDIA A100 40 GB GPU**.

Also support the A100 80 GB where it can be verified as numerically and operationally compatible, without altering scientific settings.

We previously experienced Colab problems, including:

- A100 40 GB versus 80 GB device-identification failures.
- PyTorch/CUDA compatibility and GPU convolution issues.
- FP32 numerical-equivalence failures.
- Interrupted sessions and checkpoint recovery problems.
- A Matplotlib error caused by Colab's inherited interactive backend.

Use the proven engineering components from EXP014 where appropriate.

In particular, ensure isolated subprocess environments use:

`MPLBACKEND=Agg`

Do not repeat the previous Matplotlib backend error.

The notebook must include:

- Automatic runtime checks.
- Tested dependency installation.
- Actual full-shape GPU checks.
- Model checkpoint and native-output verification.
- Conservative memory use.
- Clear real-time progress.
- Estimated remaining runtime after work begins.
- Automatic backups.
- Hash-verified, interruption-safe Google Drive recovery.
- Atomic result commits.
- No unnecessary recomputation of completed channels.
- Final scientific and integrity audits.

The notebook should be executable through **Run all**, with minimal interaction beyond Google Drive authorisation and uploading genuinely necessary immutable inputs.

Do not weaken correctness checks to make a runtime pass.

## 7. Testing before delivery

Conduct substantial engineering and numerical verification.

At minimum:

1. Verify checkpoints and score-input identities.
2. Test inference shape, causal alignment and inverse scaling.
3. Verify native EXP013 output reproduction against saved references.
4. Verify the exact EXP012 Hundman detector against established regression examples.
5. Test the common NASA evaluator independently.
6. Test interruption, backup and recovery.
7. Execute a miniature end-to-end experiment.
8. Verify the final export's hashes and manifest.
9. Check that original experiment files are never modified.

Clearly distinguish CPU/synthetic tests from real A100 NASA execution.

If you cannot access an A100 during development, do not claim the actual GPU experiment has passed.

## 8. Scope and estimated runtime

This should be a **small inference-and-detector experiment**, not another large EXP013 training run.

First determine whether saved EXP013 reference outputs already permit detector evaluation without running the models again.

Reuse them wherever scientifically valid.

If inference is required, load only the frozen selected checkpoints and run the minimum necessary computations.

Avoid creating an unnecessarily large experimental framework.

Provide an evidence-based runtime estimate, separating:

- Setup and source verification.
- GPU inference, if necessary.
- Detector processing.
- Scoring and final export.

Do not invent benchmark times before measuring them.

## 9. Deliverables

Produce:

1. Complete runnable Colab `.ipynb`.
2. Paired input/source bundle ZIP.
3. `START_HERE.md` with exact Google Drive instructions.
4. Scientific decision document explaining residual and discriminator-score compatibility.
5. Verification/test results.
6. Input inventory showing all required original EXP013/EXP014 artifacts.
7. Automated recovery and restart instructions.
8. Report-ready tables and figures.
9. A final audit and research summary.

The notebook must produce an immutable final result archive that I can send back for independent analysis and incorporation into the Overleaf report.

Do not edit the existing Overleaf paper during this task. This is a separately labelled supplementary experiment whose findings may be incorporated later.

## 10. Final instruction

**I want you to implement and deliver this experiment, not merely outline it.**

Use the established EXP012–EXP014 evidence and original source code as your authority.

Do not silently redesign the models or the detector.

If the necessary checkpoints or residual information is missing, explain exactly which files I must provide rather than retraining anything.

The most important scientific question is:

**Was EXP013's disappointing anomaly-detection performance primarily associated with its scoring/detection pipeline, and does its graph-message pathway provide useful anomaly evidence when evaluated through a Hundman-compatible detector?**

Answer this question with a bounded, controlled experiment and report both positive and negative outcomes accurately.