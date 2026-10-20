# Run EXP013 in one Colab notebook

Open `EXP013_COMMAND_REFERENCE_V1_Run_All.ipynb` in Google Colab. Select **A100 GPU (40 GB is the recommended option)**. Use a fresh runtime with Python 3.12 or 3.13. Scientific fitting requires the full-batch CUDA gate; final label evaluation is CPU work in the same notebook.

1. Run the pinned dependency cell. If the notebook requests a restart after replacing already imported packages, restart the session, then rerun from the top. Do not edit package versions or scientific settings.
2. Upload **`EXP013_COMMAND_REFERENCE_V1_input_bundle.zip`** when requested. It already contains the original byte-identical NASA arrays and audited completed EXP012 reports. You do **not** need checkpoints.zip, the EXP005 sidecar, the old score export, the original 83 MB archive, or another NASA data download. The small bundle excludes the NASA annotation CSV and old model weights. Archived EXP012 reports contain label-derived outcomes/event metadata, reflecting declared prior exposure; the new development/scoring code ignores those event intervals.
3. For a new run leave `RESTORE_SNAPSHOT = False`. To resume after a VM reset set it to True and upload the previously downloaded `EXP013_COMMAND_REFERENCE_V1_resume.zip` plus its `.sha.json` receipt. Restore into an empty local work directory. For a kernel restart in the same VM, leave it False; local verified progress is reused.
4. Run staging and engineering validation. Immutable official GDN/CATCH source ZIPs are fetched automatically, about 4 MB. Actual full batches and forward/gradient integrity are tested on your GPU before research fits. Any failure stops the run; do not change settings to force a pass.
5. Run development: 22 planned fits, TRAIN-only candidate selection, threshold calibration and study freeze. Logs identify job, epoch, timing and resume. Completed EXP012 models are never fitted or replayed. Save the generated local resume ZIP at the freeze milestone. The report records actual timing; no unmeasured GPU-hour estimate is promised.
6. Run quarantined synthetic confirmation, then frozen NASA TEST score export and verification. Save another resume ZIP after the verified score seal. All active files stay on `/content`.
7. **Only when seal verification passes**, upload your existing NASA `labeled_anomalies.csv` (or `02-labeled_anomalies.csv`) in the label cell. Its exact SHA is checked before parsing. No label file is needed earlier.
8. Run evaluation and final audit. Download the report ZIP and final resume ZIP. The notebook displays tables, the research report, the decision and final audit.

The single notebook orchestrates source modules in the bundle; it does not embed duplicate baseline implementations. `PROTOCOL.md` and `PROTOCOL.json` are frozen, not editable configuration templates. `LITERATURE_AND_DESIGN.md` explains the evidence and reproduction gaps. `BUILD_AUDIT.json` describes what was actually tested before delivery.

## Outputs

| File/folder under `/content/EXP013_work` | Meaning |
|---|---|
| `DATA_FREEZE.json`, `RUN_FREEZE.json` | Input, FIT statistics, package and runtime identities |
| `ENGINEERING_GATE.json` | Actual GPU full-batch and source validation |
| `models/…/selected.pt`, `history.json`, `COMPLETE.json` | New selected model, timing, parameter count and immutable completion |
| `development/…/selection.json` | All six candidates, nominal gates, objective and explicit failure status |
| `calibration/…/thresholds.json` | Frozen entity thresholds and GDN TRAIN score moments |
| `STUDY_FREEZE.json` | Pre-CONF/pre-TEST lock over all scientific artifacts |
| `confirmation/`, `CONFIRMATION_COMPLETE.json` | Locked synthetic case and nominal-alarm results |
| `scores/…`, `SCORE_SEAL.json` | All verified frozen NASA scores and issue-time decisions |
| `reports/RESEARCH_REPORT.md`, `DECISION.json` | Main interpretation, joint outcomes and scope limits |
| `reports/mission_metrics.csv`, `mission_seed_metrics.csv`, `channel_metrics.csv` | Strict point/event/AP/false-alarm performance with coverage and denominators |
| `reports/event_notifications.csv` | Event issue timing, late notifications and delays |
| `reports/paired_channel_bootstrap.json` | Prespecified descriptive NASA contrasts |
| `reports/synthetic_confirmation_*.csv`, `synthetic_paired_channel_bootstrap.json` | Valid/invalid cases, clean background overlap and held-out mechanism contrasts |
| `reports/PUBLISHED_FIDELITY_STATUS.json` | Source pins, adaptations, exclusions, numerical failures and unreproduced methods |
| `reports/EXP012_original_mission_metrics.csv` | Unchanged complete historical benchmark |
| `FINAL_AUDIT.json`, `ACCESS.jsonl` | Verified seals, access ordering and exposure declaration |

The final report ZIP is small; the full resume ZIP contains weights and can exceed 1 GB. Keep at least 15 GB of free local disk and use sufficient host RAM. Local archives are automatically refreshed after completed training jobs and at notebook milestones, but require a download or an optional **single archive copy** to Drive for VM-reset durability. Do not mount Drive as the work directory. Do not interrupt archive creation and assume that a partially copied archive is recoverable; receipts and manifest verification are mandatory.

## Interpreting a result

An audit PASS means that the run followed the locked protocol. It does not mean the GCN won. `NO_FEASIBLE_CANDIDATE`, a weak held-out-family result or a deterioration on either mission is a valid negative result. The original NASA arrays inherit TEST-min/max preprocessing, and TEST labels/results have been exposed in prior experiments. New NASA results are retrospective only, even if all descriptive intervals are positive. No independent SOTA, raw-data leakage-free or deployment claim is authorized.

GDN/CATCH comparisons use official architectures and losses with explicitly declared NASA adapters. STGLR could not be reproduced faithfully from located materials; OmniAnomaly is not approximated in a modern incompatible environment. The historical LSTM remains its original complete pipeline. Read the fidelity table before comparing any published headline scores.

## Failure and recovery

- Missing input: the error identifies the exact filename/path. Supply that file; do not rebuild EXP012 or substitute another NASA archive.
- Source download unavailable: supply the exact pinned `gdn.zip` / `catch.zip` with the protocol hashes to `/content/EXP013_work/official/`, then rerun. Cached verified snapshots already include them. Never switch to a new commit.
- GPU full-batch failure or deterministic-kernel failure: stop and preserve the logs. This release does not authorize smaller batches, AMP or modified published architectures.
- Epoch interruption: rerun the same development cell in the same environment. RNG/optimizer/sampler and carried mask gradients restore exactly at the saved epoch boundary.
- VM reset: restore the last downloaded/copied verified snapshot in an identical pinned runtime. Last unfinished work since that snapshot may need repeating. Completed committed work is reused.
- Hash, roster, source, environment or score-seal mismatch: stop. Never delete a seal, skip verification or request labels to continue.
- Numerical failure in a published model: the runner records explicit incomplete coverage. It does not fabricate scores or call that comparison successful.

CPU-only synthetic checks can be run outside Colab with `python tests/test_science.py`. The full fictional end-to-end fixture is `python tests/test_pipeline_fixture.py --official-work /path/to/local/test_cache`; it changes dimensions only inside isolated synthetic fixtures and never mutates the scientific protocol or reads NASA labels/TEST arrays.
