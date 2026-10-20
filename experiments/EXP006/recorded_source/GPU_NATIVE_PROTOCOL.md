# EXP006_GPU_V1 — independent GPU-native restart

## Scientific status

This is a new execution version, not a continuation of CPU EXP006 and not a
claim of CPU/CUDA equivalence. The decision to restart precedes new GPU screening.
The completed 288 CPU screening fits, any CPU robustness fits, and the failed
migration verification remain historical provenance. No CPU score, shortlist,
checkpoint, partial fit, or best epoch enters this experiment's decisions.
EXP001–EXP005 and the locked retrospective anomaly benchmark remain unchanged.
NASA TEST has historical project exposure; this restart does not restore an
unseen holdout. No NASA TEST arrays, residuals, or anomaly labels are included.

## Preserved scientific design

The authoritative original specification is in
`provenance/frozen_cpu/EXP006_PROTOCOL.md` and
`configs/v2_exp006_gcn_tuning.json`. Original source hashes are retained.

- The same 12 mission-specific candidates: RF 7/15/31, kernel 3, dilation stacks
  [1,2]/[1,2,4]/[1,2,4,8], lookback 64, widths MSL 32/64 and SMAP 64/128,
  learning rates 1e-3/3e-4. One fixed graph-message layer; message width equals
  temporal width; no new gates, dropout, normalization or edge learning.
- Identical original model classes, graph support, node retention, equal edge
  normalization, fit-only command selection and telemetry standardization.
  MSL retains fit-varying nodes but only the selected top-4 command edges enter
  telemetry. Direct-TCN control uses those exact incoming sources.
- AdamW, weight decay 1e-4, batch 256, epoch ceiling 80, patience 6,
  min_delta 1e-7, clipping norm 1.0. The frozen implementation updates the
  selected checkpoint only when validation loss improves by more than
  min_delta; that behavior is preserved (not silently replaced by an absolute
  minimum checkpoint rule). No warm start, AMP, TF32, compilation or batch change.
- The canonical 60/20/20 TRAIN target split, floor boundaries, preceding causal
  window context, and TRAIN-only scaling/support are unchanged. Calibration
  windows are constructed by the original loader but never scored or used for
  any architecture decision. No calibration thresholds are selected here.
- Screening roster MSL C-1/P-10/M-7/P-14/T-13/D-16; SMAP
  P-3/A-8/G-6/E-1/G-4/A-5. Seeds 17/29; 288 fits. Mission ranking is the
  arithmetic mean over six channels and two seeds; exact ties use mean parameter
  count then ID. The new GPU results determine the top two afresh.
- Robustness: same top two plus historical EXP002 architecture and selected
  EXP005 architecture, refitted from scratch on GPU. R1 fit/stop/assess
  [0,.4)/[.4,.5)/[.5,.6), R2 [0,.5)/[.5,.6)/[.6,.7).
  Recompute preprocessing/support within each fit block. Two seeds, 12 channels,
  four roles, two origins = 192 fits. No screening re-selection from these results.
- Confirmation: both shortlisted candidates per mission, 81 channels and seeds
  17/29/43 = 486 fits. All candidates are confirmed regardless of intermediate
  robustness outcomes. No extra search and no best-seed selection.

## Necessary versioned change to promotion comparator

Add **243 canonical GPU fits of the frozen EXP002 architecture**, using the same
81 channels, three seeds and training path. These are EXP006 reference controls,
not replacements for the historical EXP002 results. Promotion against original
CPU MSEs would confound tuning with the known backend trajectory differences.

All three original gates are preserved, applied against this GPU reference:
1. At least 2% lower full-channel mission mean validation MSE.
2. Strictly lower mission mean validation MSE for each seed 17, 29 and 43.
3. Nonnegative arithmetic mean of the R1 and R2 relative mission improvements
   against their own GPU EXP002 refits: mean_f[100*(reference_f-candidate_f)/reference_f].

Among passing candidates choose lowest full-channel mean MSE, then candidate ID
for an exact tie (as in the original final-selection code). If neither passes,
retain the EXP002 architecture for that mission and stop replacement searching.
A zero reference denominator is an audit error, not an invitation to change the rule.
These are practical development gates, not statistical significance claims.

Execution order: preflight → 288 screening → frozen shortlist → 192 robustness
→ 243 canonical reference controls → 486 confirmation → promotion → exports.
Total **1,209 research fits**, plus inexpensive synthetic engineering checks.
No all-channel EXP005 rerun is added. Its completed CPU study stays historical;
the prespecified earlier-origin comparisons to EXP005 use fresh GPU controls.

## Execution and reproducibility contract

PyTorch 2.11.0+cu130, NumPy 2.1.3, pandas 2.2.3 in a separate Colab virtual
environment. Python 3.12 or 3.13 is supported by setup; the exact interpreter,
all installed package versions, PyTorch build, CUDA/cuDNN, driver, GPU model and
capability are locked on first execution before research fits. Subsequent
sessions install the saved complete package version list and must match that
fingerprint. GPU UUID and session timestamp are recorded but are not equality
gates; a different unit of the same GPU model is allowed. GPU model, driver or
software mismatch stops execution; it never silently starts a second cohort.

Deterministic algorithms, cuDNN deterministic mode, benchmark off, CUBLAS
workspace :4096:8, TF32 off, float32, explicit AdamW foreach=False/fused=False,
CPU initialization with fixed seeds followed by CUDA training, one worker/GPU.
The explicit optimizer flags fix the GPU implementation path. No CPU fallback.
The old cross-backend tolerance tests are retained as historical evidence, not
relaxed or reused as an acceptance gate for a different experiment.

Preflight checks the complete bundle hash roster, 81 TRAIN arrays, mission
counts, finite values, all graph support/fold contracts, screen roster, persistence
initialization, RF behavior with a nonzero head, exact repeatability of tiny
synthetic GPU fits, GPU checkpoint reload, and the RF7 historical/current model
algebra. Synthetic data do not affect screening or hyperparameters.

## Persistence, resumption and failure handling

One active notebook only. A best checkpoint, epoch validation curve, result and
job specification are archived after every fit. The archive is copied to Drive,
read-back SHA256 checked, then a completion marker is written last. Resume verifies
the archive hash, complete file set, job identity and frozen execution hash.
CSV existence alone never marks a fit complete. Corrupt completed artifacts stop
the run for investigation. Uncommitted interrupted fits restart from their
original seed; optimizer state is not partially restored. At most the active fit
must be repeated. Confirm and screen are separate frozen stages, so their planned
overlap is not deduplicated or mistaken for independent evidence.

A heartbeat lease detects accidental duplicate notebooks and blocks an active
second writer. After a dead session the lease expires in ten minutes. Google
Drive is not a distributed database; never deliberately run concurrent notebooks.
Ongoing writes check ownership. No automatic OOM batch reduction, candidate skip,
seed replacement, fallback device, tolerance relaxation or partial-stage ranking.
Failures are visible; all earlier verified commits are retained.

This notebook cannot override Colab service limits or guarantee a single-session
completion. After runtime termination, reconnect to the same GPU class and Run
All again (Drive may request authorization). Completed commits are reused.
The notebook does not implement service-limit bypasses or idle keep-alives.

## Final outputs and interpretation

Final outputs include per-fit CSV, screening ranking, shortlist, canonical and
seed summaries, earlier-origin assessment results, paired channel and family
effects, all promotion gates, selected model specs, epoch-ceiling counts,
training time, peak GPU memory, epoch curves, checkpoints, hashes, code and
software/hardware manifests. Summary and full result ZIPs are written to Drive.
The small summary is offered for browser download automatically. Input TRAIN
data are separately preserved in the uploaded bundle; result ZIPs exclude them.

Forecasting improvements are not anomaly-detection improvements. This experiment
does not claim an independent test estimate: the canonical validation blocks
select candidates and overlap historical development, while earlier-origin
assessments are robustness checks with overlapping training histories. Any later
NASA anomaly evaluation remains explicitly retrospective.

## References for execution limitations

- https://docs.pytorch.org/docs/2.11/notes/randomness.html
- https://pytorch.org/get-started/previous-versions/
- https://research.google.com/colaboratory/faq.html
