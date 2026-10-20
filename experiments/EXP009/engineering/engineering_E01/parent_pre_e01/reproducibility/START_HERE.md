# EXP009 — start here

This is a CPU residual-detector study, not a GPU training job. It uses the verified EXP008 saved predictions already included in the bundle. Do not upload the NASA dataset, TEST arrays, anomaly labels, or GPU checkpoints.

## Colab

1. Open `EXP009_DETECTOR_V1_Run_All.ipynb` in Colab. A standard CPU runtime is sufficient.
2. Run All. Authorise your Drive mount once and upload `EXP009_DETECTOR_V1_bundle.zip` when prompted.
3. The notebook installs an isolated CPU environment, checks the bundle and parent export, runs synthetic tests, and executes the bounded study. Results are committed incrementally under `MyDrive/NASA_GCN/EXP009_DETECTOR_V1/results`.
4. On a runtime disconnect, run the same notebook again. The bundle retained on Drive is reused; completed detector units are hash-checked and skipped. Do not edit source/config/results to force resume. An environment mismatch stops safely; restore the recorded environment or document an engineering version before continuing.
5. Share the exported result ZIP and `STUDY_CLOSE.json` for audit. A provisional pass does not authorise final NASA TEST evaluation.

No Colab workflow can guarantee an unlimited runtime. This study is much smaller than the training experiments; CPU costs depend on Drive latency. Concurrent runs against one output directory are unsupported and guarded with a process lock. The kernel releases the lock after a crash; no stale ACTIVE_RUN file needs manual deletion.

## Local alternative

Unzip the bundle into a new directory. In a dedicated Python 3.11–3.13 environment install `numpy==2.3.5 pandas==2.2.3`, then:

```
python src/test_exp009.py
python src/test_exp008.py
python src/run_exp009.py --input data/EXP008_results.zip --output /absolute/path/to/new/EXP009_results
```

Keep the same Python/package versions for resume. Do not change the manifest. Protocol, code and config are immutable within this version.

## What to send back

- `EXP009_DETECTOR_V1_results.zip`
- the executed notebook (with outputs)
- any exception traceback, if completion failed

The archive contains decision records, all committed units, source/config/protocol copies, resolved requirements, input audit, per-stream diagnostics, synthetic probes, paired intervals and hashes.

## Interpretation

- `NO_FEASIBLE_CANDIDATE`: no candidate passed the prespecified development gates.
- `AUDIT_FAILED_NO_FALLBACK`: the selected candidate failed reserved/earlier-origin checks; do not try another candidate on those checks.
- `PROVISIONAL_FREEZE_REQUIRES_REVIEW`: a candidate passed this surrogate study; real-anomaly performance remains unknown.

Read PROTOCOL.md before interpreting outputs. Reference blocks and audit records are reused project development data, not virgin holdouts. Frozen forecasting models and historical anomaly outputs are untouched.
