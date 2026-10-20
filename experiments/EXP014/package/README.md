# EXP014_LITERATURE_COMPARISON_V1

This final bounded comparison adds **only frozen GDN inference through the exact EXP012 Hundman detector**. It performs **zero training fits** and reuses all completed GCN/LSTM and native GDN/CATCH results. The 22 EXP013 fits remain immutable.

Open `EXP014_LITERATURE_COMPARISON_V1_Run_All.ipynb` in Colab, select **A100 40 GB**, and choose **Runtime → Run all**. Follow `START_HERE.md`. The matching input bundle already contains NASA arrays, annotations, the 486 needed EXP012 score streams, original count tables, frozen sources and integrity records.

**One external input is still required:** the completed EXP013 archive at

`/content/drive/MyDrive/EXP013_COMMAND_REFERENCE_V1_E03_CONV/main/FINAL_RESULTS.zip`

The runner reads only 252 pinned constituents from it. An unpacked directory or a smaller ZIP containing those exact relative paths is also accepted. `docs/INPUTS.md`, `INPUT_CONTRACT.json`, `INPUT_INVENTORY.json` and `audit/INPUT_INVENTORY.csv` identify each path/hash and the evidence gaps. The notebook fails before dependency installation if the source path is absent. It never replaces a missing checkpoint with new weights.

New persistent outputs are confined to:

`/content/drive/MyDrive/EXP014_LITERATURE_COMPARISON_V1/main`

Keep one active notebook per run folder. Repeat **Run all** in the same folder after disconnection. Hash-verified backups restore committed channels, which are skipped. See `docs/RECOVERY.md`.

The output archive contains `FINAL_AUDIT.json`, both full-shape GPU/restoration receipts, sealed per-channel residuals/flags/timing, input/code/config bindings, report-ready CSVs, descriptive paired uncertainty, figures, LaTeX rows and a concise summary. The latest archive is identified by `LATEST_EXPORT.json`; the first `FINAL_RESULTS.zip` alias is preserved.

Read `docs/SCIENTIFIC_DECISION.md` before interpreting the tables. Common-detector evidence does not isolate architecture perfectly. Complete-pipeline NASA adaptations and paper-specific published scores remain separate. No TEST-label tuning, point adjustment, independent generalisation or deployment claim is authorised.

Development validation is CPU-only. Full native GDN shapes and a miniature interruption/recovery workflow have been tested. Real A100 memory, frozen checkpoint inference and all 81 new NASA streams remain unexecuted until the missing input is supplied and the notebook runs. `audit/TEST_REPORT.json` and `docs/COMPLETION_AUDIT.md` state the boundary precisely.

Local regression command, with the pinned dependencies installed:

```bash
python tests/run_tests.py --report /tmp/EXP014_TEST_REPORT.json
```

CPU fixtures use untrained synthetic weights and are explicitly barred from producing a production GPU PASS. No historical model is inferred or refitted by the tests.

The byte-preserved detector and evaluator live in `vendor/frozen/`. `vendor/gdn/` retains the existing pinned GDN source and its two previously audited PyG API compatibility edits. The original checkpoint reader and Drive backup routine are reused. No convolution correction is needed for this inference-only GDN stage.
