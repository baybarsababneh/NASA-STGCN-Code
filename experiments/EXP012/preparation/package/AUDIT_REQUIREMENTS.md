# Audit and reviewer addendum

Run `python src/audit_exp012.py --results /path/to/full_results --package /path/to/original_input_bundle_directory` in the pinned environment. It opens only saved TRAIN traces/checkpoints; it neither trains nor loads NASA TEST. The full results archive intentionally depends on the immutable input bundle for original486 GCN archives/data; do not duplicate those hundreds of files in every summary.

Audit before accepting preparation:
1.81-channel roster (27MSL54SMAP), three seeds,486 frozen parents,243 selected EXP005 checkpoints; exact byte/config/support and scaler hashes. EXP005 provenance must originate from existing selected checkpoints, not retraining.
2.972 model preparation +81 persistence +1053 qualification=2106 unique committed units, no missing/extraneous jobs.243 new LSTM fits,0 GCN fits. Check epoch ceilings and training cost; no retry changed settings.
3.486 original exact-GPU numerical replays; all parents/state unchanged; repeatability. Historical TCN CPU→GPU prediction differences are reported, not equated with a fresh scientific fit or CPU/CUDA equivalence.
4.All model weights fixed before first statistical calibration read. Inspect incremental data-open phase log and model/calibration/benchmark hash chain. Integrity hashing of calibration bytes is permitted pre-freeze; statistical array loading is blocked.
5.Recompute1053 q99 thresholds from saved calibration traces, all4212 diagnostic rows and both detector flags/notification indices/window logs. Verify empty/short/degenerate cases represented and no channel-specific rescue.
6.Check synthetic engineering/golden parity evidence, corrected tied AP, exact epoch-resume and checkpoint corruption rejection. Local engineering does not substitute for the production CUDA gates.
7.Verify environment/bootstrap/session logs, interrupted-run single-writer handling, output/source manifests, archived notebooks. Colab provisioning workaround is engineering only; unchanged scientific source/config is mandatory.
8.No NASA TEST arrays, label files, TEST metrics, threshold winner or detector promotion. COMPLETE_PREPARED_AWAITING_REVIEW is not evaluation permission.

Send a separable reviewer addendum: summary ZIP, this protocol, evaluation contract, model_inventory.csv, forecasting_summary.csv, thresholds.csv, nominal_diagnostics.csv, nominal_detector_summary.csv, calibration_resolution_summary.csv, freeze/audit/source/environment records; full result ZIP, original input bundle and EXP012_execution_provenance.zip for independent checkpoint/trace/execution audit. Historical report/results remain authoritative for their completed experiments. Add EXP012 as preparation only; no anomaly numbers, no positive detector conclusion, no rewriting EXP008–011 negatives or historical AP.

Outstanding before final evaluation: independent review; separate immutable score-export and label-evaluation runners; actually executed Colab notebook (download/save after completion); any serious reviewer issue. Historic TEST exposure and finalTRAIN prior use must remain explicit. A successful preparation does not eliminate those limitations.
