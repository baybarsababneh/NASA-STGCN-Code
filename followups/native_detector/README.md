# Native-detector comparison

Analysis ID: `HUNDMAN_NATIVE_DETECTOR_CROSSOVER_V1`. Applies the native and adapted detectors to frozen forecasts. No training or parameter selection is performed. This follow-up belongs to the EXP001–EXP015 study.

Stage both follow-ups with `tools/stage_workspace.py followups` as described in `docs/REPRODUCTION.md` at the repository root. The staged parent must contain `hundman_native_crossover/`, `lstm_benchmark_audit/`, `project_sources/03-archive.zip`, and `project_sources/02-labeled_anomalies.csv`. Full detector processing also needs the original EXP012 score export under the earlier audit's `materialized/NASA Research Project/` directory.

The recorded environment used Python 3.12.14, NumPy 2.3.5, and pandas 2.2.3. Install the earlier audit's dependencies as well; no GPU or neural inference stack is required.

From the staged parent directory:

The public copies currently fail the original full-tree package seals. These commands require the exact original sealed inputs for replay; the reviewed ZIP alone is insufficient.

The public copies currently fail the original full-tree package seals. These commands require the exact original sealed inputs for replay; the reviewed ZIP alone is insufficient.

The public copies currently fail the original full-tree package seals. These commands require the exact original sealed inputs for replay; the reviewed ZIP alone is insufficient.

```bash
python hundman_native_crossover/run_equivalent_cached.py
python hundman_native_crossover/verify_extension.py
python hundman_native_crossover/summarise_verified.py
```

Run in a working copy: these scripts write output files. The E01 wrapper uses exact memoization and skips zero-standard-deviation searches that cannot change the fallback threshold. Original source, parity receipts, and cache counts are retained. The second and third commands recount saved stream outcomes without repeating detector processing; they still need the earlier audit outputs and labels.

Principal outputs are `outputs/mission_metrics.csv`, `channel_metrics.csv`, `paired_channel_bootstrap.json`, and the verification receipts. Decision streams are companion artefacts. The released LSTM has one seed slot; it is not three independent runs. Notification times are reconstructed from information availability. Undefined delay-vector fields must not be interpreted as measured zero delay.

The original package seals and legacy source hashes are retained. Privacy edits already present in this distribution mean the seals cannot certify the current public tree. Exact legacy verification requires the originals. See the root `docs/PUBLIC_RELEASE_REVIEW.md`; do not overwrite seals to force a pass. `ASSESSMENT.md` retains the scientific interpretation.
