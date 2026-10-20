# Immutable input inventory

The paired bundle contains the exact EXP013 architecture, scientific/engineering protocols, DATA/RUN/STUDY/engineering/final freezes, original graph/control reports, EXP012 detector/config/evaluator and reports, completed EXP014 reports/final receipts, exact NASA arrays and annotation bytes. The 162 included TRAIN/TEST array files were independently byte-compared with the attached original `archive.zip`; all match. Annotation content is parsed only in post-seal evaluation, not to choose this adaptation.

The exact original source is **`MyDrive/EXP013_COMMAND_REFERENCE_V1_E03_CONV/main/FINAL_RESULTS.zip`**. Its 18 selected checkpoints and 729 native references are not present in this development workspace. The Colab notebook reads them automatically from Drive. No additional EXP014 upload is required. Missing input is a stopped verification step, never a request to retrain.

| Original EXP013 constituent | Count | Hash authority |
|---|---:|---|
| `models/{mission}__{role}__{seed}/selected.pt` | 18 | Original STUDY_FREEZE |
| `models/{mission}__{role}__{seed}/IDENTITY.json` | 18 | Original STUDY_FREEZE |
| `models/{mission}__{role}__{seed}/COMPLETE.json` | 18 | Original STUDY_FREEZE |
| `calibration/{mission}__{role}__{seed}/thresholds.json` | 18 | Original STUDY_FREEZE |
| `scores/SCORE_SEAL.json` | 1 | Original FINAL_AUDIT |
| `scores/units/{mission}__{role}__{seed}__{channel}/scores.npz` | 729 | Hashes inside the pinned original score seal |
| The same unit's `record.json` | 729 | Pinned original score seal |
| The same unit's `COMMIT.json` | 729 | Pinned original score seal |
| **Total required original constituents** | **2,260** | No invented file hashes |

Mission is MSL/SMAP; role is graph/removed/fusion; seed is 17/29/43. Native reference units follow the exact common 81-channel roster. Threshold files are read only to reproduce the old native output. They do not calibrate/select the new residual detector.

`INPUT_CONTRACT.json` lists all 73 independently known constituent hashes. `audit/REQUIRED_ORIGINAL_INPUTS.csv` enumerates every exact path, including all 2,187 dynamic unit-file bindings. Those remaining hashes are obtained from the verified original score seal at runtime, not guessed. A smaller ZIP with these exact paths, or an unpacked directory, is accepted. Do not add a top-level folder prefix. A Drive backup object directory is not the experiment root.

`INPUT_INVENTORY.json` pins every embedded original input. The package `MANIFEST.json` pins the complete paired input/source bundle. Golden detector examples are hash-bound to their respective original EXP012/EXP014 score seals. Prior reports are preserved byte for byte and their original report manifests are verified. Historical count aggregation is checked without repeating training or historical scientific evaluations.

The completed EXP014 GDN/Hundman values and all original EXP012/EXP013 results are reused. EXP014 GDN checkpoints, CATCH checkpoints, optimisers, TRAIN fitting caches and synthetic confirmation cases are not inputs to EXP015. No unseen weights or incomplete checkpoint are substituted.

At startup, every required original constituent is read and verified even when a local cache exists. Cached selected weights are additionally payload-verified, strict-loaded, checked for their exact role/seed/roster/scaler and model-state digest. Native output provenance, target support and threshold/state bindings are checked. After execution the used originals are read again and required to be unchanged. Unused original archive constituents are not claimed to have been replayed.
