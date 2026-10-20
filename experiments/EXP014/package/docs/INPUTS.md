# Immutable input contract and remaining gap

The new bundle contains the released NASA arrays, exact annotation CSV, EXP012 detector/configuration, 486 byte-preserved GCN/LSTM score streams, complete historical count/timing tables, all supplied EXP013 report exports, DATA/RUN/STUDY/engineering/final freezes, original report/audit/protocol sources, and pinned native GDN source. No NASA data download or historical retraining is required.

`INPUT_INVENTORY.json` records every included file hash. `audit/INPUT_INVENTORY.csv` lists present files and all 252 missing EXP013 constituents. The original EXP012 complete seal remains unchanged; its explicit subset is not described as a newly verified entire1053-stream export.

| Missing immutable file | SHA-256 |
|---|---|
|models/MSL__gdn__17/selected.pt|32ed9cb7d217bd198544856120b49c657b23141e4d6fb8cbc77da8dea57dda4d|
|models/SMAP__gdn__17/selected.pt|53eac339653dadfeef344c4622606d1050b51a0355ea3f2bf82564fb13c02a29|
|models/MSL__gdn__17/COMPLETE.json|36944234297e9602d9216000bf30817fb193ef2e4f41f157d19deec4908c79ea|
|models/SMAP__gdn__17/COMPLETE.json|28576ca930b97c5bfec59ab0c5f28ea9cc2c8fdbbf92b86bc05ed8434b207678|
|models/MSL__gdn__17/IDENTITY.json|20e4b46c7c1defb535e131807e44b3ded45906ef89b2eb3d4b540f7e95b51fd6|
|models/SMAP__gdn__17/IDENTITY.json|88237a81f0700fe19608a6743604a0998dd9cd72b807a2a80294ac87dca9d1f1|
|calibration/MSL__gdn__17/thresholds.json|339eeb003b4bd303b53371b7515fbdf169ce7f4f8fc6a868c9a2b27c601b2016|
|calibration/SMAP__gdn__17/thresholds.json|2061ded58f2fd94b8173629e6bcc7e8030578006ee96820d3d971d867175f8c0|
|scores/SCORE_SEAL.json|b9f76f695947aa0675cf9864da839eae700e74fa4f224e351547ce9548387bc6|

Also required for each of the 81 channels:

`scores/units/{MSL or SMAP}__gdn__17__{channel}/scores.npz`

`scores/units/{MSL or SMAP}__gdn__17__{channel}/record.json`

`scores/units/{MSL or SMAP}__gdn__17__{channel}/COMMIT.json`

These 243 hashes are inside the original pinned score seal. Its bytes are currently missing, so this delivery does not invent individual hash values. All exact paths are enumerated in the CSV. During recovery the runner obtains those values from the verified seal and checks every reference before inference. Threshold files are used only for replaying the **old** native scores/flags; they do not select or calibrate the new Hundman detector.

Expected location:

`/content/drive/MyDrive/EXP013_COMMAND_REFERENCE_V1_E03_CONV/main/FINAL_RESULTS.zip`

The completed notebook receipt reports a 1230763737-byte final archive. Size is descriptive, not a compatibility hash. The runner accepts that ZIP or an unpacked directory with the same relative paths. A smaller ZIP containing just the 252 constituents is sufficient; use paths relative to the experiment root, without an extra top-level directory. The Drive `objects/commits` backup directory alone is not the unpacked experiment root.

No CATCH checkpoint, native CATCH replay, EXP013 graph/fusion/removal checkpoint, optimiser state or detailed P01 element audit is needed for this additional comparison. Those remain original evidence gaps for an end-to-end audit of all EXP013 work; omission is explicitly scoped and not an assertion that the whole archive was reproduced.

If missing/corrupt input is discovered, `cache/MISSING_OR_INVALID_INPUTS.json` states the exact path, reason and full contract. There is no fallback to regenerated weights, altered architecture, new training or changed thresholds.
