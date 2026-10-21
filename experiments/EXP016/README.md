# EXP016 — ST-GCN / Hundman author-released LSTM comparison

This directory preserves the EXP016 scientific source code, Colab runner, training/detection protocol, dependency pins, tests, third-party notices, and **selected results from the completed A100 evaluation**. It extends EXP001–EXP015 without changing those experiments.

**Primary finding:** The released LSTM scored higher than the newly trained ST-GCN on both missions under the uniform author-derived detector (standard event F0.5: MSL 0.882353 versus 0.637445; SMAP 0.847222 versus 0.718982). These retrospective comparisons cannot isolate architectural effects because validation methods and original-training information differ. See [`results/RESULTS.md`](results/RESULTS.md), [`results/FINAL_AUDIT.json`](results/FINAL_AUDIT.json), and [`PROTOCOL.md`](PROTOCOL.md).

## Public checkout versus executed package

This is a **source-only public release**, not the byte-identical original `EXP016_Package.zip` that was used in Colab. To avoid redistributing the third-party NASA/telemanom dataset and pretrained model archive without checking its redistribution terms, `inputs/archive.zip` has been excluded. Large ST-GCN trained weights, per-channel forecasts, score masks, and the complete run-results archive are also not included. The original Colab runner validates the complete package by SHA-256 and therefore **cannot be executed from this checkout alone**. It requires the matching original complete package, including that archived input data. Do not remove the checksum gate or substitute another archive without independent verification.

- **Executed notebook:** [`EXP016_Run_All.ipynb`](EXP016_Run_All.ipynb)
- **Scientific implementation:** [`exp016/`](exp016/), [`config.json`](config.json)
- **Tests and recorded preparation evidence:** [`tests/`](tests/), [`evidence/`](evidence/), [`VALIDATION.md`](VALIDATION.md)
- **Training/scoring protocol:** [`PROTOCOL.md`](PROTOCOL.md)
- **Released-detector source and licence:** [`vendor/`](vendor/), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
- **Selected actual final evaluation records:** [`results/`](results/)
- **Original full-package documentation and hash manifest:** [`ORIGINAL_PACKAGE_README.md`](ORIGINAL_PACKAGE_README.md), [`ORIGINAL_PACKAGE_MANIFEST.json`](ORIGINAL_PACKAGE_MANIFEST.json); these describe the complete archived package, **not** this reduced public checkout.

## Completed experiment

Three ST-GCN seeds (17, 29, 43) were fitted per channel: **246 jobs across 82 channels**, with **81 annotated channels scored**, no recorded detector failures and no primary degenerate branches. A new LSTM was not trained. See the final audit for machine-produced completion status and the results summary for the limitations of the comparison.

The model outputs were archived outside this Git checkout; reporting a successful run does not mean the full per-fit archives are present here.
