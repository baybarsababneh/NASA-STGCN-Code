# Dataset

- [SMAP/MSL benchmark download](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl)
- [Telemanom original implementation and data notes](https://github.com/khundman/telemanom)
- [Original research paper](https://arxiv.org/abs/1802.04431)

**The full telemetry archive, pretrained HDF5 models, arrays and restored experiment inputs are not included in ordinary Git.** Five separate artifact archives hold the preserved inputs and selected run states. Do not publish those archives without separate redistribution clearance.

The small `labeled_anomalies.csv` used by the experiments is retained with original source attribution. Preserve historical first-entry treatment of duplicated P-2 and the exclusion of unlabelled T-10.

The model graph uses command context rather than verified inter-channel synchronisation. Forecasting TRAIN-only splits and final scoring seals vary by experiment. Reuse each experiment's own saved manifests; do not feed TEST labels into model or threshold selection.
