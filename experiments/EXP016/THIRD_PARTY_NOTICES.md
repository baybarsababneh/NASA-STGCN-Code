# Sources and redistribution

The telemanom source files under `vendor/` are from
https://github.com/khundman/telemanom at commit
2e6c5b6c3558e7835601519b7bdef37c649bdbdc. Its Apache-2.0 licence is preserved
as `vendor/TELEMANOM_LICENSE.txt`. Original files remain unchanged; documented
execution compatibility changes are made by `exp016/detector.py` in memory.

The supplied `archive.zip` and annotation CSV are included for the user's
private experiment. They contain NASA benchmark arrays and the author release's
weights, forecasts and logs. This package does not grant new redistribution
rights or establish NASA affiliation. Preserve original source/licence terms
before publishing data or models.

The inference-only NumPy replay function adapts the user's public repository
followup `released_lstm/verify_sources.py`, pinned to
fc9554c6463cf882acfdd90b001e0a248680aa68. Graph architecture lineage is documented
in PROTOCOL.md. Prior experiment files are neither modified nor required at run time.

Reference: Hundman et al. (2018), Detecting Spacecraft Anomalies Using LSTMs
and Nonparametric Dynamic Thresholding, KDD 2018, arXiv:1802.04431v3.
