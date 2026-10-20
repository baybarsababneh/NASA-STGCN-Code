# Hundman-style adaptation and verification

Reference: Hundman et al., KDD2018, https://arxiv.org/abs/1802.04431; official https://github.com/khundman/telemanom (`config.yaml`, `telemanom/errors.py`). Exact fetched source bytes/SHA256 are in provenance/INPUT_PROVENANCE.json; sources retrieved2026-10-05. A branch URL alone is not an immutable citation: the recorded byte digest is the pin. The full upstream source is not redistributed here; source comparison was performed locally.

HUNDMAN_ADAPT_V1 independently implements threshold objective, original forward/inverse branch, smoothing, guards, buffering and pruning. Six nondegenerate synthetic streams (1400–4900 samples, full70-sample batches) matched official alarm masks exactly and thresholds at atol/rtol1e-12. Reference execution needed only an integer cast for the upstream `np.delete` row-index bug; reference masks/window thresholds are bundled as synthetic golden fixtures. This establishes tested-domain parity, **not identity on every possible stream**.

Explicit adaptations/safeguards:
- No C-2 exception to initial smoothing. A channel-name exception informed by early anomalies is unsuitable for this common label-blind comparison.
- Absolute original indices provided by the caller; no implicit extra `l_s` shift. All output ranges are half-open.
- Original-style final window includes its remainder. New-target restriction includes every remaining new target, including an irregular final batch; no silently dropped tail targets.
- Streams shorter than70 return an explicit NO_OUTPUT_SHORT_STREAM, zero flags and issue=-1. They are still represented in diagnostics/evaluation denominators, not treated as a detector success.
- Zero residual variance/undefined objective and empty nominal-peak cases cannot produce NaN/JSON crashes: retain fallback threshold12 SD, suppress degenerate-window decisions, empty nominal peak=0. These are flagged through window guards; absence of alarms is not evidence of sensitivity.
- Independent group-index pruning uses integer sets, avoiding the upstream floating-point `np.delete` indices. Inverse-branch nominal peak uses the inverted signal, as upstream does.
- Batch notification availability is exported explicitly. Startup smoothing's up-to500-observation lookahead is charged to issue time. Backward buffering does not get credited as an early notification.
- Original-scale .05 guards are retained across all arms; FIT-standardised errors are used only in the raw_q99 reference.

The controlled LSTM changes original ten-step forecast aggregation to one-step output, uses PyTorch rather than Keras and the common feasible64-sample context/splits/selected command contract. Therefore original published LSTM numbers remain literature references, not controlled results from this implementation. Published defaults80/80, dropout.3, Adam,35epochs, batch64, patience10 and min_delta.0003 are retained; no superiority claim to an exhaustively tuned LSTM is warranted.
