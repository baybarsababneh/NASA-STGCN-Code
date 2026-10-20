# Should the frozen GCN use Hundman's released detector?

**Independent follow-up, 16 October 2026.** Analysis ID: `HUNDMAN_NATIVE_DETECTOR_CROSSOVER_V1`. This supplements `LSTM_RELEASE_BENCHMARK_AUDIT_V1`; it does not revise EXP001–EXP015 or the earlier audit.

**Yes, as a supplementary robustness comparison. It is useful, but does not justify replacing the existing common-detector analysis.** The appropriate test applies the same released detector to the GCN, controlled LSTM and actual author-released LSTM. Applying it only to the GCN and comparing against an LSTM processed differently would confound the forecaster and detector. Copying the LSTM's numerical thresholds would also be inappropriate: the adaptive rule must calculate thresholds from each forecaster's own residuals.

I completed this test using frozen forecasts, with no training, model inference, threshold tuning or historical-result changes. The released LSTM retains higher corrected event scores and strict point F1 on both missions. The GCN's apparent MSL improvement under the native detector is entirely associated with a historical C-2 startup exception; it adds no within-event notifications.

## What the existing comparison establishes

The earlier audit already compared the actual released LSTM forecasts and project forecasts under one detector, one evaluator, the same telemetry units and identical target support. That is a valid controlled comparison of the available forecast evidence. The original controlled LSTM remains useful for the narrower project input/training contract, but does not represent the released model's effectiveness, especially on MSL.

The remaining question was whether the project detector's departures from the release caused the unfavourable GCN-versus-released-LSTM result. Completing the detector-by-forecaster comparison answers that question directly. My earlier assessment that native processing of GCN forecasts was unnecessary was limited to establishing the recreated-baseline gap. This follow-up warrants the additional sensitivity test; it does not invalidate the earlier common-detector results.

## What was executed

The protocol was sealed before new scoring. The pinned author implementation is `telemanom/errors.py` at commit `2e6c5b6c3558e7835601519b7bdef37c649bdbdc`, whose released-LSTM intervals had already matched all 81 official CSV records. This is a reproducible **author-release detector**, not a claim to reconstruct the printed 2018 table exactly. Earlier source versions, the paper's algorithm description and its printed metrics differ.

- Newly processed 486 frozen GCN/controlled-LSTM streams: 81 channels × three saved seeds × two forecasters. Reused 567 prior common-detector streams and 81 verified native released-LSTM streams. The combined table contains 1,134 decision streams, not 1,134 independent fits.
- Retained raw TEST support `250 <= t < N-10`, the frozen 81-channel roster, first P-2 label policy, inclusive label conversion and released telemetry units. No alignment search or label-based selection was performed.
- Retained native 70-point batches, 30-batch windows, EWMA span 105, threshold grid 2.5–11.5 by 0.5 with fallback 12, inverse-error branch, telemetry-scale guards, 100-point buffers, pruning `p=0.13`, startup behaviour, singleton removal and final-batch rules. Each stream's thresholds were computed from its own unlabelled residuals.
- Supplied the same original ten-target **observed telemetry** matrix to the native scale guards for every role. Only the first-column scalar forecast residual is scored; ten-step GCN predictions are unnecessary. The guards require nine additional observations. Reconstructed issue times charge this availability equally; they are not archived operator notifications.
- Evaluated every forecaster with the frozen project metric, paper-style unpenalised event overlap, strict point metrics without point adjustment, nominal burden, residual average precision and issue-time diagnostics. The release's literal event-counting method was also executed separately. Its first-overlapped-event convention differs from the paper-style rule, but changes no counts in these 1,134 streams.

The original code needs an integer-index cast in `np.delete` under the current numerical environment. A separately recorded execution amendment memoizes exactly identical inputs and skips zero-standard-deviation threshold searches that cannot replace the initialized fallback. Fourteen unwrapped reference checks agree exactly, and all 486 new decision masks agree with the original `Errors.process_batches` control flow using this equivalent execution path. This is not a detector safeguard or a change to statistical decisions. The initial slow run was stopped after two streams; the unchanged protocol and amendment are retained. Four streams emit native numerical warnings; none fails or receives a substitute detector.

Independent verification recounts all 1,134 saved masks, event overlaps, point counts and notification counts; checks all 28 mission/seed aggregates and 12 mission means; and confirms that all 1,415 manifest-listed files in the earlier audit remain unchanged.

## Verified results

Project results pool channel counts within each saved seed and then average the three seed metrics. The released reference has one saved fit per channel.

| Detector | Forecaster | MSL corrected event F0.5 | MSL strict point F1 | SMAP corrected event F0.5 | SMAP strict point F1 |
|---|---|---:|---:|---:|---:|
| Existing common adaptation | Released LSTM | 0.843424 | 0.431214 | 0.833718 | 0.334602 |
| Existing common adaptation | GCN | 0.769346 | 0.342594 | 0.718832 | 0.180842 |
| Existing common adaptation | Controlled LSTM | 0.707107 | 0.360485 | 0.814321 | 0.327769 |
| Pinned native release | Released LSTM | 0.851975 | 0.439303 | 0.833845 | 0.330725 |
| Pinned native release | GCN | 0.783924 | 0.352086 | 0.718831 | 0.180210 |
| Pinned native release | Controlled LSTM | 0.721478 | 0.369506 | 0.814320 | 0.322898 |

The native detector narrows the GCN's MSL corrected-event deficit slightly, from 0.07408 to 0.06805. Its SMAP deficit is essentially unchanged. Unpenalised native event F0.5 also favours the release: **0.892857 versus GCN 0.807981 on MSL; 0.847222 versus 0.726548 on SMAP**. Thus the conclusion does not depend on the project's nominal-occupancy correction.

### Why the MSL increase is not evidence to prefer the native detector

Every change in event hit/miss/wholly nominal-run counts is on **C-2**. The native source exempts C-2 from startup smoothing, with the explicit comment “anomaly occurs early in window.” This is benchmark-specific prior anomaly knowledge, even though no labels were newly consulted during replay. Each role gains one overlapped event and additional nominal flagged samples. It gains **no additional within-event notification**.

Excluding C-2, native and adapted masks are exactly equal for all 182 remaining MSL role/seed streams. Their corrected scores are identical: release **0.854274**, GCN **0.783159**, controlled **0.717712**. The native GCN increase therefore supplies no evidence of a general detector improvement. The C-2 exclusion is a diagnostic sensitivity, not the selected headline result. The alternative P-2 labels likewise leave the SMAP release lead intact (native release 0.833943, GCN 0.718909, controlled 0.814410).

### Burden and timing remain consequential

| Native-detector endpoint | MSL release / GCN / controlled | SMAP release / GCN / controlled |
|---|---|---|
| Nominal occupancy | 6.07% / 4.38% / 5.93% | 1.94% / 1.40% / 1.84% |
| Within-event issued-flag rate | 13.89% / 13.89% / 13.89% | 66.18% / 37.75% / 57.84% |

The GCN retains lower nominal occupancy and lower event recall. Native timing charges the ten-target guard's nine extra observations; the common adaptation does not require that lookahead. Therefore timing differences describe complete information requirements as well as alarm logic. High retrospective event overlap is not evidence of timely spacecraft monitoring. Nominal samples are outside annotations, not independently certified healthy operation; occupancy is not an operational false-alarm rate.

### Conditional uncertainty

The predeclared 10,000 paired-channel resamples use RNG seed 20261016, preserve all project seeds within each channel and keep the single released fit fixed.

| Native GCN minus released LSTM | Difference | Descriptive 95% interval |
|---|---:|---|
| MSL corrected event F0.5 | -0.06805 | [-0.19269, +0.02367] |
| SMAP corrected event F0.5 | -0.11501 | [-0.20488, -0.02768] |
| MSL strict point F1 | -0.08722 | [-0.18293, -0.00484] |
| SMAP strict point F1 | -0.15051 | [-0.19865, -0.10275] |

These intervals condition on the saved fits and potentially dependent channels. They are descriptive and do not resolve shared-incident dependence or release fit-to-fit variability. Wider 98.75% intervals are exported; the MSL point-F1 interval then includes zero. No multiplicity-adjusted or noninferiority claim follows from the 95% intervals. The MSL event interval crossing zero establishes neither equality nor competitiveness within a prespecified acceptable margin.

## Scientifically preferable benchmarking strategy

1. **Keep the common adaptation as the principal frozen-artifact control.** It already gives all forecasters identical rules, without C-2's channel-specific exception, and preserves continuity with EXP012/EXP014. The actual released LSTM must remain in this comparison; the recreated baseline alone is insufficient.
2. **Add the native-release comparison as a separate sensitivity.** Apply it to all forecasters, as completed here, and retain both favourable and unfavourable outcomes. Do not select a detector separately for each forecaster or promote the best score after inspecting TEST.
3. **Retain unpenalised event overlap, strict point metrics, burden and timing.** Using native detection but only the project's corrected event endpoint would not fully address the concern about the original evaluation requirements. These secondary views agree about the release's point-estimate lead.
4. **For a future operational claim, use new calibration and evaluation data.** Identical adaptive rules do not enforce an identical false-alarm budget. A prespecified burden-constrained comparison on independent, out-of-fit nominal calibration data would answer a different and stronger operational question. Copying thresholds, retrospectively sweeping NASA TEST labels, or calibrating the released model on TRAIN portions it may already have fitted would not supply that evidence. Matched training inputs, temporal splits and repeated fits would be needed to isolate architecture, rather than to complete this frozen-artifact test.

## What changes and what remains unresolved

This extension closes a specific detector-confounding concern: the GCN's deficit against the released LSTM persists when both use the released decision rules and when occupancy penalisation is removed. It reinforces the earlier restrained conclusion. The released LSTM leads by point estimate on both missions; SMAP shows a substantial conditional deficit. MSL event performance remains uncertain, with no demonstrated equivalence. The lower GCN nominal burden is a trade-off, not proof of equal detection effectiveness.

The extension does not explain the causal source of the released model's advantage. Lookback, selected versus full command inputs, fitting exposure, objective, gates, initialisation and scaling remain unequal. Historical TEST exposure, inherited TEST-extrema normalisation, unavailable original training histories and lack of independent operational calibration remain unresolved. These limitations were not repaired by changing the detector.

Only a short results subsection, one conclusion sentence and a pinned source-reference update are warranted. `Report_Updates.tex` provides exact paste-ready text. The report, figures, historical scores, previous audit and EXP001–EXP015 remain unchanged. No new paper is needed for this detector follow-up.
