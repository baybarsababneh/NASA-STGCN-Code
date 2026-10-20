# EXP009 diagnosis and EXP010 decision

## Decision

Keep EXP009 closed as a valid negative result. Run **EXP010_MECHANISMS_V1**, a bounded inference-only study of residual mechanisms and end-to-end telemetry perturbations. Do not choose a detector, expand the multiplier grid, change alarm limits, or reopen forecasting architecture development.

The strongest justified next question is: **Which aspects of the residual stream represent forecasting adaptation, routine telemetry activity, calibration degeneracy or sustained unexplained change, and how faithfully did residual-only injections represent sensor disturbances?**

A new detector family is worth considering, but the evidence does not yet justify selecting its parameters. The leading hypothesis is a causal context-conditioned reference model that explicitly handles quiet/active regimes and point masses, coupled to evidence with controlled persistence. That is a hypothesis for later testing, not a promoted method.

## Integrity findings

I independently verified the 4,194 committed unit hashes and the output manifest. The archive contains **936 pre-E01 commits**, all byte-preserved. The earlier '900' was a progress message, not the exact number present at interruption. The executed E01 notebook's code cells match the supplied E01 notebook. The original outcome remains NO_FEASIBLE_CANDIDATE for both missions. No historical result was changed.

## What failed, with evidence

### 1. Incremental sensitivity mixes detectability with background alarms

For MSL absolute_h4, the high-amplitude spike incremental hit rate was about 78.1%. For 32-sample shifts and ramps, **56.2% of unmodified event windows already had an alarm**. Under the frozen incremental-hit definition, even a detector that detected every injection in every remaining window could achieve only **43.8%** incremental recall. The required 50% gate was mathematically unreachable at that setting.

The observed high-amplitude ramp hit rate was 42.0%, or about 95.8% conditional on a background-clean window. This does not rescue the candidate: it was noisy and failed its frozen criteria. It does show that 'all MSL candidates failed sensitivity' must not be translated into 'the GCN residuals contain no anomaly information'.

EXP009's conservative definition was implemented as specified. It is not a software defect. Future analyses should report background overlap, unconditional injected-event hits, conditional hits on clean windows and incremental hits separately. Changing this reporting does not retrospectively alter the EXP009 decision.

### 2. The residual reference is not one smooth stationary distribution

Examples below are seed 17 in fit-standardised telemetry units, not physical sensor units.

| Channel | Observed structure | Implication |
|---|---|---|
| MSL M-4 | Telemetry-change frequency rises from 14.4% of reference samples to 54.8% of selection samples. Residual MAD rises from 0.00118 to 0.109. | Strong change in activity and error spread. A threshold based on the earlier dominant quiet regime transfers poorly. |
| SMAP G-3 | All 262 reference residuals are exactly zero. Selection has only two nonzero residuals out of 132, both magnitude 2 at telemetry changes. | Point mass plus rare transitions. A smooth central-scale model is inappropriate without an explicit degenerate-reference policy. |
| SMAP G-7 | Reference and selection both have near-constant residual centres with rare very large errors. Their RMS values are similar, about 1.23 and 1.28. | Not every failure is reference-to-selection drift. Persistent mixture structure and event ordering also matter. |
| SMAP F-3 | Median residual drifts from 0.000276 to 0.000303 and then 0.000317. Persistence residuals remain near 0.00200. | Forecasting is accurate, but a narrow error distribution makes small model-bias drift look extreme. This is not evidence of a meaningful physical anomaly. |
| MSL T-8 | A reference-block residual spike reaches about 7.12, while selection residual magnitude never exceeds about 0.101. | Alarm carry-over and temporal scoring must be inspected before interpreting later flagged occupancy as new instability. |

The data support heterogeneous mechanisms. They do not prove which samples are operationally normal: TRAIN is presumed nominal, not independently certified. Nor do they establish that commands explain the changes. EXP010 measures causal command context rather than assuming it.

### 3. Robust central scales and smoothing interact badly with degenerate references

Five of 27 MSL channels and 12 of 54 SMAP channels hit the residual-scale floor in this development reference. On G-3, a zero reference also produces zero score IQR. The two numerical floors then imply an extremely small effective raw-error threshold. An isolated error can leave EWMA above that threshold for many samples. **EWMA span 10 does not mean memory is limited to 10 samples.**

For ewma10_h8, G-3's seed-averaged flagged occupancy is about 92.4%, despite only two nonzero residuals in the seed-17 selection trace. At multiscale_h16 it is about 30.3%. The bounded 32-sample window behaves differently from the asymptotic EWMA tail.

This is a scientific limitation of the frozen calibration/scoring design, not a reason to issue another engineering-only patch. Arbitrarily increasing the floor would itself be a new detector choice.

### 4. Mission summaries hide strong concentration

At SMAP multiscale_h16, G-3, G-7 and F-3 contribute about **86.0%** of the sum underlying equal-channel mean occupancy. At MSL multiscale_h16, M-4, T-8 and T-9 contribute about **95.1%**. These are explanatory concentration diagnostics, not permission to exclude channels or select these settings.

### 5. Short exposure limits interpretation of alarm-rate constraints

The canonical selection segments have median length 108 samples for MSL and 142 for SMAP. A single start corresponds to 9.26 and 7.04 starts per 1,000 respectively. Their minima are 22 and 16 samples. This makes per-channel rate estimates coarse and highly variable.

Even under an illustrative independent Poisson model, zero events in 108 samples gives a one-sided 95% upper rate near 27.7 per 1,000, far above 2 per 1,000. Actual dependence makes that simple model only illustrative. A zero count therefore cannot certify the research target. Conversely, short exposure does not excuse genuinely high observed occupancy.

## What should happen to the operating constraints?

Keep the EXP009 limits and failed decisions unchanged. Do not relax limits until something passes. For a later detector protocol, distinguish:

- desired operating cost, which needs a defensible unit of exposure, alarm duration/latency requirements and ideally domain input;
- the empirical alarm burden observed on these short presumed-nominal segments;
- statistical certification, which this dataset cannot provide per channel at such low target rates without additional assumptions or exposure.

Equal-channel summaries and sample-weighted burden answer different questions and should both be reported. Pooling seeds does not create more independent temporal exposure. A future change to the budget requires an explicit operational justification and a new protocol, not reuse of NASA TEST outcomes.

## Why end-to-end perturbations are useful without a passing detector

EXP009 added disturbances to residuals while holding predictions fixed. A real telemetry corruption enters subsequent forecasting windows. A persistence-like forecast can absorb an offset, turning a sustained sensor error into an onset/recovery residual rather than a sustained residual. A stuck-value injection can also be a null intervention if the original signal was already constant.

EXP010 compares actual reforecast residuals with the residual-only counterfactual on the same unchanged nominal trace. It measures the information available to any downstream residual detector and preserves all failed EXP009 policies as diagnostic instruments only. It is neither another detector contest nor evidence about real NASA anomaly recall.

## What is not supported yet

- A claim that the final GCN is invalid or should be retrained.
- A claim that EVT, conformal methods, CUSUM or a classifier will fix these mechanisms automatically.
- A claim that all nominal alarms are false alarms, or that all injected disturbances are realistic spacecraft faults.
- A retrospective pass for any EXP009 candidate.
- A final NASA TEST evaluation at this stage.

## After EXP010

If causal command/activity context separates nominal burst regimes consistently across the mechanical cohort and seeds, that motivates one bounded context-conditioned calibration study. If prediction adaptation removes sustained evidence, any proposed detector must address onset/recovery sensitivity explicitly rather than merely smoothing residuals longer. If nominal and injected responses remain indistinguishable with the available information, seek independent labelled development examples or additional operational context. Do not continue an unbounded residual-threshold search.

These next decisions must use the complete response tables, null-intervention counts and cohort separation. No detector is automatically promoted by EXP010. The independent reviewer should receive this diagnosis and EXP010 as an addendum, preserving the existing report and results.
