# Audit checklist and reviewer addendum

## Before executing

Verify MANIFEST.json; config parent hash; all 534 EXP008 replay/marker hashes, identities and numerical-replay assertions; finite arrays and target-index boundaries. Package excludes raw data, TEST and labels. Confirm no model training or weights loading imports. Confirm corrected EXP008 metric module matches its original source hash. Run both synthetic test suites.

## After executing

1. Recompute OUTPUT_MANIFEST.json. Check 2,916 unique selection units and <=1,917 audit units. Reconcile every unit with its commit marker and run hash; there must be no extra, duplicate or missing planned units. The total is 4,833 if both missions select a candidate; fewer is legitimate only through recorded no-selection branches.
2. Independently regenerate ranking and selected mission configuration from selection CSVs. Verify channel-then-mission weighting, seed gates, tie-breaking and high-amplitude sensitivity floor. Confirm DETECTOR_CANDIDATE_FREEZE precedes auditing in source control flow. Timestamp ordering is supporting evidence, not the sole proof.
3. Recompute canonical and separate R1/R2 audit gates, and final status. Audit failure must never trigger fallback. Verify all three GCN seeds are reported and persistence is not replicated as three independent runs.
4. Verify threshold/reference boundaries, strict comparison, causal state, no merging, floors, and common additive stress amplitude across model controls. Inspect short-block duplicate placements and floor incidence. Variance probes are explicitly model-relative diagnostics.
5. Recompute selected-vs-reference paired intervals; inspect influential channels, family concentration and per-seed distributions without revising selection. No significance claim from synthetic sample counts.
6. Check the executed notebook, environment freeze, isolated requirements, per-unit resume validation, and any interruptions. No source change without a new engineering/scientific version.
7. Preserve historical AP outputs. Future AP uses tied-threshold implementation EXP008_METRICS_V2_1. EXP009 does not evaluate NASA anomaly labels.

## Reviewer package update

Add `addenda/EXP009_DETECTOR_V1/` rather than replacing the reviewed forecasting study. Include the protocol, source/config/manifest, EXP008_INPUT_AUDIT.json, executed notebook, final decision, ranking/gate tables, all per-channel nominal and stress tables, uncertainty tables and output manifest. Large per-unit records can be a separate evidence ZIP.

Do not change EXP001–EXP008 results or the final GCN architecture freeze. The report may add a detector-development methods/results subsection only after auditing EXP009. Clearly label synthetic residual findings and nominal burden. Do not insert synthetic recall into a NASA literature-performance table or imply detection improvement before real-anomaly evaluation.

## Expected substantive interpretation

A pass supports a bounded detector calibration direction under reused TRAIN conditions. A failure is evidence that the requested sensitivity/burden trade-off is not established with this candidate family and data. Neither justifies reopening architecture search or tuning on NASA TEST. Lack of independent anomaly-development labels remains the main limitation; domain-expert or external labelled development data would be more valuable than an unbounded detector search.
