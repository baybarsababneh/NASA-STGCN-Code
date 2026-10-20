# NASA SMAP/MSL report revision audit

Audit date: 7 October 2026

## Delivered report

The complete, standalone LaTeX source is `NASA_SMAP_MSL_Anomaly_Detection_Report.tex`. It uses standard TeX Live packages and an inline bibliography, with no external figures or bibliography files. The Overleaf bundle contains a byte-identical copy named `main.tex`, the compiled PDF and this audit record.

## Scientific and editorial changes

- Reframed the report around anomaly detection, with forecasting as one source of anomaly evidence. Condensed the early experimental history into its methodological implications and moved exhaustive provenance and historical numerical tables into appendices.
- Replaced outdated primary benchmark descriptions with the completed, frozen EXP012 Hundman-derived evaluation. Retained all five model arms and both missions, corrected event and strict point metrics, nominal alarm burden, notification timing and paired channel-bootstrap contrasts. Kept raw-q99 and historical published figures clearly separate.
- Discussed the positive MSL event contribution and adverse SMAP graph contribution, including the MSL LSTM advantage in strict point F1 and the delayed notifications concealed by retrospective overlap metrics.
- Described the implemented EXP013 command-reference architecture, synthetic discrimination objective, chronology, calibration, controls and official GDN/CATCH adaptations. Disclosed STGLR/OmniAnomaly fidelity gaps and differences in information access. Included no EXP013 scientific results, unfinished findings, reserved empty sections or overall conclusion.
- Made historical TEST exposure, inherited TEST-based scaling, missing cross-stream alignment, retrospective uncertainty and implementation-audit limitations explicit. Attributed published ideas and code and included a factual AI-assistance statement without inventing an institutional policy.

## Verification performed

- Verified all nine report payload hashes against the EXP012 completion receipt and the original frozen report ZIP.
- Checked 2,108 arithmetic identities using 2,106 frozen channel/detector records, 52 mission/seed rows and 20 mission rows. These checks use exported counts and notifications; they do not rerun scoring, models, thresholds or bootstrap sampling.
- Generated numerical tables directly from frozen CSV/JSON records. Transcribed stored bootstrap intervals and checked their difference estimates against the frozen mission means.
- Inspected the current EXP013 protocol, notebook, bundle and build audit. Confirmed their bytes match the existing audited release. In particular, the discriminator exports raw logits; sigmoid is used inside the loss, not before score calibration.
- Compiled with pdfLaTeX through latexmk. The 24-page A4 PDF has no compilation errors, unresolved references/citations or overfull boxes. All 16 bibliography keys and 10 referenced internal labels resolve.
- Visually inspected every rendered page and the main result tables. An additional PDF geometry check found every extracted word box within its page boundaries.
- Confirmed that the original attached source remains unchanged locally. Source and evidence hashes are recorded in `REPORT_EVIDENCE_LEDGER.json`.

No training, NASA TEST inference, label evaluation, detector retuning or new scientific resampling was performed for this report revision. Frozen experiments were not modified. Existing EXP013 CPU build tests were reviewed rather than rerun, and are not presented as model-performance evidence or a completed full-batch GPU run.

## Remaining factual boundaries

1. The supplied annotation file has 82 rows but 81 unique identifiers, including conflicting SMAP P-2 records; T-10 has no label record. EXP012's frozen first-P-2 policy yields 81 streams and 104 supported events. This is not silently reconciled with the original publication's 82 channels and 105 events, and the report does not claim a paper-equivalent roster.
2. The F-scores printed in Hundman et al.'s Table 2 do not equal standard F0.5 values calculated from the accompanying printed precision/recall pairs. The report retains these as printed historical figures and does not substitute corrected publication scores or use them for a direct ranking.
3. The recovered EXP011 material is a summary-level record, not a complete fit-payload re-audit. The report limits its claims to the documented failed feasibility/abstention outcome.
4. EXP013 has implemented and tested methods but no available scientific results. Its required actual full-batch GPU gate and subsequent runtime completion are not certified by this document.

## Reproduction of the document

Upload the Overleaf ZIP, choose `main.tex` and use pdfLaTeX. The project needs no NASA data, checkpoints or experiment bundles to compile. In a local TeX Live installation, run:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The evidence ledger provides file identities and numerical provenance; it is not a new independent review of the underlying research.
