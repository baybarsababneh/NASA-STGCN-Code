# Evidence review and design judgement — EXP013

Reviewed 6 October 2026. This document separates published ideas, recovered project evidence and new hypotheses. No claim of improvement is made before the locked run.

## What the history actually supports

The original research objective concerned tuning a spatiotemporal GCN on the supplied NASA data. The recovered EXP005/EXP007 provenance, EXP008 weight verification, EXP009–EXP011 detector limitations, EXP012 protocol, upstream adaptation notes, selected graph/model code and final sealed evaluation were inspected. Earlier detector failures remain part of the evidence: static residual thresholds were not established as feasible, sustained faults could be absorbed by the forecaster, and a deployable fusion detector was not demonstrated.

The old GCN is already a **command-to-telemetry** model, not a physical graph connecting separate NASA files. It has a persistence-plus-residual forecast head. MSL uses a short RF7/h32 model and selected incoming command support; SMAP RF31/h64 and broader support. The controlled LSTM is a 64-history one-step PyTorch adaptation using the selected command contract; it is not the original ten-output Keras model. This matters when interpreting historical comparisons.

The completed EXP012 results support a limited graph contribution on MSL under that architecture/detector. They do not establish universal graph superiority: point F1 is below the LSTM on MSL, and on SMAP both the removal arm and controlled LSTM have higher corrected event F0.5. Point recall is particularly weak for the GCN. Improving prediction MSE or event overlap alone would fail the new objective. The exact historical tables and bootstrap are copied unchanged into the bundle.

My inference is that the next useful question concerns **discrimination under command context**, rather than another forecast-capacity sweep. An autoregressive reference can follow a sustained fault; a better normal forecast can also reduce some benign errors without improving the separation between benign and faulty residuals. Learning more edges does not by itself solve either problem.

## Primary literature and how it changes the experiment

| Work | Primary source | Useful evidence/idea | Decision in EXP013 |
|---|---|---|---|
| Hundman et al., KDD 2018 | [Paper](https://arxiv.org/abs/1802.04431), [official NASA/telemanom](https://github.com/khundman/telemanom) | One target plus encoded command-module activity; forecasting and adaptive error thresholding | Preserve the historical controlled LSTM; model command context explicitly; audit issue time rather than overlap alone |
| Deng & Hooi, GDN, AAAI 2021 | [Official implementation](https://github.com/d-ailin/GDN) and linked paper | Learned embedding graph, sparse top-k neighbours and feature error scoring | Run official GDN with declared NASA adapter; use a sparse learned command message in the new architecture without interpreting edges as physics |
| Lai et al., STGLR, Sensors 2025 | [Full paper, DOI 10.3390/s25020310](https://pmc.ncbi.nlm.nih.gov/articles/PMC11769452/) | Learned graph, GraphSAGE/temporal attention and variational reconstruction | Review ideas; do not claim faithful reproduction without located official code and resolved implementation details |
| CATCH, ICLR 2025 | [Paper](https://arxiv.org/abs/2410.12261), [official implementation](https://github.com/decisionintelligence/CATCH) | Frequency-dependent channel relationships and frequency reconstruction can expose structure a time-domain predictor misses | Include native time+frequency CATCH, preserving published model/loss rather than inventing a small “CATCH-like” model |
| Su et al., OmniAnomaly, KDD 2019 | [Official implementation](https://github.com/NetManAIOps/OmniAnomaly) and linked paper | Stochastic recurrent latent modelling and reconstruction probabilities | CATCH is the agreed alternative; do not call an unvalidated modern VAE a faithful OmniAnomaly |
| Carmona et al., NCAD | [Paper](https://arxiv.org/abs/2107.07702) | Synthetic outlier exposure and contextual anomaly discrimination can train detectors beyond nominal prediction loss | Use balanced synthetic supervision and contextual telemetry swaps; this new detector is inspired by the idea, not an NCAD reproduction |
| Kim et al., rigorous time-series evaluation | [Paper](https://arxiv.org/abs/2109.05257) | Point adjustment can inflate apparent performance and conceal weak discrimination | Keep strict point metrics, AP tie handling, false alarms, real notification timing and explicit interval support; no point adjustment |

NASA's released arrays are not aligned multivariate telemetry collected simultaneously across entity files. A within-file GDN/CATCH variable axis contains the telemetry target and command dimensions. This differs from their common multivariate sensor interpretation. Pooling only complete windows is a controlled adaptation; joining file boundaries into a continuous trajectory would be an avoidable error. Neither published-model success nor a learned graph on these data establishes a spacecraft physical topology.

## Fidelity decisions that prevent a misleading leaderboard

Official GDN still targets an old Torch/PyG stack. We retain its mathematical architecture and native forecast loss using two declared API updates verified against an independent dense forward/gradient implementation. Its released normalization helper uses TEST error moments. Using that helper would violate the new protocol, so we preserve the score structure but fit moments on nominal CAL. This is a disclosed adapter, not a literal reproduction of every upstream evaluation step.

Official CATCH source is pinned by commit and archive hash. Its large mission-specific models, native frequency objectives, RevIN, mask generation and unusual mask-optimizer ordering are preserved. Its offline all-window/block scoring can use future values relative to interior points. We score only the right edge of a past-inclusive window. Stochastic masks are retained with deterministic case/channel seeds. Common splits, pooled entity training, causal evaluation and TRAIN thresholds remain deviations from its published evaluation. The new run cannot be compared numerically to its paper's headline table without these qualifications.

STGLR's paper provides equations and several hyperparameters, but its data-availability statement points to NASA/telemanom rather than an implementation. Searches for the exact method/acronym and author-associated code did not locate official source. Unspecified details of latent/decoder structure, attention, graph construction and threshold estimation would require choices that alter the method. Its point-adjusted evaluation is not commensurable with our strict scores. It is a **documented reproduction gap**, not a baseline failure or a claim that no code exists anywhere. No fabricated STGLR result will appear.

OmniAnomaly's official legacy TensorFlow/TFSnippet environment is outside the modern single Colab panel. An isolated legacy container would add engineering burden and another environment; a Torch approximation would weaken fidelity. Choosing CATCH satisfies the requested CATCH/OmniAnomaly alternative while keeping the workflow focused. No additional imitation is trained.

Historical LSTM results retain the original EXP012 data/support, metrics, adaptive detector and batch notification timing. They are an honest complete-pipeline reference; they do not become an architecture-only matched comparison simply by copying their table. The new exact removal and direct-fusion controls provide that more focused mechanism test, with capacity differences disclosed.

## Why the proposed architecture is focused

The command-only normal reference removes the reference's telemetry persistence shortcut. A discriminator then combines observed telemetry, context and residual/dynamics summaries. It is trained for fault discrimination, not ranked only by nominal MSE. The learned sparse message can discard misleading command associations; the null message avoids forced mixing, and FIT-only variation masks remove inactive command slots. No graph across unaligned entity streams is introduced.

The exact command-path removal tests whether context contributes at all. Direct command fusion tests whether a graph is necessary for any context benefit. Both use the same corruption/data schedule and scoring/detector procedure; graph/removal initialization is byte-identical. A richer fusion control may beat the graph; that would be useful evidence rather than a failed experiment.

Shared per-mission modelling reduces fits from hundreds per architecture to a few while exposing the model to varied nominal dynamics. Channel embeddings and per-entity FIT scales retain identity. This is an inductive choice, not guaranteed superiority over per-entity training. New training uses balanced channel sampling; published pooled loaders keep their native losses/budgets and are not silently made into supervised fault classifiers. Per-channel window support is reported. The short D-12 FIT segment cannot provide a 192-point CATCH window; the shared MSL model is trained on other entities without a per-channel sequence substitution.

Synthetic cases cannot stand in for real spacecraft anomalies. Their locations, amplitudes and forms are limited, contextual swaps may be physically implausible, and benign TRAIN can contain unlabelled faults. Confirmation introduces two new families and separate nominal alarm assessment, but the temporal segment has been historically exposed and many magnitudes overlap training. We report invalid cases and pre-existing background alarms. A positive synthetic result supports only the injected mechanisms.

Six threshold/smoothing candidates are enough to test sensitivity versus alarm burden without a large detector search. No TEST calibration, test-labelled threshold search, postprocessing rescue or best-seed selection is permitted. The detector might fail the nominal gate; diagnostic fallback is labelled as failure, not promoted.

## What can and cannot be demonstrated now

This release can establish a reproducible mechanism comparison, a new locked retrospective NASA result, a controlled official-code GDN/CATCH comparison and a transparent audit. It can also establish that the proposed graph offers no useful gain or cannot meet nominal alarm constraints.

The preserved historical report ZIP contains label-derived outcomes and event metadata. It is retained for integrity and unchanged historical comparisons, not a claim of physical label isolation. New fitting/selection/scoring never parse its event intervals. The score-seal gate protects the new annotation-CSV parsing step, while the declared historical exposure remains.

It cannot turn repeatedly exposed NASA TEST data into independent evidence. The official release also inherits TEST-min/max scaling. An apparent improvement on both missions is therefore exploratory, even with positive descriptive intervals. An independent real-anomaly evaluation would require a genuinely unused dataset or new spacecraft collection, a separately frozen transfer protocol and sufficient operational nominal coverage. Adding such a dataset now would be a separate research scope with acquisition/preprocessing costs; it is not simulated by calling the last TRAIN tenth a virgin holdout.

Published scores under point adjustment, different feature preprocessing, optimal TEST thresholds or offline future context are literature references only. The notebook outputs its own full-roster metrics and fidelity status; it never merges incompatible published headline numbers into its result table.
