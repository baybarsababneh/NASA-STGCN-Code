# EXP013 final build audit

**Build and CPU engineering: PASS. Scientific NASA experiment: not yet run. Actual A100 full-batch validation: required in the notebook.**

The deliverables implement the complete new experiment: 18 new GCN/control fits, 4 official-code published-model fits, TRAIN-only detector selection, a pre-confirmation study lock, quarantined synthetic assessment, frozen retrospective NASA scoring, score-seal verification before requesting labels, unchanged historical metrics, paired comparisons, decision reporting and a final runtime audit. Completed EXP012 results are reused; no old models are retrained or replayed.

## Verified before delivery

- 24 meaningful synthetic scientific/integrity tests pass with the specified Colab NumPy/pandas/SciPy/scikit-learn/einops/PyG versions and Torch2.6 CPU.
- A complete fictional four-entity pipeline passes: 22 fixture jobs, study freeze, confirmation, 44 score streams, evaluation, result generation, audit, verified stage reuse and corruption blocking before label parsing. Fixture dimensions/epochs are changed only in private test scopes; NASA settings are untouched.
- New-model epoch interruption/resume matches the uninterrupted selected state exactly. Separate GDN and CATCH two-epoch fixture resumes also match; CATCH's carried mask gradients, sampler and RNG states are retained.
- Official GDN forward values, input/embedding/parameter gradients match an independent dense equation oracle after the two declared PyG API changes, for both mission feature counts.
- Official native CATCH mission architectures, losses, backward gradients, Adam allocation and native frequency score are finite on CPU at batch2. Native MSL has210,881,152 parameters; SMAP53,464,208. This is not a published-batch GPU memory test.
- TRAIN-only source staging validates all81 streams, exact historical report hashes and FIT-only new statistics. The20 scale-floor channels and D-12's zero native CATCH FIT-window support are disclosed.
- All162 original NASA NPY payloads in the arrays-only archive retain their byte hashes. TEST payloads were copied opaquely, without loading their float arrays. The annotation CSV/event-interval reports were not parsed during design/build; archived label-derived performance was reviewed. Historical reports include event metadata, so physical isolation from historical label information is not claimed. The historical metric module remains byte-identical.
- Python source and notebook cell syntax, notebook structure, archive extraction and full package manifest are checked at release. Source ZIP pins, thresholds, role/seed/mission rosters, causal support and issue indices are verified at execution.

## Required at execution

Use an A10040GB runtime. There is no80GB scientific gate. The notebook tests both official published mission batches, every new control at batch256 and exact GPU resume with fictional data before permitting research training. This workspace has no GPU, so no A100 fit, replay or full-batch success is claimed here. A failure aborts; it does not authorize altered dimensions, batch size, precision or scoring.

The runtime final audit must pass after evaluation. Build PASS is not scientific success, an independent reviewer endorsement or deployment authorization. Actual runtime, selected epochs, parameter counts, GPU memory and numerical/excluded published coverage are reported by the run.

## Scientific limits carried into every report

NASA TEST labels/results and final TRAIN were historically exposed; this architecture uses that evidence. NASA comparisons remain exploratory retrospective. The released NASA telemetry also inherits TEST-min/max scaling; extra FIT-only transforms do not restore raw-data leakage freedom. Synthetic confirmation supports injected mechanisms only. Event-centred anonymized snippets cannot certify operational false-alarm rates.

GDN and CATCH use official architectures/losses under a declared controlled NASA adaptation, not their published leaderboard protocol. The historical controlled LSTM retains its original detector/timing. No located official STGLR source supported a faithful reproduction; OmniAnomaly is not replaced by a modern approximation. CATCH supplies the agreed alternative. Shared models, command context, one published seed and historical adaptation constrain inference.

Negative results, no feasible detector, missing published coverage, poor confirmation and inconsistency between missions remain reportable outcomes. No post-TEST tuning or winner selection is authorized. Independent real-anomaly confirmation remains outstanding.

`BUILD_AUDIT.json` and the `build_evidence/` files in the supporting bundle record the checks and exact input/source identities. This audit was performed by the build author, not an independent external reviewer.
