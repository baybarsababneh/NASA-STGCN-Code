# EXP006 device amendment — CPU to CUDA execution migration

**Date:** 2026-10-03  
**Parent freeze:** `exp006_pretrain_freeze_v1`  
**Amendment purpose:** execution-environment migration only.

## Rationale

The authoritative EXP006 screening stage completed on the Mac CPU with 288/288 fits and 0 failures. The remaining robustness and confirmation stages are computationally expensive on that backend. This amendment permits deterministic CUDA execution in Google Colab while preserving the frozen experimental specification and the completed screening decision.

PyTorch does not guarantee numerically identical CPU and GPU trajectories even with identical seeds. The migration is therefore documented as preserving the experimental specification, not as guaranteeing bitwise CPU–GPU equivalence.

## Locked screening outcome

The completed CPU screening remains authoritative and is not rerun for selection.

- MSL: `g03_rf7_large_lr1e3`, `g04_rf7_large_lr3e4`
- SMAP: `g11_rf31_large_lr1e3`, `g09_rf31_base_lr1e3`

No shortlist substitutions are permitted on the basis of CUDA verification, anomaly outcomes, or TEST data.

## Permitted implementation changes

The amendment is restricted to:

- explicit `--device cpu|cuda` selection, with a clear failure if CUDA is requested but unavailable;
- moving models, minibatches and evaluation tensors to the requested device;
- CPU initialisation followed by device transfer before optimizer construction;
- deterministic CUDA configuration and TF32 disabling;
- explicit non-foreach/non-fused AdamW execution to avoid backend-dependent optimizer implementation selection;
- portable CPU-form checkpoint state dictionaries;
- device and execution-environment metadata;
- amendment-aware freeze verification that preserves the original freeze manifest and verifies both the original preimage hashes and the amended file hashes;
- serial one-process-per-GPU execution for CUDA to avoid forked CUDA contexts.

## Frozen items unchanged

The following remain unchanged: candidate definitions, shortlist, data arrays, chronological splits, preprocessing, command selection, graph construction and fixed aggregation, model equations, initialisation procedure, sample order, batch size, seeds, learning rates, weight decay, loss, gradient clipping, maximum epochs, patience, minimum delta, validation aggregation, checkpoint selection, robustness folds, confirmation roster, and promotion rules.

No mixed precision, TF32, `torch.compile`, fused optimizer, larger batch, additional data-loader workers, architecture change, candidate change, or NASA TEST/anomaly-label access is introduced.

## CUDA reproducibility settings

CUDA production runs use:

- existing Python/NumPy/PyTorch seeds, plus CUDA seed propagation;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- cuDNN benchmarking disabled;
- deterministic algorithms enabled with errors raised for unsupported nondeterministic operations;
- cuDNN deterministic mode enabled;
- TF32 disabled for matrix multiplication and convolution where supported by the installed PyTorch version;
- float32 throughout;
- AdamW with `foreach=False` and `fused=False` where the installed API exposes `fused`;
- one training process per GPU.

The exact Colab environment is written to `exp006_execution_environment.json` and each production result row records the device and execution-environment identifier.

## Prespecified verification bridge

Verification uses TRAIN data only. For each mission, the verification channel is the first channel in the frozen screening roster with non-zero fit-block telemetry variance. The selected channel is written before backend comparisons are viewed.

The verification package checks:

1. exact split, command-column, graph support/weight, parameter-count and initial-state identity;
2. original frozen trainer versus amended trainer on CPU for three epochs of identical seeded training;
3. CPU/CUDA predictions and loss at `rtol=1e-5`, `atol=1e-6`;
4. CPU/CUDA gradients and one-step parameter updates at `rtol=1e-4`, `atol=1e-6`;
5. numerical parity from a trained non-zero-head checkpoint;
6. full CPU/CUDA training for all four shortlisted configurations on the mission-specific verification channels at seed 17;
7. investigation if full-training validation MSE differs by more than `max(1e-6, 0.01*MSE_CPU)`;
8. one repeated CUDA bridge fit per mission;
9. CPU reload of a CUDA-produced portable checkpoint and prediction comparison.

This produces eight full bridge fits (four CPU and four CUDA) plus two GPU repeats, in addition to short CPU regression checks.

If any verification criterion fails, production migration pauses for investigation. Tolerances must not be loosened post hoc and the backend producing the more favourable validation result must not be selected opportunistically.

## Remaining EXP006 execution

After verification passes:

- all 192 robustness fits, including refitted EXP002 and EXP005 controls, run under the same Colab CUDA execution configuration;
- all 486 confirmation fits run on CUDA, including combinations previously encountered during CPU screening;
- CPU screening checkpoints are not substituted into the CUDA confirmation table;
- interrupted fits restart from the original seed unless a complete training state, including optimizer, RNG and early-stopping state, is available.

The original `EXP006_PRETRAIN_FREEZE.json` remains unchanged. This amendment is linked to it through `EXP006_DEVICE_AMENDMENT.json`.
