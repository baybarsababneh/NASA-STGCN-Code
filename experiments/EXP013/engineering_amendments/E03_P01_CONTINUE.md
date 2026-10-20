# EXP013 E03 P01: bounded parameter-gate amendment

Keep the current Colab runtime and E03 package, WORK, CACHE and DRIVE directories. Insert the cell below immediately after the failed Section 4. Download and upload only `EXP013_E03_P01_PARAMETER_GATE.py`. Do not rerun the setup/smoke cells or restart the work directory. This is an external engineering amendment, not a replacement input bundle or notebook.

```python
from pathlib import Path
import os, sys, hashlib
from google.colab import files

uploaded = files.upload()
name = 'EXP013_E03_P01_PARAMETER_GATE.py'
if set(uploaded) != {name}:
    raise RuntimeError('Upload only the P01 Python amendment')
p = Path('/content') / name
p.write_bytes(uploaded[name])
del uploaded
assert hashlib.sha256(p.read_bytes()).hexdigest() == 'fb4522f3926c09d9b52b46a2dd776e79608b505b7081b513832d4a9020d63eea'
os.environ['EXP013_E03_PACKAGE'] = str(PACKAGE)
launch([sys.executable, '-u', str(p), 'hardware',
        '--work', str(WORK), '--drive', str(DRIVE),
        '--cache', str(CACHE), '--device', 'cuda',
        '--parent-pid', str(os.getpid())], CACHE, total_timeout=1800)
run('develop')
```

The existing runner reuses completed hardware receipts and the corrected MSL graph checkpoint. The amended SMAP fusion regression performs the original two native updates, two corrected updates and two corrected replay updates, at the unchanged full batch size 256. Its original operation deadline is 600 seconds. Remaining original hardware checks retain their deadlines; the continuation cell has a 30-minute hardware-action ceiling. Training begins automatically only after the complete hardware action succeeds. `develop` resumes the original 22-fit schedule, selection, study freeze, synthetic confirmation and TEST-score sealing. After it finishes, use the existing Section 6 for sealed evaluation. Drive backups remain under the original runner.

## What is known and what must still be measured

The reported maxima, 6.02286309e-6 and 6.02309592e-6, differ by 2.3283e-10. This is consistent with a first-update discrepancy carrying through Adam; aggregate maxima alone do not identify its cause. Actual GPU element values and gradients were not available for this investigation. No new A100 validation is claimed.

For a cold ordinary Adam update, with the unchanged learning rate 0.001 and epsilon 1e-8, the movement is `0.001*g/(abs(g)+1e-8)`. Its maximum local sensitivity to gradient perturbations is `0.001/1e-8 = 100000`. Tiny near-zero FP32 gradient differences can therefore exceed a 5e-6 parameter gate while outputs, gradients and Adam moments still satisfy their existing gates. Relative parameter error also depends on the actual parameter magnitudes; a tensor-wide absolute maximum cannot establish that error.

A CPU sensitivity example reproduces a narrow stable failure using this equation. Nine CPU tests passed, including two updates of the actual small SMAP fusion architecture and exact corrected replay. These establish the mechanism and amendment behavior, not the values of the actual A100 failing elements or full-batch GPU equivalence.

## Fixed acceptance rule

Only SMAP fusion `root.discriminator.0.weight` may use parameter atol=7.5e-6, with unchanged rtol=2e-5. All other parameter tensors retain atol=5e-6 and rtol=2e-5. Outputs, injection, gradients, Adam, independent Adam equations and exact corrected RNG replay retain their original gates. An additional relative L2 parameter-error ceiling of 2e-5 applies to the target tensor. The absolute ceiling is 0.75% of the 0.001 learning rate; this alone is not a justification for acceptance.

Acceptance additionally requires identical cold initialization, ordinary frozen Adam configuration/counters, FP64 checks of the FP32 gradient-to-moment recurrences, the original Adam equation bound, and an explanation of the native/corrected parameter difference from their before-values and Adam movements within an explicit rounding budget. Unexplained discrepancies, additional failing tensors/categories, or errors above the fixed ceiling abort. There is no automatic further relaxation.

The real A100 check records every original failing index with both reference magnitudes, gradients, first/second moments, Adam movements, epsilon ratios, predicted differences and explanation residuals. The worst element is also printed. Read `WORK/engineering/hardware/E03_P01_PARAMETER_GATE.json` for the complete record. The audit retains the original failed comparisons, amendment source/hash and unchanged scientific-checkpoint hashes; the original hardware gate includes it in its evidence and backups.

This conditional amendment is defensible as an engineering equivalence threshold for a mathematically equivalent FP32 implementation. It does not assert bitwise equivalence to native convolution or bound divergence over an entire native training trajectory. If the actual element checks fail, production does not start; retain the audit for a minimum implementation investigation instead of raising tolerances again.

## Integrity and bounded local regression

All 101 members of the authoritative E03 package manifest were rechecked and remain unchanged. No architecture, objective, hyperparameter, data, RNG rule, detector protocol, TRAIN-only selection, scientific gate or evaluator was edited. No scientific fits were performed locally. The supplied amendment snapshots private regression state and modifies only its comparison; it does not edit a scientific checkpoint. Existing corrected E03 scientific progress remains compatible. Native-convolution checkpoint continuation remains prohibited.

To repeat the CPU tests, place the original untouched E03 package at `package/` beside the supplied amendment and `test_parameter_gate.py`, with torch 2.6.0 and numpy 2.1.3 available, then run `python test_parameter_gate.py`. The nine tests completed in 1.57 seconds here. They test the fixed allowance, actual small fusion two-update/replay, unchanged other roles/categories, and rejection of unexplained Adam errors, initialization differences and excess errors. They are CPU tests, not production authorization.

Production authorization is conditional on the amended real A100 regression and all remaining original hardware/scientific gates passing. The successful 17.33-second MSL epoch is user-reported evidence and is reused by the original runner; this amendment does not validate it again or retrain it.
