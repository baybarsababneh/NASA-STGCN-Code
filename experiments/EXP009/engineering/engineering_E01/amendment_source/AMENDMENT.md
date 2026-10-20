# EXP009_DETECTOR_V1_E01 — engineering amendment

## Scope

Fix JSON serialization of supported NumPy scalar values. This is an engineering correction to EXP009_DETECTOR_V1, not a new scientific experiment or a restart. Config, residual data, detector calculations, disturbance grid, splits, candidates, seeds, ranking, gates, controls and stopping rules remain byte-identical in the original package.

Root cause reproduced on synthetic traces with the T-9 and D-12 split lengths: `np.flatnonzero` returns NumPy integers. When `duration=min(duration,end-start)` caps a long probe to a short segment, `duration` can be `np.int64`. Undetected probes can also propagate this type into `delay_censored`. The values are correct. Standard `json.dumps` rejects the type. The previous preparation tests failed to cover serialization of short-segment result payloads. E01 adds that regression coverage.

## Exact runtime change

The E01 launcher imports the hash-verified, unmodified original `run_exp009.py` and assigns its global `dumps` function to `dumps_e01`. This uses the original formatting, sorting and `allow_nan=False`, adding only a JSON `default` handler converting supported `np.integer`, `np.floating` and `np.bool_` scalars via `.item()` to Python int/float/bool. It does not round values, stringify numeric fields, convert nonfinite numbers to null, or accept arbitrary arrays/objects. Unsupported types and NaN/infinity still fail closed.

Every payload V1 already encoded retains identical bytes. Hence original run hashes, unit IDs and marker hashes remain valid. The amendment is separately identified and hashed. The original run manifest is not falsely replaced with a manifest of changed source.

## Provenance and resume

- Keep original `results/`, `RUN_FREEZE.json`, commits and `.done.json` markers.
- Keep the original bundle. Do not manually edit or regenerate its manifest.
- E01 has its own source manifest and execution record, linked to the original run hash and manifest.
- Before resuming, verify every current committed unit against its marker and planned identity.
- Snapshot the pre-E01 commit inventory, original freeze, saved source/config/protocol and requirements.
- Record the actual committed count. A progress line saying 900 can precede additional commits before failure, so the launcher does not invent an exact count.
- Execute the original runner with only the serializer extension. Verified units are reused. An uncommitted failing unit is recalculated under identical scientific settings and then written successfully.
- Verify all pre-E01 result bytes, marker bytes and the original freeze remain unchanged.
- Keep new engineering evidence in sibling `engineering_E01/`, not inside the parent's scientific identity. Export both the final results and engineering evidence.

The parent RUN_FREEZE continues to identify the original scientific/cache namespace. It is no longer a complete description of executed code by itself: reviewers must also receive ENGINEERING_AMENDMENT.json and amendment_source. This distinction is explicit, not hidden.

## Verification and limits

Tests reproduce the actual short-segment serializer error, check nested scalar handling and legacy-byte compatibility, and preserve rejection of nonfinite/unknown values. A synthetic integration test reproduces 900 successful commits followed by the int64 error, resumes through the full 4,833-unit scheduling path, verifies no recomputation of old commits, checks byte preservation, reruns without recomputation, and rejects corruption. Numerical detector outputs in that integration test are synthetic mocks. Separate short-trace tests use the actual unchanged detector functions.

The user's Drive results have not been inspected here. Their count, identity, environment and hashes are checked by the notebook before resuming. No NASA TEST access, anomaly evaluation, model training or detector redesign is performed by this amendment.
