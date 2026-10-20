# EXP009 E01 — resume the existing run

1. Stop any still-running original EXP009 cell. Keep all Drive files. Do not reset or delete results, edit source, regenerate a parent manifest, or start a new experiment directory.
2. Open `EXP009_DETECTOR_V1_E01_Resume.ipynb` in Colab. Use the same runtime if available. A CPU runtime remains correct.
3. Run All and upload `EXP009_DETECTOR_V1_E01_bundle.zip` once when prompted. The notebook reuses `MyDrive/NASA_GCN/EXP009_DETECTOR_V1/results` automatically. No data re-upload is needed.
4. E01 checks the original bundle/source hashes, exact recorded environment, original scientific freeze and every committed result/marker. It records the actual committed count. The last progress line may have said 900 even if more units committed afterward.
5. It snapshots provenance into sibling `engineering_E01`, then resumes using the original scientific/cache identity plus an explicitly recorded serializer amendment. Verified completed units are not recalculated or rewritten.
6. On another interruption, rerun this E01 notebook. Do not switch back to the original notebook. Reuse the same environment. Any identity/hash mismatch stops safely rather than bypassing verification.
7. Share the exported E01 evidence ZIP and executed notebook after completion. Both original scientific provenance and E01 execution provenance are included.

## What changes

Only the JSON writer accepts supported NumPy scalar values as equivalent native Python values. Detector calculations, thresholds, data, selection rules, controls and the frozen scientific design do not change. Original code and manifest remain intact. The amendment launcher and source manifest record the changed runtime implementation separately.

## What was verified locally

- Actual short-segment detector payloads reproduce the original `np.int64` exception and serialize under E01 without changing values.
- Native JSON payload bytes and identity hashes are unchanged.
- NaN/infinity and unsupported objects still fail.
- A synthetic 900-commit interrupted run resumes through all 4,833 scheduled units.
- Pre-amendment results, markers and RUN_FREEZE.json remain byte-identical.
- Repeated resume avoids recomputation and corruption is rejected.

Your Drive files have not been accessed locally. The notebook verifies them. Hosted Colab execution of this amendment has not yet been performed here. No new scientific selection, NASA TEST access or anomaly evaluation was performed while preparing E01.

If the environment check fails, preserve the message and stop. E01 does not authorise changing the numerical environment or relaxing verification.
