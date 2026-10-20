# Runtime amendment entry point

Read EXP012_A10040GB_COMPATIBILITY_AMENDMENT_1.md and COMPATIBILITY_AMENDMENT.json. All scientific settings remain in the unchanged evaluation contract in EVALUATION_BINDING.json. Real40GB authorization is pending until the separate nominal-only gate produces a complete, matching COMPATIBILITY_PASS.json.

Nominal validation CLI:
```bash
python src/validate_40gb.py --results /path/completed-results --package /path/original-package --output /path/separate-compatibility-evidence --validate-nominal-only
```
Phase A retains its old arguments and requires `--compatibility-directory /path/separate-compatibility-evidence`. Phase B code and arguments are unchanged. Both source arrays and completed results are read-only. No scientific fit, threshold replacement or numerical tolerance is authorised.
