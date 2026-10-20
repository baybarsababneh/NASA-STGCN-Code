from __future__ import annotations

import json
from pathlib import Path

from io_utils import package_dir, sha256_file, write_json


REQUIRED = [
    "protocol.json",
    "io_utils.py",
    "adapters.py",
    "scores.py",
    "metrics.py",
    "audit_readiness.py",
    "calibrate.py",
    "export_test_scores.py",
    "evaluate_locked.py",
    "test_metrics.py",
    "checkpoint_manifest.csv",
    "readiness_report.json",
    "metric_test_report.json",
    "thresholds.csv",
    "calibration_diagnostics.csv",
    "calibration_report.json",
]


def main():
    root = package_dir()
    readiness = json.loads((root / "readiness_report.json").read_text()) if (root / "readiness_report.json").exists() else {}
    calibration = json.loads((root / "calibration_report.json").read_text()) if (root / "calibration_report.json").exists() else {}
    if readiness.get("status") != "PASS":
        raise RuntimeError("readiness_report.json is not PASS")
    if calibration.get("status") != "PASS":
        raise RuntimeError("calibration_report.json is not PASS")
    metric_tests = json.loads((root / "metric_test_report.json").read_text()) if (root / "metric_test_report.json").exists() else {}
    if metric_tests.get("status") != "PASS":
        raise RuntimeError("metric_test_report.json is not PASS")
    if int(readiness.get("canonical_checkpoints_audited", -1)) != 1215:
        raise RuntimeError("Full 81-channel readiness audit was not completed (expected 1215 checkpoints)")
    if int(calibration.get("channels", -1)) != 81 or int(calibration.get("threshold_rows", -1)) != 5184:
        raise RuntimeError("Full 81-channel calibration was not completed (expected 5184 threshold rows)")
    missing = [name for name in REQUIRED if not (root / name).exists()]
    if missing:
        raise RuntimeError(f"missing freeze inputs: {missing}")
    hashes = {name: sha256_file(root / name) for name in REQUIRED}
    manifest = {
        "freeze_id": "nasa_smap_msl_anomaly_freeze_v2",
        "status": "FROZEN_PRE_TEST",
        "files_sha256": hashes,
        "warning": "Any change to a hashed file after this point creates a new protocol version. Do not select rules after inspecting TEST anomaly outcomes."
    }
    write_json(root / "FREEZE_MANIFEST.json", manifest)
    print("PRE-TEST FREEZE: COMPLETE")
    print("Saved FREEZE_MANIFEST.json")


if __name__ == "__main__":
    main()
