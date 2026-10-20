from __future__ import annotations

import json
import math
import re
from pathlib import Path
import argparse

import numpy as np

from adapters import MODEL_KEYS, SEEDS, load_checkpoint_model, predict_numpy, select_windows, train_split_data
from io_utils import load_protocol, package_dir, project_root, sha256_file, write_csv_rows, write_json


SOURCE_BY_MODEL = {
    "exp001": "src/v2_models.py",
    "exp002": "src/v2_models_exp002.py",
    "exp003": "src/v2_models_exp003.py",
    "exp004": "src/v2_models_exp004.py",
    "exp005": "src/v2_models_exp005.py",
}
CONFIG_BY_MODEL = {
    "exp001": "configs/v2_exp001_tcn_baseline.json",
    "exp002": "configs/v2_exp002_tcn_gcn_fixed.json",
    "exp003": "configs/v2_exp003_tcn_no_graph.json",
    "exp004": "configs/v2_exp004_tcn_gcn_learned_edges.json",
    "exp005": "configs/v2_exp005_command_tcn_tuning.json",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-channels", type=int, default=None, help="Smoke-test only; default audits all 81 channels.")
    args = parser.parse_args()
    protocol = load_protocol()
    root = project_root()
    out = package_dir()

    import sys
    sys.path.insert(0, str(root / "src"))
    from v2_data import list_channels

    channels = list_channels()
    if args.max_channels is not None:
        channels = channels[:int(args.max_channels)]
    failures = []
    rows = []
    cal_counts = []

    print("=" * 76)
    print("ANOMALY FREEZE V2 — TRAIN-ONLY READINESS AUDIT")
    print("=" * 76)
    print(f"Channels: {len(channels)} | models: {len(MODEL_KEYS)} | seeds/model: {len(SEEDS)}")
    print("No NASA TEST arrays or anomaly labels are opened by this script.\n")

    # Hash static source/config files once.
    static = {}
    for model_key in MODEL_KEYS:
        source_path = root / SOURCE_BY_MODEL[model_key]
        config_path = root / CONFIG_BY_MODEL[model_key]
        if not source_path.exists():
            failures.append(f"missing source file: {source_path}")
            continue
        if not config_path.exists():
            failures.append(f"missing config file: {config_path}")
            continue
        static[model_key] = (sha256_file(source_path), sha256_file(config_path))

    # Channel outer loop ensures TRAIN windows are materialised only once per channel.
    for ci, channel in enumerate(channels, 1):
        try:
            data = train_split_data(channel, lookback=64)
        except Exception as exc:
            failures.append(f"{channel}: TRAIN split load failed: {type(exc).__name__}: {exc}")
            continue

        for model_key in MODEL_KEYS:
            if model_key not in static:
                continue
            source_hash, config_hash = static[model_key]
            for seed in SEEDS:
                try:
                    loaded = load_checkpoint_model(model_key, channel, seed)
                    if loaded.lookback != 64:
                        raise RuntimeError(f"unexpected lookback {loaded.lookback}; frozen benchmark expects 64")
                    ckpt = loaded.checkpoint
                    result = ckpt["result"]
                    x_val = select_windows(data.x_val, loaded.feature_indices)
                    pred = predict_numpy(loaded.model, x_val)
                    mse = float(np.mean((pred - data.y_val.astype(np.float64)) ** 2))
                    recorded = float(result["validation_mse"])
                    tol = max(2e-7, 2e-6 * abs(recorded))
                    if not math.isfinite(mse) or abs(mse - recorded) > tol:
                        raise RuntimeError(
                            f"validation MSE mismatch computed={mse:.12g} recorded={recorded:.12g} tol={tol:.3g}"
                        )
                    expected_cal = len(data.y_cal)
                    if int(result["calibration_samples"]) != expected_cal:
                        raise RuntimeError(
                            f"calibration count mismatch checkpoint={result['calibration_samples']} current={expected_cal}"
                        )
                    cal_counts.append((channel, data.mission, expected_cal))
                    rows.append({
                        "model_key": model_key,
                        "experiment_id": result["experiment_id"],
                        "channel": channel,
                        "mission": data.mission,
                        "seed": seed,
                        "checkpoint_path": str(loaded.checkpoint_path.relative_to(root)),
                        "checkpoint_sha256": sha256_file(loaded.checkpoint_path),
                        "model_source_path": SOURCE_BY_MODEL[model_key],
                        "model_source_sha256": source_hash,
                        "config_path": CONFIG_BY_MODEL[model_key],
                        "config_sha256": config_hash,
                        "lookback": loaded.lookback,
                        "feature_count": len(loaded.feature_indices),
                        "fit_samples": len(data.y_fit),
                        "validation_samples": len(data.y_val),
                        "calibration_samples": expected_cal,
                        "recorded_validation_mse": f"{recorded:.17g}",
                        "reproduced_validation_mse": f"{mse:.17g}",
                        "validation_mse_abs_error": f"{abs(mse-recorded):.17g}",
                    })
                except Exception as exc:
                    failures.append(f"{model_key} {channel} seed={seed}: {type(exc).__name__}: {exc}")

        if ci % 10 == 0 or ci == len(channels):
            print(f"Audited {ci}/{len(channels)} channels")

    # Canonical coverage check: 5 * 81 * 3 = 1215 selected checkpoints.
    expected = len(MODEL_KEYS) * len(channels) * len(SEEDS)
    if len(rows) != expected:
        failures.append(f"canonical manifest has {len(rows)} rows, expected {expected}")

    # Verify calibration counts agree across model/seed repeats for each channel.
    by_channel = {}
    for channel, mission, n in cal_counts:
        by_channel.setdefault(channel, set()).add(int(n))
    inconsistent = {k: sorted(v) for k, v in by_channel.items() if len(v) != 1}
    if inconsistent:
        failures.append(f"inconsistent calibration counts by channel: {inconsistent}")

    # Canonical files only; smoke/benchmark extras do not enter the manifest.
    write_csv_rows(out / "checkpoint_manifest.csv", rows)
    report = {
        "status": "PASS" if not failures else "FAIL",
        "canonical_checkpoints_expected": expected,
        "canonical_checkpoints_audited": len(rows),
        "failures": failures,
        "notes": [
            "TRAIN-only audit; no TEST arrays or anomaly labels accessed.",
            "Checkpoint validation MSE was independently reproduced from current TRAIN validation windows.",
            "Only canonical selected checkpoints are included; smoke/benchmark extras are ignored.",
        ],
    }
    write_json(out / "readiness_report.json", report)

    print("\n" + "=" * 76)
    print(f"READINESS AUDIT: {report['status']}")
    print(f"Canonical checkpoints audited: {len(rows)}/{expected}")
    if failures:
        for f in failures[:25]:
            print("FAIL:", f)
        if len(failures) > 25:
            print(f"... {len(failures)-25} more failures in readiness_report.json")
        raise SystemExit(1)
    print("Saved checkpoint_manifest.csv and readiness_report.json")


if __name__ == "__main__":
    main()
