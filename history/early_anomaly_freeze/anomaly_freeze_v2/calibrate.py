from __future__ import annotations

import json
from collections import defaultdict
import argparse

import numpy as np

from adapters import MODEL_KEYS, SEEDS, load_checkpoint_model, predict_numpy, select_windows, train_split_data
from io_utils import load_protocol, package_dir, read_csv_rows, write_csv_rows, write_json
from scores import absolute_residual, causal_ewma, empirical_quantile_higher


def threshold_rows(model_key, channel, mission, seed_label, raw_scores, quantiles, ewma_span):
    scorer_map = {
        "raw": raw_scores,
        f"ewma_span{ewma_span}": causal_ewma(raw_scores, span=ewma_span),
    }
    out = []
    for scorer, scores in scorer_map.items():
        distinct = int(len(np.unique(scores)))
        score_max = float(np.max(scores))
        for q in quantiles:
            thr = empirical_quantile_higher(scores, q)
            out.append({
                "model_key": model_key,
                "channel": channel,
                "mission": mission,
                "seed": seed_label,
                "scorer": scorer,
                "quantile": f"{q:.6f}",
                "method": "higher",
                "alarm_rule": ">",
                "calibration_scores": len(scores),
                "distinct_scores": distinct,
                "constant_scores": int(distinct <= 1),
                "threshold": f"{thr:.17g}",
                "score_max": f"{score_max:.17g}",
                "threshold_is_max": int(thr == score_max),
            })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-channels", type=int, default=None, help="Smoke-test only; default calibrates all 81 channels.")
    args = parser.parse_args()
    out = package_dir()
    readiness_path = out / "readiness_report.json"
    manifest_path = out / "checkpoint_manifest.csv"
    if not readiness_path.exists() or not manifest_path.exists():
        raise RuntimeError("Run audit_readiness.py first")
    readiness = json.loads(readiness_path.read_text())
    if readiness.get("status") != "PASS":
        raise RuntimeError("Readiness audit did not pass")

    protocol = load_protocol()
    q = [float(protocol["calibration"]["primary_quantile"])] + [
        float(x) for x in protocol["calibration"]["sensitivity_quantiles"]
    ]
    ewma_span = int(protocol["scoring"]["sensitivity_scorer"]["span"])

    import sys
    from io_utils import project_root
    root = project_root()
    sys.path.insert(0, str(root / "src"))
    from v2_data import list_channels
    channels = list_channels()
    if args.max_channels is not None:
        channels = channels[:int(args.max_channels)]

    rows = []
    diagnostics = []
    print("=" * 76)
    print("ANOMALY FREEZE V2 — TRAIN CALIBRATION ONLY")
    print("=" * 76)
    print("No NASA TEST arrays or anomaly labels are opened by this script.\n")

    # Channel outer loop: one TRAIN window materialisation per channel.
    for ci, channel in enumerate(channels, 1):
        data = train_split_data(channel, lookback=64)

        # Persistence: deterministic, one threshold per channel/scorer/q.
        raw_p = absolute_residual(data.y_cal, data.persistence_cal)
        rows.extend(threshold_rows("persistence", channel, data.mission, "deterministic", raw_p, q, ewma_span))

        for model_key in MODEL_KEYS:
            for seed in SEEDS:
                loaded = load_checkpoint_model(model_key, channel, seed)
                if loaded.lookback != 64:
                    raise RuntimeError(f"{model_key} {channel}: unexpected lookback {loaded.lookback}")
                x_cal = select_windows(data.x_cal, loaded.feature_indices)
                pred = predict_numpy(loaded.model, x_cal)
                if len(pred) != len(data.y_cal):
                    raise RuntimeError(f"{model_key} {channel} seed={seed}: invalid calibration score count")
                raw = absolute_residual(data.y_cal, pred)
                rows.extend(threshold_rows(model_key, channel, data.mission, seed, raw, q, ewma_span))

        if ci % 10 == 0 or ci == len(channels):
            print(f"Calibrated {ci}/{len(channels)} channels")

    # Diagnostics summarised by quantile/scorer across unique channel-model-seed rows.
    groups = defaultdict(lambda: {"rows": 0, "threshold_is_max": 0, "constant_scores": 0, "min_n": 10**9, "max_n": 0, "ns": []})
    for r in rows:
        key = (r["scorer"], r["quantile"])
        g = groups[key]
        g["rows"] += 1
        g["threshold_is_max"] += int(r["threshold_is_max"])
        g["constant_scores"] += int(r["constant_scores"])
        n = int(r["calibration_scores"])
        g["min_n"] = min(g["min_n"], n)
        g["max_n"] = max(g["max_n"], n)
        g["ns"].append(n)
    for (scorer, quantile), g in sorted(groups.items()):
        diagnostics.append({
            "scorer": scorer,
            "quantile": quantile,
            "rows": g["rows"],
            "threshold_is_max_rows": g["threshold_is_max"],
            "constant_score_rows": g["constant_scores"],
            "calibration_n_min": g["min_n"],
            "calibration_n_median": float(np.median(g["ns"])),
            "calibration_n_max": g["max_n"],
        })

    write_csv_rows(out / "thresholds.csv", rows)
    write_csv_rows(out / "calibration_diagnostics.csv", diagnostics)
    write_json(out / "calibration_report.json", {
        "status": "PASS",
        "channels": len(channels),
        "threshold_rows": len(rows),
        "quantiles": q,
        "scorers": ["raw", f"ewma_span{ewma_span}"],
        "alarm_rule": ">",
        "notes": [
            "Thresholds use only the final 20% of official TRAIN.",
            "Calibration windows may use earlier observed TRAIN history but calibration targets are disjoint.",
            "No TEST arrays or anomaly labels were accessed.",
        ],
    })
    print("\nCALIBRATION PASS")
    print(f"Threshold rows: {len(rows)}")
    print("Saved thresholds.csv, calibration_diagnostics.csv, calibration_report.json")


if __name__ == "__main__":
    main()
