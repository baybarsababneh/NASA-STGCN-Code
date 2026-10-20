from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adapters import MODEL_KEYS, SEEDS, load_checkpoint_model, load_standardized_test, make_test_windows, predict_numpy, select_windows
from io_utils import package_dir, project_root, sha256_file, write_csv_rows, write_json
from scores import absolute_residual, causal_ewma


def validate_freeze():
    root = package_dir()
    path = root / "FREEZE_MANIFEST.json"
    if not path.exists():
        raise RuntimeError("FREEZE_MANIFEST.json missing; run freeze.py before TEST export")
    manifest = json.loads(path.read_text())
    for name, expected in manifest["files_sha256"].items():
        actual = sha256_file(root / name)
        if actual != expected:
            raise RuntimeError(f"frozen file changed after freeze: {name}")


def save_npz(path, target_idx, prediction, y_true, raw, ewma):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        target_idx=np.asarray(target_idx, dtype=np.int64),
        prediction=np.asarray(prediction, dtype=np.float32),
        y_true_standardized=np.asarray(y_true, dtype=np.float32),
        raw=np.asarray(raw, dtype=np.float32),
        ewma_span10=np.asarray(ewma, dtype=np.float32),
    )


def main():
    validate_freeze()
    root = package_dir()
    out_dir = root / "test_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(project_root() / "src"))
    from v2_data import list_channels
    channels = list_channels()
    manifest_rows = []

    print("=" * 76)
    print("LABEL-BLIND TEST SCORE EXPORT")
    print("=" * 76)
    print("This script opens NASA TEST arrays but does NOT open anomaly labels.\n")

    for ci, channel in enumerate(channels, 1):
        standardized, train = load_standardized_test(channel)
        idx, x, y, persistence = make_test_windows(standardized, lookback=64)

        raw_p = absolute_residual(y, persistence)
        ewma_p = causal_ewma(raw_p, span=10)
        ppath = out_dir / "persistence" / f"{channel}.npz"
        save_npz(ppath, idx, persistence, y, raw_p, ewma_p)
        manifest_rows.append({
            "model_key": "persistence", "channel": channel, "mission": train.mission,
            "seed": "deterministic", "score_path": str(ppath.relative_to(root)),
            "sha256": sha256_file(ppath), "n_scores": len(idx), "start_index": int(idx[0]), "end_index_inclusive": int(idx[-1])
        })

        for model_key in MODEL_KEYS:
            for seed in SEEDS:
                loaded = load_checkpoint_model(model_key, channel, seed)
                xin = select_windows(x, loaded.feature_indices)
                pred = predict_numpy(loaded.model, xin)
                raw = absolute_residual(y, pred)
                ewma = causal_ewma(raw, span=10)
                path = out_dir / model_key / f"{channel}_seed{seed}.npz"
                save_npz(path, idx, pred, y, raw, ewma)
                manifest_rows.append({
                    "model_key": model_key, "channel": channel, "mission": train.mission,
                    "seed": seed, "score_path": str(path.relative_to(root)),
                    "sha256": sha256_file(path), "n_scores": len(idx), "start_index": int(idx[0]), "end_index_inclusive": int(idx[-1])
                })
        if ci % 10 == 0:
            print(f"Exported {ci}/{len(channels)} channels")

    write_csv_rows(root / "test_score_manifest.csv", manifest_rows)
    write_json(root / "test_export_report.json", {
        "status": "PASS",
        "rows": len(manifest_rows),
        "label_access": False,
        "notes": ["Only continuous scores/predictions were exported; no anomaly labels were read."],
    })
    print("\nLABEL-BLIND TEST EXPORT: PASS")
    print(f"Score streams: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
