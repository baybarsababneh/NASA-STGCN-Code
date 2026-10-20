"""Strict manifests, atomic writes, fit commits, and pure selection rules."""
import hashlib
import io
import json
import os
import shutil
import time
import uuid
import zipfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with open(temp, "wb") as f:
        f.write(value)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def write_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode())


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"Frozen artifact mismatch: {path}. Do not mix executions.")
    else:
        write_json(path, value)


def validate_package(root):
    root = Path(root)
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts
              and p.name != "PACKAGE_MANIFEST.json"}
    if actual != set(manifest["files"]):
        raise RuntimeError("Bundle file roster differs from its manifest")
    for rel, expected in manifest["files"].items():
        if sha(root / rel) != expected:
            raise RuntimeError(f"Bundle checksum mismatch: {rel}")
    npys = sorted(root.rglob("*.npy"))
    if len(npys) != 81 or any(p.parent != root / "train" for p in npys):
        raise RuntimeError("Only the 81 allowlisted TRAIN arrays may be present")
    return manifest


def job_id(job):
    return digest(job)[:24]


def read_commit(out, job, run_hash):
    base = Path(out) / "fits" / job["stage"] / job_id(job)
    done = base.with_suffix(".done.json")
    if not done.exists():
        return None
    commit = json.loads(done.read_text())
    archive = base.with_suffix(".zip")
    if commit["run_hash"] != run_hash or commit["job"] != job:
        raise RuntimeError(f"Commit identity mismatch: {base}")
    if not archive.exists() or sha(archive) != commit["archive_sha256"]:
        raise RuntimeError(f"Corrupt committed fit: {base}; preserve and investigate")
    with zipfile.ZipFile(archive) as z:
        if set(z.namelist()) != {"result.json", "checkpoint.pt", "history.json", "job.json"}:
            raise RuntimeError(f"Incomplete fit archive: {base}")
        if json.loads(z.read("job.json")) != job:
            raise RuntimeError("Archive job mismatch")
        result = json.loads(z.read("result.json"))
    return result


def commit_fit(out, job, run_hash, local_archive):
    base = Path(out) / "fits" / job["stage"] / job_id(job)
    base.parent.mkdir(parents=True, exist_ok=True)
    final = base.with_suffix(".zip")
    temp = base.with_suffix(".zip.tmp-" + uuid.uuid4().hex)
    expected = sha(local_archive)
    with open(local_archive, "rb") as src, open(temp, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    if sha(temp) != expected:
        raise RuntimeError("Durable checkpoint copy verification failed")
    os.replace(temp, final)
    write_json(base.with_suffix(".done.json"),
               {"run_hash": run_hash, "job": job, "archive_sha256": expected})


def screen_selection(rows, config):
    import pandas as pd
    df = pd.DataFrame(rows)
    summary = df.groupby(["mission", "candidate_id"], as_index=False).agg(
        validation_mse_mean=("validation_mse", "mean"),
        validation_mse_median=("validation_mse", "median"),
        parameters_mean=("parameters", "mean"), runs=("seed", "size"))
    summary = summary.sort_values(
        ["mission", "validation_mse_mean", "parameters_mean", "candidate_id"], kind="stable")
    summary["rank"] = summary.groupby("mission").cumcount() + 1
    selected = {m: summary[summary.mission == m].head(2).candidate_id.tolist()
                for m in config["dataset"]["missions"]}
    if any(len(v) != 2 for v in selected.values()):
        raise RuntimeError("Cannot select two candidates per mission")
    return summary, selected


def promotion(rows, selected, config):
    """GPU reference only. Historical CPU MSEs are never input to this function."""
    import pandas as pd
    df = pd.DataFrame(rows)
    checks, decisions = [], {}
    for mission, candidates in selected.items():
        ref = df[(df.stage == "reference") & (df.mission == mission)]
        ref_mean = float(ref.validation_mse.mean())
        if not ref_mean > 0:
            raise RuntimeError("Zero/nonpositive reference MSE: relative gate undefined")
        eligible = []
        for cid in candidates:
            can = df[(df.stage == "confirm") & (df.mission == mission) & (df.candidate_id == cid)]
            cm = float(can.validation_mse.mean())
            improvement = 100 * (ref_mean - cm) / ref_mean
            seed_improvements = {}
            for seed in config["confirmation"]["seeds"]:
                r = float(ref[ref.seed == seed].validation_mse.mean())
                c = float(can[can.seed == seed].validation_mse.mean())
                if not r > 0:
                    raise RuntimeError("Nonpositive seed reference MSE")
                seed_improvements[str(seed)] = 100 * (r - c) / r
            rel = []
            for fold in config["robustness"]["folds"]:
                rr = df[(df.stage == "robust") & (df.mission == mission) & (df.fold == fold)]
                r = float(rr[rr.role == "exp002_reference"].assessment_mse.mean())
                c = float(rr[(rr.role == "gcn_candidate") & (rr.candidate_id == cid)].assessment_mse.mean())
                if not r > 0:
                    raise RuntimeError("Nonpositive fold reference MSE")
                rel.append(100 * (r - c) / r)
            robust = sum(rel) / len(rel)
            gates = [improvement >= config["promotion"]["minimum_mission_mean_improvement_pct_vs_exp002"],
                     all(x > 0 for x in seed_improvements.values()), robust >= 0]
            row = {"mission": mission, "candidate_id": cid, "validation_mse": cm,
                   "gpu_exp002_validation_mse": ref_mean, "improvement_pct": improvement,
                   "robust_relative_improvement_pct": robust,
                   "passes_2pct": gates[0], "passes_each_seed": gates[1],
                   "passes_robustness": gates[2], "promotable": all(gates),
                   **{f"seed{s}_improvement_pct": v for s, v in seed_improvements.items()}}
            checks.append(row)
            if all(gates):
                eligible.append(row)
        eligible.sort(key=lambda r: (r["validation_mse"], r["candidate_id"]))
        decisions[mission] = {
            "decision": "promote_gpu_exp006" if eligible else "retain_exp002_architecture",
            "selected_candidate": eligible[0]["candidate_id"] if eligible else "exp002_reference",
            "validation_mse": eligible[0]["validation_mse"] if eligible else ref_mean,
            "comparison_backend": "GPU-native matched execution",
            "not_anomaly_detection_evidence": True,
        }
    return checks, decisions
