"""EXP006_GPU_V1: one GPU, frozen stage plan, fit-granularity durable resume."""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import gc
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gpu_common import (atomic_bytes, canonical, commit_fit, digest, freeze_json, job_id,
                        promotion, read_commit, screen_selection, sha, validate_package, write_json)
import gpu_train as trainer
from v2_data import TRAIN_DIR, infer_mission, list_channels
from v2_graph import build_channel_graph
from v2_models_exp002 import TCNGCNFixed
from v2_models_exp005 import CommandAwareTCN
from v2_models_exp006 import TCNGCNFixedTuned

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "EXP006_GPU_V1"


def configure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; CPU fallback is prohibited")
    if torch.__version__ != "2.11.0+cu130" or np.__version__ != "2.1.3" or pd.__version__ != "2.2.3":
        raise RuntimeError("Pinned package versions do not match the GPU protocol")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_default_dtype(torch.float32)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines()[0].strip()
    packages = {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}
    return {"python": platform.python_version(), "platform_machine": platform.machine(),
            "libc": list(platform.libc_ver()), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "driver": driver, "gpu": props.name,
            "capability": list(torch.cuda.get_device_capability(device)), "gpu_total_memory": props.total_memory,
            "packages": packages, "torch_build": torch.__config__.show(),
            "dtype": "float32", "tf32": False, "amp": False, "compile": False,
            "deterministic": True, "cudnn_benchmark": False, "cublas_workspace": ":4096:8",
            "optimizer_foreach": False, "optimizer_fused": False, "threads": 1}


def install_data_guard():
    """Defense in depth: the supplied bundle contains no TEST/label data."""
    def guard(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0]))
        lower = p.name.lower()
        if lower.endswith(".npy") and p.resolve().parent != TRAIN_DIR.resolve():
            raise PermissionError(f"Array read outside TRAIN allowlist denied: {p}")
        if "labeled_anomal" in lower or "labelled_anomal" in lower:
            raise PermissionError("Anomaly label access is prohibited")
    sys.addaudithook(guard)


class Lease:
    """Single active notebook only. Lease helps detect accidental concurrent runs.

    Drive is not a distributed transactional filesystem. Do not run two notebooks.
    An abandoned lease expires after 10 minutes; every commit checks ownership.
    """
    def __init__(self, out):
        self.path = Path(out) / "ACTIVE_RUN.json"
        self.token = uuid.uuid4().hex
        self.stop = threading.Event()
        self.error = None

    def __enter__(self):
        if self.path.exists():
            old = json.loads(self.path.read_text())
            if time.time() - old["heartbeat"] < 600:
                raise RuntimeError("Another run may be active. Use one notebook only. After a crashed runtime, wait 10 minutes and Run All again.")
            self.path.rename(self.path.with_name("abandoned_lease_" + old["token"] + ".json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        value = {"token": self.token, "heartbeat": time.time(), "pid": os.getpid()}
        with open(self.path, "x") as f:
            json.dump(value, f)
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def check(self):
        if self.error:
            raise RuntimeError("Durable run lease failed") from self.error
        if json.loads(self.path.read_text())["token"] != self.token:
            raise RuntimeError("Lost run lease; refusing to commit")

    def _heartbeat(self):
        while not self.stop.wait(30):
            try:
                self.check()
                write_json(self.path, {"token": self.token, "heartbeat": time.time(), "pid": os.getpid()})
            except Exception as exc:
                self.error = exc
                return

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.path.exists() and json.loads(self.path.read_text())["token"] == self.token:
            self.path.unlink()


def plans(config, channels, selected=None):
    lookup = {c["candidate_id"]: c for c in config["candidates"]}
    jobs = {s: [] for s in ("screen", "robust", "reference", "confirm")}
    for mission, roster in config["screening"]["channels"].items():
        for spec in config["candidates"]:
            for ch in roster:
                for seed in config["screening"]["seeds"]:
                    jobs["screen"].append(dict(stage="screen", mission=mission, channel=ch, seed=seed,
                                               role="gcn_candidate", candidate=spec, fold="canonical"))
    if selected is None:
        return jobs
    for mission, roster in config["screening"]["channels"].items():
        roles = [("gcn_candidate", lookup[c]) for c in selected[mission]]
        roles += [("exp002_reference", config["references"]["exp002"]),
                  ("exp005_reference", config["references"]["exp005_selected_by_mission"][mission])]
        for fold in config["robustness"]["folds"]:
            for role, spec in roles:
                for ch in roster:
                    for seed in config["robustness"]["seeds"]:
                        jobs["robust"].append(dict(stage="robust", mission=mission, channel=ch, seed=seed,
                                                  role=role, candidate=spec, fold=fold))
    for ch, mission in channels.items():
        for seed in config["confirmation"]["seeds"]:
            jobs["reference"].append(dict(stage="reference", mission=mission, channel=ch, seed=seed,
                role="exp002_reference", candidate=config["references"]["exp002"], fold="canonical"))
        for cid in selected[mission]:
            for seed in config["confirmation"]["seeds"]:
                jobs["confirm"].append(dict(stage="confirm", mission=mission, channel=ch, seed=seed,
                    role="gcn_candidate", candidate=lookup[cid], fold="canonical"))
    if {k: len(v) for k, v in jobs.items()} != {"screen": 288, "robust": 192, "reference": 243, "confirm": 486}:
        raise RuntimeError("Frozen fit counts changed")
    if len({job_id(j) for group in jobs.values() for j in group}) != 1209:
        raise RuntimeError("Duplicate job identities")
    return jobs


def preflight(config, out):
    channels, data_rows, graph_rows = {}, [], []
    for ch in list_channels():
        raw = np.load(TRAIN_DIR / f"{ch}.npy", allow_pickle=False)
        if raw.ndim != 2 or not np.isfinite(raw).all():
            raise RuntimeError(f"Invalid TRAIN array: {ch}")
        mission = infer_mission(raw)
        channels[ch] = mission
        data_rows.append(dict(channel=ch, mission=mission, rows=len(raw), columns=raw.shape[1],
                              sha256=sha(TRAIN_DIR / f"{ch}.npy")))
        for fraction in (0.4, 0.5, 0.6):
            g = build_channel_graph(ch, fit_fraction=fraction)
            fit_target = raw.astype(np.float32)[:int(np.floor(len(raw)*fraction)), 0]
            target_mean = float(fit_target.mean())
            target_scale = float(fit_target.std())
            if not np.isfinite(target_scale) or target_scale < 1e-8:
                target_scale = 1.0
            a = g.adjacency
            assert a.shape == (g.num_nodes, g.num_nodes)
            assert g.feature_indices[0] == 0 and np.all(a[1:] == 0) and a[0, 0] == 0
            assert np.count_nonzero(a) == g.num_edges
            assert (g.num_edges == 0 and np.all(a == 0)) or np.isclose(a[0].sum(), 1)
            assert g.num_edges <= 4 if mission == "MSL" else g.num_edges == g.num_nodes - 1
            graph_rows.append(dict(channel=ch, mission=mission, fit_fraction=fraction,
                features=g.feature_indices.tolist(), sources=g.source_original_columns.tolist(),
                target_mean=target_mean, target_scale=target_scale,
                adjacency_sha256=digest(a.tolist())))
    if list(channels.values()).count("MSL") != 27 or list(channels.values()).count("SMAP") != 54:
        raise RuntimeError("Mission/channel inventory mismatch")
    for mission, roster in config["screening"]["channels"].items():
        if len(roster) != 6 or any(channels[ch] != mission for ch in roster):
            raise RuntimeError("Screening roster mismatch")
    manifest = pd.read_csv(ROOT / "configs/v2_exp006_screening_manifest.csv")
    if set(manifest.channel) != {ch for r in config["screening"]["channels"].values() for ch in r}:
        raise RuntimeError("Frozen screening manifest mismatch")
    checks = []
    # Tiny synthetic training verifies the production CUDA path, not CPU equivalence.
    adjacency = np.array([[0, 1], [0, 0]], dtype=np.float32)
    for dilations in ((1, 2), (1, 2, 4), (1, 2, 4, 8)):
        for kind in ("gcn", "direct"):
            def factory():
                if kind == "gcn":
                    return TCNGCNFixedTuned(2, 8, adjacency, 3, dilations)
                return CommandAwareTCN(2, 8, 3, dilations)
            states, losses = [], []
            rng = np.random.default_rng(123)
            x = rng.normal(size=(24, 2, 64)).astype(np.float32)
            y = rng.normal(size=24).astype(np.float32)
            for repeat in range(2):
                trainer.set_seed(17)
                model = factory().cuda()
                probe = torch.from_numpy(x[:4]).cuda()
                assert next(model.parameters()).is_cuda
                assert torch.equal(model(probe), probe[:, 0, -1])
                # Make the head nonzero so receptive-field check is nontrivial.
                with torch.no_grad():
                    model.head.weight.fill_(0.1)
                    altered = probe.clone()
                    rf = model.receptive_field
                    altered[:, :, :-rf] += 10
                    assert torch.equal(model(probe), model(altered))
                common = dict(config["training_common"], batch_size=8)
                model, loss, _, _, _ = trainer._train_with_early_stopping(
                    model, x[:16], y[:16], x[16:], y[16:], 17, 0.001, common, smoke_epochs=3)
                states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
                losses.append(loss)
                # Reload the CPU-serialized state back into the same GPU architecture.
                with tempfile.TemporaryDirectory() as td:
                    cp = Path(td) / "state.pt"
                    torch.save(states[-1], cp)
                    loaded = factory().cuda()
                    loaded.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True))
                    assert torch.equal(model(probe), loaded(probe))
            if losses[0] != losses[1] or any(not torch.equal(states[0][k], states[1][k]) for k in states[0]):
                raise RuntimeError("Within-GPU deterministic repeatability failed")
            checks.append({"kind": kind, "dilations": list(dilations), "repeatable": True,
                           "persistence_rf_checkpoint_checks": True})
    # Confirm historical EXP002 and variable-stack RF7 are the same algebra.
    trainer.set_seed(17)
    old = TCNGCNFixed(2, 8, adjacency).cuda()
    trainer.set_seed(17)
    new = TCNGCNFixedTuned(2, 8, adjacency).cuda()
    remapped = {k.replace("temporal.conv1.", "temporal.convs.0.")
                 .replace("temporal.conv2.", "temporal.convs.1."): v for k, v in old.state_dict().items()}
    assert set(remapped) == set(new.state_dict())
    assert all(torch.equal(remapped[k], new.state_dict()[k]) for k in remapped)
    with torch.no_grad():
        old.head.weight.fill_(0.1)
        new.head.weight.fill_(0.1)
        probe = torch.randn(8, 2, 64, device="cuda")
        assert torch.equal(old(probe), new(probe))
    freeze_json(out / "data_manifest.json", data_rows)
    freeze_json(out / "graph_manifest.json", graph_rows)
    write_json(out / "preflight.json", {"status": "PASS", "checks": checks,
        "exp002_rf7_algebra_match": True, "array_policy": "TRAIN only", "channels": channels})
    return channels


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    return value


def execute_fit(job, config, out, run_hash, lease):
    lease.check()
    wall_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if job["stage"] == "robust":
        model, result = trainer.train_robustness(job["channel"], job["seed"], config,
            job["role"], job["fold"], candidate=job["candidate"] if job["role"] == "gcn_candidate" else None)
    else:
        model, result = trainer.train_standard(job["channel"], job["seed"], config,
            job["candidate"], role=job["role"])
    torch.cuda.synchronize()
    result.update(stage=job["stage"], fold=job["fold"], role=job["role"], job_id=job_id(job),
                  run_hash=run_hash, experiment_id=EXPERIMENT,
                  fit_wall_seconds=time.perf_counter()-wall_start,
                  gpu_peak_memory_bytes=torch.cuda.max_memory_allocated())
    metric = "assessment_mse" if job["stage"] == "robust" else "validation_mse"
    if not math.isfinite(result[metric]) or result["mission"] != job["mission"]:
        raise RuntimeError("Invalid fit result")
    result = clean_json(result)
    checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "result": result, "job": job, "candidate": job["candidate"],
                  "experiment_config": config, "run_hash": run_hash,
                  "format": "EXP006_GPU_V1-best-model-no-midfit-resume"}
    with tempfile.TemporaryDirectory(prefix="exp006_fit_") as td:
        td = Path(td)
        torch.save(checkpoint, td / "checkpoint.pt")
        archive = td / "fit.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            z.write(td / "checkpoint.pt", "checkpoint.pt")
            z.writestr("result.json", canonical(result))
            z.writestr("job.json", canonical(job))
            z.writestr("history.json", canonical(model.exp006_history))
        lease.check()
        commit_fit(out, job, run_hash, archive)
    del model, checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return result


def stage_run(stage, jobs, config, out, run_hash, lease):
    freeze_json(out / "plans" / f"{stage}.json", jobs)
    rows = []
    for number, job in enumerate(jobs, 1):
        result = read_commit(out, job, run_hash)
        if result is None:
            print(f"[{stage} {number}/{len(jobs)}] {job['mission']} {job['channel']} {job['candidate']['candidate_id']} seed={job['seed']} {job['fold']}", flush=True)
            result = execute_fit(job, config, out, run_hash, lease)
        else:
            print(f"[{stage} {number}/{len(jobs)}] verified resume {job_id(job)}", flush=True)
        rows.append(result)
        lease.check()
        atomic_bytes(out / "tables" / f"{stage}_runs.csv", pd.DataFrame(rows).to_csv(index=False).encode())
        write_json(out / "STATUS.json", {"stage": stage, "completed_in_stage": number,
                   "expected_in_stage": len(jobs), "last_job": job_id(job), "run_hash": run_hash})
    if len(rows) != len(jobs) or len({r["job_id"] for r in rows}) != len(jobs):
        raise RuntimeError("Stage completeness validation failed")
    return rows


def final_tables(rows, selected, config, out):
    df = pd.DataFrame(rows)
    if len(df) != 1209 or df.job_id.nunique() != 1209:
        raise RuntimeError("Final fit inventory incomplete")
    atomic_bytes(out / "tables/all_runs.csv", df.to_csv(index=False).encode())
    canonical_rows = df[df.stage.isin(["confirm", "reference"])]
    summary = canonical_rows.groupby(["mission", "stage", "candidate_id"], as_index=False).agg(
        channels=("channel", "nunique"), fits=("seed", "size"),
        validation_mse_mean=("validation_mse", "mean"), validation_mse_median=("validation_mse", "median"),
        parameters_mean=("parameters", "mean"), best_epoch_mean=("best_epoch", "mean"),
        epochs_mean=("epochs_ran", "mean"), ceiling_hits=("hit_epoch_ceiling", "sum"),
        best_at_ceiling=("best_at_ceiling", "sum"), fit_seconds=("elapsed_seconds", "sum"))
    atomic_bytes(out / "tables/canonical_summary.csv", summary.to_csv(index=False).encode())
    for grouping, name in [(["mission", "stage", "candidate_id", "seed"], "seed_summary"),
                           (["mission", "stage", "candidate_id", "channel"], "channel_summary")]:
        table = canonical_rows.groupby(grouping, as_index=False).agg(validation_mse_mean=("validation_mse", "mean"))
        atomic_bytes(out / f"tables/{name}.csv", table.to_csv(index=False).encode())
    robust = df[df.stage == "robust"].groupby(["mission", "fold", "role", "candidate_id"], as_index=False).agg(
        assessment_mse_mean=("assessment_mse", "mean"), parameters_mean=("parameters", "mean"),
        fits=("seed", "size"), epochs_mean=("epochs_ran", "mean"))
    atomic_bytes(out / "tables/robustness_summary.csv", robust.to_csv(index=False).encode())
    # Paired channel effects: absolute error and guarded percentages; never a selection gate.
    paired = canonical_rows.groupby(["mission", "stage", "candidate_id", "channel"], as_index=False).validation_mse.mean()
    references = paired[paired.stage == "reference"][["mission", "channel", "validation_mse"]].rename(columns={"validation_mse": "reference_mse"})
    paired = paired[paired.stage == "confirm"].merge(references, on=["mission", "channel"], validate="many_to_one")
    paired["absolute_improvement"] = paired.reference_mse - paired.validation_mse
    paired["relative_improvement_pct"] = np.where(paired.reference_mse > 1e-8, 100*paired.absolute_improvement/paired.reference_mse, np.nan)
    paired["family"] = paired.channel.str.split("-").str[0]
    atomic_bytes(out / "tables/paired_channel_effects.csv", paired.to_csv(index=False).encode())
    family = paired.groupby(["mission", "candidate_id", "family"], as_index=False).agg(
        channels=("channel", "size"), absolute_improvement_mean=("absolute_improvement", "mean"),
        relative_improvement_median_pct=("relative_improvement_pct", "median"))
    atomic_bytes(out / "tables/family_effects.csv", family.to_csv(index=False).encode())
    checks, decisions = promotion(rows, selected, config)
    atomic_bytes(out / "tables/promotion_checks.csv", pd.DataFrame(checks).to_csv(index=False).encode())
    write_json(out / "promotion_decision.json", decisions)
    specs = {c["candidate_id"]: c for c in config["candidates"]}
    specs["exp002_reference"] = config["references"]["exp002"]
    write_json(out / "selected_models.json", {m: {**d, "spec": specs[d["selected_candidate"]]} for m, d in decisions.items()})
    text = ["# EXP006_GPU_V1 completed", "", "1209/1209 fits verified. TRAIN-only forecasting development; no anomaly evaluation.",
            "", "Promotion uses same-backend EXP002 architecture refits. CPU EXP006 and original EXP001–EXP005 are unchanged.", ""]
    for m, d in decisions.items():
        text.append(f"- {m}: {d['decision']}; {d['selected_candidate']}; mission mean validation MSE {d['validation_mse']:.9g}.")
    text += ["", "See tables/promotion_checks.csv for each prespecified gate and tables/paired_channel_effects.csv for concentration of gains.",
             "This does not establish improved anomaly detection or a new independent holdout. Do not open another replacement search if promotion fails.",
             "", "Checkpoints are inside fits/<stage>/<job-id>.zip; each includes preprocessing metadata, graph features, specification, history and hashes."]
    atomic_bytes(out / "SUMMARY.md", "\n".join(text).encode())
    print("\n" + "\n".join(text), flush=True)
    return decisions


def export_results(out):
    export = out / "exports"
    export.mkdir(exist_ok=True)
    entries = [p for p in out.rglob("*") if p.is_file() and "exports" not in p.relative_to(out).parts
               and not p.name.startswith("ACTIVE_RUN") and ".tmp-" not in p.name]
    for full in (False, True):
        name = "EXP006_GPU_V1_results.zip" if full else "EXP006_GPU_V1_summary.zip"
        target = export / name
        temp = target.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
            for p in sorted(entries):
                if full or "fits" not in p.relative_to(out).parts:
                    z.write(p, p.relative_to(out))
        os.replace(temp, target)
        atomic_bytes(export / (name + ".sha256"), (sha(target) + "  " + name + "\n").encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    if ROOT in out.parents or out == ROOT:
        raise RuntimeError("Output must be separate from immutable input bundle")
    out.mkdir(parents=True, exist_ok=True)
    with Lease(out) as lease:
        package = validate_package(ROOT)
        env = configure_cuda()
        freeze = {"experiment": EXPERIMENT, "package_manifest_sha256": sha(ROOT / "PACKAGE_MANIFEST.json"),
                  "environment": env, "protocol_sha256": sha(ROOT / "GPU_NATIVE_PROTOCOL.md")}
        freeze_json(out / "RUN_FREEZE.json", freeze)
        run_hash = digest(freeze)
        write_json(out / "session_provenance" / (time.strftime("%Y%m%dT%H%M%S") + ".json"),
                   {"run_hash": run_hash, "platform": platform.platform(), "created_unix": time.time(),
                    "nvidia_smi": subprocess.check_output(["nvidia-smi"], text=True),
                    "gpu_unit": subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv"], text=True)})
        config = json.loads((ROOT / "configs/v2_exp006_gcn_tuning.json").read_text())
        config["experiment_id"] = EXPERIMENT
        config["execution_amendment"] = {
            "promotion_comparator": "243 canonical GPU-native EXP002 architecture reference fits",
            "historical_cpu_results_used_for_selection": False,
            "research_fit_count": 1209,
            "scientific_grid_and_training_settings": "unchanged",
        }
        config["promotion"]["if_none_pass"] = "retain EXP002 architecture; no replacement search"
        freeze_json(out / "effective_config.json", config)
        # Preserve source/protocol alongside exports without copying raw TRAIN into result exports.
        for folder in ("src", "configs", "provenance"):
            shutil.copytree(ROOT / folder, out / "reproducibility" / folder, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "PACKAGE_MANIFEST.json", out / "reproducibility/PACKAGE_MANIFEST.json")
        shutil.copy2(ROOT / "GPU_NATIVE_PROTOCOL.md", out / "reproducibility/GPU_NATIVE_PROTOCOL.md")
        install_data_guard()
        channels = preflight(config, out)
        first = plans(config, channels)
        screen = stage_run("screen", first["screen"], config, out, run_hash, lease)
        ranking, selected = screen_selection(screen, config)
        atomic_bytes(out / "tables/screen_ranking.csv", ranking.to_csv(index=False).encode())
        freeze_json(out / "selected_top2.json", selected)
        jobs = plans(config, channels, selected)
        all_rows = screen
        for stage in ("robust", "reference", "confirm"):
            all_rows += stage_run(stage, jobs[stage], config, out, run_hash, lease)
        final_tables(all_rows, selected, config, out)
        write_json(out / "STATUS.json", {"status": "COMPLETE", "fits": 1209, "run_hash": run_hash})
        lease.check()
        export_results(out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit("Stopped safely. Completed fit commits are preserved. Do not edit scientific settings to bypass an error.")
