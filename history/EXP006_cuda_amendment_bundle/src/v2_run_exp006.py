import argparse
import hashlib
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v2_data import TRAIN_DIR, infer_mission, list_channels
from v2_graph import build_channel_graph
from v2_models_exp006 import TCNGCNFixedTuned
from v2_train_exp006 import train_robustness, train_standard
from v2_exp006_device import (
    configure_device,
    portable_state_dict,
    write_environment_record,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "v2_exp006_gcn_tuning.json"
SCREEN_MANIFEST_PATH = ROOT / "configs" / "v2_exp006_screening_manifest.csv"
FREEZE_PATH = ROOT / "EXP006_PRETRAIN_FREEZE.json"
RESULTS_ROOT = ROOT / "results" / "v2_exp006_gcn_tuned"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "v2_exp006_gcn_tuned"
EXP002_RUNS = ROOT / "results" / "v2_exp002_tcn_gcn_fixed" / "all_runs.csv"
AMENDMENT_PATH = ROOT / "EXP006_DEVICE_AMENDMENT.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def candidate_lookup(config):
    return {c["candidate_id"]: c for c in config["candidates"]}


def verify_freeze(strict_reference_results=False):
    if not FREEZE_PATH.exists():
        raise FileNotFoundError(f"Missing freeze manifest: {FREEZE_PATH}")
    freeze = json.loads(FREEZE_PATH.read_text())

    amendment = None
    if AMENDMENT_PATH.exists():
        amendment = json.loads(AMENDMENT_PATH.read_text())
        if amendment.get("parent_freeze_id") != freeze.get("freeze_id"):
            raise RuntimeError("EXP006 device amendment does not reference the active freeze")

    patched = set((amendment or {}).get("permitted_patched_files", []))
    original_hashes = (amendment or {}).get("original_files_sha256", {})
    patched_hashes = (amendment or {}).get("patched_files_sha256", {})

    failures = []
    for rel, expected in freeze["files_sha256"].items():
        p = ROOT / rel
        if not p.exists():
            failures.append(f"missing frozen file: {rel}")
            continue
        actual = sha256_file(p)
        if rel in patched:
            if original_hashes.get(rel) != expected:
                failures.append(f"amendment original hash does not match freeze: {rel}")
            if patched_hashes.get(rel) != actual:
                failures.append(f"patched-file hash mismatch: {rel}")
        elif actual != expected:
            failures.append(f"hash mismatch: {rel}")

    if amendment is not None:
        for rel, expected in amendment.get("new_files_sha256", {}).items():
            p = ROOT / rel
            if not p.exists():
                failures.append(f"missing amendment file: {rel}")
            elif sha256_file(p) != expected:
                failures.append(f"amendment-file hash mismatch: {rel}")

    if strict_reference_results:
        ref = freeze.get("reference_artifacts_sha256", {})
        if "results/v2_exp002_tcn_gcn_fixed/all_runs.csv" in ref:
            if not EXP002_RUNS.exists():
                failures.append(f"missing EXP002 reference results: {EXP002_RUNS}")
            elif sha256_file(EXP002_RUNS) != ref[
                "results/v2_exp002_tcn_gcn_fixed/all_runs.csv"
            ]:
                failures.append("EXP002 all_runs.csv hash mismatch")

    if failures:
        raise RuntimeError("EXP006 frozen-protocol verification failed:\n- " + "\n- ".join(failures))
    return freeze, amendment

def _worker_init():
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def screen_or_confirm_paths(stage, candidate_id, channel, seed):
    stem = f"{candidate_id}__{channel}_seed{seed}"
    result = RESULTS_ROOT / stage / "runs" / f"{stem}.csv"
    checkpoint = CHECKPOINT_ROOT / stage / candidate_id / f"{channel}_seed{seed}.pt"
    return result, checkpoint


def robust_path(role, candidate_id, fold, channel, seed):
    safe = candidate_id if candidate_id else role
    stem = f"{fold}__{role}__{safe}__{channel}_seed{seed}"
    return RESULTS_ROOT / "robust" / "runs" / f"{stem}.csv"


def _job_complete(job):
    stage = job["stage"]
    if stage in {"screen", "confirm"}:
        r, c = screen_or_confirm_paths(
            stage, job["candidate"]["candidate_id"], job["channel"], job["seed"]
        )
        return r.exists() if stage == "screen" else (r.exists() and c.exists())
    if stage == "robust":
        candidate_id = (
            job["candidate"]["candidate_id"]
            if job.get("candidate") is not None
            else job["role"]
        )
        return robust_path(
            job["role"], candidate_id, job["fold"], job["channel"], job["seed"]
        ).exists()
    raise ValueError(stage)


def execute_job(job):
    stage = job["stage"]
    if stage in {"screen", "confirm"}:
        model, result = train_standard(
            channel=job["channel"],
            seed=int(job["seed"]),
            base_config=job["config"],
            candidate=job["candidate"],
            device_name=job.get("device", "cpu"),
            execution_environment_id=job.get("execution_environment_id"),
        )
        rpath, cpath = screen_or_confirm_paths(
            stage,
            job["candidate"]["candidate_id"],
            job["channel"],
            int(job["seed"]),
        )
        rpath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(rpath, index=False)
        if stage == "confirm":
            cpath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": portable_state_dict(model),
                    "result": result,
                    "candidate": job["candidate"],
                    "experiment_config": job["config"],
                },
                cpath,
            )
        return {
            "stage": stage,
            "candidate_id": result["candidate_id"],
            "channel": result["channel"],
            "seed": result["seed"],
            "metric": result["validation_mse"],
            "epochs": result["epochs_ran"],
            "elapsed": result["elapsed_seconds"],
        }

    if stage == "robust":
        result = train_robustness(
            channel=job["channel"],
            seed=int(job["seed"]),
            base_config=job["config"],
            role=job["role"],
            fold_name=job["fold"],
            candidate=job.get("candidate"),
            device_name=job.get("device", "cpu"),
            execution_environment_id=job.get("execution_environment_id"),
        )
        candidate_id = result["candidate_id"]
        rpath = robust_path(
            job["role"], candidate_id, job["fold"], job["channel"], int(job["seed"])
        )
        rpath.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(rpath, index=False)
        return {
            "stage": stage,
            "candidate_id": candidate_id,
            "channel": result["channel"],
            "seed": result["seed"],
            "fold": result["fold"],
            "role": result["role"],
            "metric": result["assessment_mse"],
            "epochs": result["epochs_ran"],
            "elapsed": result["elapsed_seconds"],
        }

    raise ValueError(stage)


def run_jobs(stage, jobs, workers, device_name="cpu", execution_environment_id=None):
    verify_freeze(strict_reference_results=False)
    device_name = str(device_name).lower()
    configure_device(device_name)

    if device_name == "cuda" and int(workers) != 1:
        raise RuntimeError(
            "EXP006 CUDA amendment requires --workers 1 (one training process per GPU)."
        )

    for job in jobs:
        job["device"] = device_name
        job["execution_environment_id"] = execution_environment_id

    pending = [j for j in jobs if not _job_complete(j)]

    print("=" * 78)
    print(f"EXP006 {stage.upper()} STAGE")
    print("=" * 78)
    print(f"Expected jobs: {len(jobs)}")
    print(f"Already complete: {len(jobs) - len(pending)}")
    print(f"Jobs this launch: {len(pending)}")
    print(f"Workers: {workers}")
    print(f"Device: {device_name}")
    print(f"Execution environment: {execution_environment_id}\n")

    if not pending:
        return []

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    failures = []
    completed = 0

    def report_success(out):
        nonlocal completed
        completed += 1
        prefix = (
            f"{out.get('fold')} | {out.get('role')} | "
            if stage == "robust"
            else ""
        )
        print(
            f"[DONE] {stage} | {prefix}{out['candidate_id']} | "
            f"{out['channel']} seed={out['seed']} | "
            f"mse={out['metric']:.8f} | epochs={out['epochs']} | "
            f"time={out['elapsed']:.1f}s",
            flush=True,
        )

    def report_failure(job, exc):
        failures.append(
            {
                "stage": stage,
                "job": {
                    k: v
                    for k, v in job.items()
                    if k not in {"config", "candidate"}
                },
                "candidate_id": (job.get("candidate", {}) or {}).get("candidate_id"),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"[FAIL] {stage} | {exc}", flush=True)

    if device_name == "cuda":
        # Avoid forked CUDA contexts. One serial training process per GPU.
        for job in pending:
            try:
                report_success(execute_job(job))
            except Exception as exc:
                report_failure(job, exc)
    else:
        with ProcessPoolExecutor(max_workers=int(workers), initializer=_worker_init) as pool:
            future_map = {pool.submit(execute_job, job): job for job in pending}
            for future in as_completed(future_map):
                job = future_map[future]
                try:
                    report_success(future.result())
                except Exception as exc:
                    report_failure(job, exc)

    print(f"\nNew jobs completed: {completed}")
    print(f"Failures: {len(failures)}")
    if failures:
        p = RESULTS_ROOT / stage / "failures.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(failures, indent=2))
    return failures

def collect_stage(stage):
    run_dir = RESULTS_ROOT / stage / "runs"
    rows = []
    if run_dir.exists():
        for p in sorted(run_dir.glob("*.csv")):
            df = pd.read_csv(p)
            if len(df) != 1:
                raise RuntimeError(f"Expected one row in {p}, found {len(df)}")
            rows.append(df.iloc[0].to_dict())
    return pd.DataFrame(rows)


def build_screen_jobs(config):
    jobs = []
    for c in config["candidates"]:
        for mission, channels in config["screening"]["channels"].items():
            for ch in channels:
                for seed in config["screening"]["seeds"]:
                    jobs.append(
                        {
                            "stage": "screen",
                            "channel": ch,
                            "seed": int(seed),
                            "candidate": c,
                            "config": config,
                        }
                    )
    return jobs


def summarise_screen(config):
    df = collect_stage("screen")
    expected = len(config["candidates"]) * 12 * len(config["screening"]["seeds"])
    if len(df) != expected:
        raise RuntimeError(f"Screening incomplete: {len(df)}/{expected}")
    if df.duplicated(["candidate_id", "channel", "seed"]).any():
        raise RuntimeError("Duplicate screening rows")
    if not np.isfinite(df["validation_mse"].astype(float)).all():
        raise RuntimeError("Non-finite screening MSE")

    summary = (
        df.groupby(["mission", "candidate_id"], as_index=False)
        .agg(
            channels=("channel", "nunique"),
            runs=("validation_mse", "size"),
            validation_mse_mean=("validation_mse", "mean"),
            validation_mse_median=("validation_mse", "median"),
            parameters_mean=("parameters", "mean"),
            elapsed_seconds_total=("elapsed_seconds", "sum"),
            ceiling_hits=("hit_epoch_ceiling", "sum"),
        )
        .sort_values(
            ["mission", "validation_mse_mean", "parameters_mean", "candidate_id"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    summary["rank_within_mission"] = summary.groupby("mission").cumcount() + 1

    out = RESULTS_ROOT / "screen"
    out.mkdir(parents=True, exist_ok=True)
    df.sort_values(["mission", "candidate_id", "channel", "seed"]).to_csv(
        out / "all_runs.csv", index=False
    )
    summary.to_csv(out / "candidate_summary.csv", index=False)

    top_n = int(config["screening"]["select_top_per_mission"])
    selected = {}
    for mission in config["dataset"]["missions"]:
        part = summary[summary["mission"] == mission].head(top_n)
        if len(part) != top_n:
            raise RuntimeError(f"Could not select {top_n} candidates for {mission}")
        selected[mission] = part["candidate_id"].tolist()
    (out / "selected_top2.json").write_text(json.dumps(selected, indent=2) + "\n")

    print("\nSCREENING RANKING")
    for mission in config["dataset"]["missions"]:
        print(f"\n{mission}")
        print(
            summary[summary["mission"] == mission][
                ["rank_within_mission", "candidate_id", "validation_mse_mean", "parameters_mean"]
            ].to_string(index=False)
        )
        print(f"Selected: {selected[mission]}")
    return selected


def _load_selected_top2():
    p = RESULTS_ROOT / "screen" / "selected_top2.json"
    if not p.exists():
        raise RuntimeError("Run and summarise screening before robustness")
    selected = json.loads(p.read_text())
    if AMENDMENT_PATH.exists():
        amendment = json.loads(AMENDMENT_PATH.read_text())
        locked = amendment.get("locked_shortlist")
        if locked is not None and selected != locked:
            raise RuntimeError(
                "Screening shortlist differs from the shortlist locked before the device migration"
            )
    return selected


def build_robust_jobs(config, selected):
    lookup = candidate_lookup(config)
    jobs = []
    for mission, channels in config["screening"]["channels"].items():
        roles = [
            ("gcn_candidate", lookup[selected[mission][0]]),
            ("gcn_candidate", lookup[selected[mission][1]]),
            ("exp002_reference", None),
            ("exp005_reference", None),
        ]
        for fold in config["robustness"]["folds"]:
            for role, candidate in roles:
                for ch in channels:
                    for seed in config["robustness"]["seeds"]:
                        jobs.append(
                            {
                                "stage": "robust",
                                "fold": fold,
                                "role": role,
                                "channel": ch,
                                "seed": int(seed),
                                "candidate": candidate,
                                "config": config,
                            }
                        )
    return jobs


def summarise_robust(config, selected):
    df = collect_stage("robust")
    expected = 4 * 12 * len(config["robustness"]["seeds"]) * len(config["robustness"]["folds"])
    if len(df) != expected:
        raise RuntimeError(f"Robustness incomplete: {len(df)}/{expected}")
    key = ["role", "candidate_id", "fold", "channel", "seed"]
    if df.duplicated(key).any():
        raise RuntimeError("Duplicate robustness rows")
    if not np.isfinite(df["assessment_mse"].astype(float)).all():
        raise RuntimeError("Non-finite robustness assessment MSE")

    summary = (
        df.groupby(["mission", "fold", "role", "candidate_id"], as_index=False)
        .agg(
            channels=("channel", "nunique"),
            runs=("assessment_mse", "size"),
            assessment_mse_mean=("assessment_mse", "mean"),
            assessment_mse_median=("assessment_mse", "median"),
            parameters_mean=("parameters", "mean"),
            elapsed_seconds_total=("elapsed_seconds", "sum"),
        )
    )

    # Relative results against refit EXP002 and selected EXP005 references.
    rows = []
    for mission in config["dataset"]["missions"]:
        for fold in config["robustness"]["folds"]:
            part = summary[(summary.mission == mission) & (summary.fold == fold)]
            exp002 = float(
                part[part.role == "exp002_reference"].iloc[0].assessment_mse_mean
            )
            exp005 = float(
                part[part.role == "exp005_reference"].iloc[0].assessment_mse_mean
            )
            for cid in selected[mission]:
                cand = float(
                    part[(part.role == "gcn_candidate") & (part.candidate_id == cid)]
                    .iloc[0]
                    .assessment_mse_mean
                )
                rows.append(
                    {
                        "mission": mission,
                        "fold": fold,
                        "candidate_id": cid,
                        "candidate_assessment_mse": cand,
                        "exp002_assessment_mse": exp002,
                        "exp005_assessment_mse": exp005,
                        "relative_improvement_pct_vs_exp002": 100.0 * (exp002 - cand) / exp002,
                        "relative_improvement_pct_vs_exp005": 100.0 * (exp005 - cand) / exp005,
                    }
                )
    relative = pd.DataFrame(rows)
    mean_rel = (
        relative.groupby(["mission", "candidate_id"], as_index=False)
        .agg(
            mean_relative_improvement_pct_vs_exp002=("relative_improvement_pct_vs_exp002", "mean"),
            mean_relative_improvement_pct_vs_exp005=("relative_improvement_pct_vs_exp005", "mean"),
        )
    )

    out = RESULTS_ROOT / "robust"
    df.sort_values(["mission", "fold", "role", "candidate_id", "channel", "seed"]).to_csv(
        out / "all_runs.csv", index=False
    )
    summary.to_csv(out / "robust_summary.csv", index=False)
    relative.to_csv(out / "robust_relative_by_fold.csv", index=False)
    mean_rel.to_csv(out / "robust_relative_mean.csv", index=False)

    print("\nROBUSTNESS RELATIVE RESULTS")
    print(relative.to_string(index=False))
    print("\nMean across R1/R2")
    print(mean_rel.to_string(index=False))
    return mean_rel


def build_confirm_jobs(config, selected):
    lookup = candidate_lookup(config)
    jobs = []
    for ch in list_channels():
        raw = np.load(TRAIN_DIR / f"{ch}.npy", mmap_mode="r")
        mission = infer_mission(raw)
        for cid in selected[mission]:
            for seed in config["confirmation"]["seeds"]:
                jobs.append(
                    {
                        "stage": "confirm",
                        "channel": ch,
                        "seed": int(seed),
                        "candidate": lookup[cid],
                        "config": config,
                    }
                )
    return jobs


def summarise_confirmation(config, selected):
    df = collect_stage("confirm")
    expected = 486
    if len(df) != expected:
        raise RuntimeError(f"Confirmation incomplete: {len(df)}/{expected}")
    if df.duplicated(["candidate_id", "channel", "seed"]).any():
        raise RuntimeError("Duplicate confirmation rows")
    if not np.isfinite(df["validation_mse"].astype(float)).all():
        raise RuntimeError("Non-finite confirmation MSE")

    summary = (
        df.groupby(["mission", "candidate_id"], as_index=False)
        .agg(
            channels=("channel", "nunique"),
            runs=("validation_mse", "size"),
            validation_mse_mean=("validation_mse", "mean"),
            validation_mse_median=("validation_mse", "median"),
            persistence_mse_mean=("persistence_validation_mse", "mean"),
            parameters_mean=("parameters", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            epochs_ran_mean=("epochs_ran", "mean"),
            ceiling_hits=("hit_epoch_ceiling", "sum"),
            elapsed_seconds_total=("elapsed_seconds", "sum"),
        )
        .sort_values(["mission", "validation_mse_mean", "candidate_id"])
        .reset_index(drop=True)
    )
    summary["rank_within_mission"] = summary.groupby("mission").cumcount() + 1

    seed_summary = (
        df.groupby(["mission", "candidate_id", "seed"], as_index=False)
        .agg(
            channels=("channel", "nunique"),
            validation_mse_mean=("validation_mse", "mean"),
        )
    )

    out = RESULTS_ROOT / "confirm"
    out.mkdir(parents=True, exist_ok=True)
    df.sort_values(["mission", "candidate_id", "channel", "seed"]).to_csv(
        out / "all_runs.csv", index=False
    )
    summary.to_csv(out / "candidate_summary.csv", index=False)
    seed_summary.to_csv(out / "candidate_seed_summary.csv", index=False)

    print("\nCONFIRMATION SUMMARY")
    print(summary.to_string(index=False))
    return df, summary, seed_summary


def load_exp002_reference(config, verify_hash=True):
    verify_freeze(strict_reference_results=verify_hash)
    if not EXP002_RUNS.exists():
        raise FileNotFoundError(EXP002_RUNS)
    df = pd.read_csv(EXP002_RUNS)
    if len(df) != 243:
        raise RuntimeError(f"EXP002 reference should contain 243 rows, found {len(df)}")
    if df.duplicated(["channel", "seed"]).any():
        raise RuntimeError("Duplicate EXP002 reference rows")
    return df


def promotion_decision(config, selected):
    robust_mean_path = RESULTS_ROOT / "robust" / "robust_relative_mean.csv"
    if not robust_mean_path.exists():
        raise RuntimeError("Robustness summary missing")
    robust_mean = pd.read_csv(robust_mean_path)
    confirm_df, confirm_summary, seed_summary = summarise_confirmation(config, selected)
    exp002 = load_exp002_reference(config, verify_hash=True)

    exp002_mission = (
        exp002.groupby("mission", as_index=False)
        .agg(validation_mse_mean=("validation_mse", "mean"))
    )
    exp002_seed = (
        exp002.groupby(["mission", "seed"], as_index=False)
        .agg(validation_mse_mean=("validation_mse", "mean"))
    )

    min_pct = float(config["promotion"]["minimum_mission_mean_improvement_pct_vs_exp002"])
    rows = []
    decisions = {}

    for mission in config["dataset"]["missions"]:
        ref_mean = float(
            exp002_mission[exp002_mission.mission == mission].iloc[0].validation_mse_mean
        )
        mission_rows = []
        for cid in selected[mission]:
            cand_mean = float(
                confirm_summary[
                    (confirm_summary.mission == mission)
                    & (confirm_summary.candidate_id == cid)
                ].iloc[0].validation_mse_mean
            )
            improvement = 100.0 * (ref_mean - cand_mean) / ref_mean

            seed_pass = True
            seed_deltas = {}
            for seed in config["confirmation"]["seeds"]:
                cseed = float(
                    seed_summary[
                        (seed_summary.mission == mission)
                        & (seed_summary.candidate_id == cid)
                        & (seed_summary.seed == seed)
                    ].iloc[0].validation_mse_mean
                )
                rseed = float(
                    exp002_seed[
                        (exp002_seed.mission == mission)
                        & (exp002_seed.seed == seed)
                    ].iloc[0].validation_mse_mean
                )
                seed_deltas[str(seed)] = 100.0 * (rseed - cseed) / rseed
                if not (cseed < rseed):
                    seed_pass = False

            robust_rel = float(
                robust_mean[
                    (robust_mean.mission == mission)
                    & (robust_mean.candidate_id == cid)
                ].iloc[0].mean_relative_improvement_pct_vs_exp002
            )
            pass_2pct = improvement >= min_pct
            pass_robust = robust_rel >= 0.0
            passes = bool(pass_2pct and seed_pass and pass_robust)

            row = {
                "mission": mission,
                "candidate_id": cid,
                "candidate_validation_mse_mean": cand_mean,
                "exp002_validation_mse_mean": ref_mean,
                "improvement_pct_vs_exp002": improvement,
                "seed17_improvement_pct_vs_exp002": seed_deltas.get("17"),
                "seed29_improvement_pct_vs_exp002": seed_deltas.get("29"),
                "seed43_improvement_pct_vs_exp002": seed_deltas.get("43"),
                "robust_mean_relative_improvement_pct_vs_exp002": robust_rel,
                "passes_2pct_rule": pass_2pct,
                "passes_all_seed_direction_rule": seed_pass,
                "passes_robustness_rule": pass_robust,
                "passes_all_promotion_rules": passes,
            }
            rows.append(row)
            mission_rows.append(row)

        passing = [r for r in mission_rows if r["passes_all_promotion_rules"]]
        if passing:
            passing.sort(key=lambda r: (r["candidate_validation_mse_mean"], r["candidate_id"]))
            best = passing[0]
            decisions[mission] = {
                "decision": "promote_exp006",
                "selected_candidate": best["candidate_id"],
                "validation_mse_mean": best["candidate_validation_mse_mean"],
                "improvement_pct_vs_exp002": best["improvement_pct_vs_exp002"],
            }
        else:
            decisions[mission] = {
                "decision": "retain_historical_exp002",
                "selected_candidate": "EXP002",
                "validation_mse_mean": ref_mean,
                "reason": "Neither shortlisted EXP006 candidate passed all prespecified promotion rules.",
            }

    decision_df = pd.DataFrame(rows)
    out = RESULTS_ROOT / "promotion"
    out.mkdir(parents=True, exist_ok=True)
    decision_df.to_csv(out / "promotion_checks.csv", index=False)
    (out / "promotion_decision.json").write_text(json.dumps(decisions, indent=2) + "\n")

    print("\nPROMOTION CHECKS")
    print(decision_df.to_string(index=False))
    print("\nFINAL EXP006 PROMOTION DECISION")
    print(json.dumps(decisions, indent=2))
    return decisions


def preflight(config, device_name="cpu"):
    freeze, amendment = verify_freeze(strict_reference_results=False)
    configure_device(device_name)

    channels = list_channels()
    if len(channels) != 81:
        raise RuntimeError("Expected 81 TRAIN channels")
    screen_channels = [
        ch for mission in config["screening"]["channels"].values() for ch in mission
    ]
    if len(screen_channels) != 12 or len(set(screen_channels)) != 12:
        raise RuntimeError("Screening roster must contain 12 unique channels")
    if not set(screen_channels).issubset(set(channels)):
        raise RuntimeError("Screening roster contains a channel outside the TRAIN study set")

    manifest = pd.read_csv(SCREEN_MANIFEST_PATH)
    if set(manifest["channel"]) != set(screen_channels) or len(manifest) != 12:
        raise RuntimeError("EXP006 screening manifest does not match frozen roster")

    candidates = config["candidates"]
    if len(candidates) != 12:
        raise RuntimeError("Expected exactly 12 candidates")
    ids = [c["candidate_id"] for c in candidates]
    if len(set(ids)) != 12:
        raise RuntimeError("Candidate IDs are not unique")

    expected_rf = {7: (1, 2), 15: (1, 2, 4), 31: (1, 2, 4, 8)}
    for rf, dilations in expected_rf.items():
        model = TCNGCNFixedTuned(
            num_nodes=2,
            hidden_size=4,
            adjacency=np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
            kernel_size=3,
            dilations=dilations,
        )
        if model.receptive_field != rf:
            raise RuntimeError(
                f"RF verification failed for {dilations}: {model.receptive_field} != {rf}"
            )
        x = torch.randn(3, 2, 64)
        y = model(x)
        if y.shape != (3,) or not torch.isfinite(y).all():
            raise RuntimeError("EXP006 model forward smoke test failed")
        if not torch.allclose(y, x[:, 0, -1], atol=1e-7, rtol=0.0):
            raise RuntimeError("EXP006 model does not initialise to persistence")

    # TRAIN-only graph construction smoke checks, no TEST/labels.
    for ch in ["C-1", "E-1"]:
        g = build_channel_graph(ch, fit_fraction=0.60)
        if g.num_nodes < 1:
            raise RuntimeError(f"{ch}: invalid graph")

    expected_screen = 288
    expected_robust = 192
    expected_confirm = 486
    if expected_screen + expected_robust + expected_confirm != 966:
        raise RuntimeError("Planned fit budget arithmetic changed")

    print("=" * 78)
    print("EXP006 PRETRAIN PREFLIGHT: PASS")
    print("=" * 78)
    print("Frozen files verified against EXP006_PRETRAIN_FREEZE.json")
    print("TRAIN study channels: 81")
    print("Screening channels: 12 (exact EXP005 roster)")
    print("Candidates: 12")
    print("Verified receptive fields: 7, 15, 31")
    print("Planned fits: 288 screen + 192 robustness + 486 confirmation = 966 max")
    print("NASA TEST arrays and anomaly labels are not used by EXP006 selection code.")
    print(f"Freeze ID: {freeze['freeze_id']}")
    if amendment is not None:
        print(f"Device amendment ID: {amendment['amendment_id']}")
    print(f"Requested device: {device_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=["preflight", "screen", "robust", "confirm", "promote"],
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    config = load_config()

    if args.stage == "screen" and args.device != "cpu":
        raise SystemExit(
            "The authoritative 288-fit screening remains the completed Mac CPU stage; "
            "do not rerun screening on CUDA."
        )

    env = write_environment_record(ROOT, args.device)
    env_id = env["execution_environment_id"]

    if args.stage == "preflight":
        preflight(config, device_name=args.device)
        print(f"Execution environment ID: {env_id}")
        return

    if args.stage == "screen":
        failures = run_jobs(
            "screen", build_screen_jobs(config), args.workers,
            device_name=args.device, execution_environment_id=env_id
        )
        if failures:
            raise SystemExit("Screening had failures; rerun after resolving them")
        summarise_screen(config)
        return

    selected = _load_selected_top2()

    if args.stage == "robust":
        failures = run_jobs(
            "robust", build_robust_jobs(config, selected), args.workers,
            device_name=args.device, execution_environment_id=env_id
        )
        if failures:
            raise SystemExit("Robustness stage had failures; rerun after resolving them")
        summarise_robust(config, selected)
        return

    if args.stage == "confirm":
        # Enforce Astra's stage order: robustness must be complete first.
        summarise_robust(config, selected)
        failures = run_jobs(
            "confirm", build_confirm_jobs(config, selected), args.workers,
            device_name=args.device, execution_environment_id=env_id
        )
        if failures:
            raise SystemExit("Confirmation had failures; rerun after resolving them")
        summarise_confirmation(config, selected)
        return

    if args.stage == "promote":
        summarise_robust(config, selected)
        promotion_decision(config, selected)
        return


if __name__ == "__main__":
    main()
