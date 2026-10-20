import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from v2_data import list_channels
from v2_train_exp005 import train_one


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "v2_exp005_command_tcn_tuning.json"
RESULTS_ROOT = ROOT / "results" / "v2_exp005_command_tcn_tuned"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "v2_exp005_command_tcn_tuned"


def load_config():
    with open(CONFIG_PATH, "r") as handle:
        return json.load(handle)


def candidate_lookup(config):
    return {
        c["candidate_id"]: c
        for c in config["candidates"]
    }


def _worker_init():
    # Two independent CPU workers gave near-ideal parallelism for EXP004.
    # Keep each worker single-threaded to avoid oversubscription.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def run_paths(stage, candidate_id, channel, seed):
    stem = f"{candidate_id}__{channel}_seed{seed}"
    result_dir = RESULTS_ROOT / stage / "runs"
    checkpoint_dir = CHECKPOINT_ROOT / stage / candidate_id
    result_path = result_dir / f"{stem}.csv"
    checkpoint_path = checkpoint_dir / f"{channel}_seed{seed}.pt"
    return result_path, checkpoint_path


def job_complete(stage, candidate_id, channel, seed):
    result_path, checkpoint_path = run_paths(
        stage, candidate_id, channel, seed
    )
    if stage == "screen":
        return result_path.exists()
    return result_path.exists() and checkpoint_path.exists()


def execute_job(payload):
    stage = payload["stage"]
    channel = payload["channel"]
    seed = int(payload["seed"])
    candidate = payload["candidate"]
    config = payload["config"]

    model, result = train_one(
        channel=channel,
        seed=seed,
        base_config=config,
        candidate=candidate,
        smoke_epochs=None,
    )

    result_path, checkpoint_path = run_paths(
        stage,
        candidate["candidate_id"],
        channel,
        seed,
    )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(result_path, index=False)

    if stage == "confirm":
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "result": result,
                "candidate": candidate,
                "experiment_config": config,
            },
            checkpoint_path,
        )

    return {
        "stage": stage,
        "candidate_id": candidate["candidate_id"],
        "channel": channel,
        "mission": result["mission"],
        "seed": seed,
        "validation_mse": result["validation_mse"],
        "epochs_ran": result["epochs_ran"],
        "elapsed_seconds": result["elapsed_seconds"],
    }


def run_jobs(stage, jobs, workers):
    pending = []
    for job in jobs:
        cid = job["candidate"]["candidate_id"]
        if job_complete(stage, cid, job["channel"], job["seed"]):
            continue
        pending.append(job)

    print("=" * 76)
    print(f"EXP005 {stage.upper()} STAGE")
    print("=" * 76)
    print(f"Expected jobs: {len(jobs)}")
    print(f"Already complete: {len(jobs) - len(pending)}")
    print(f"Jobs this launch: {len(pending)}")
    print(f"Workers: {workers}")
    print()

    failures = []
    completed = 0

    if not pending:
        return failures

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_worker_init,
    ) as pool:
        future_map = {
            pool.submit(execute_job, job): job
            for job in pending
        }

        for future in as_completed(future_map):
            job = future_map[future]
            try:
                out = future.result()
                completed += 1
                print(
                    f"[DONE] {out['stage']} | {out['candidate_id']} | "
                    f"{out['channel']} seed={out['seed']} | "
                    f"val_mse={out['validation_mse']:.8f} | "
                    f"epochs={out['epochs_ran']} | "
                    f"time={out['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                failures.append({
                    "stage": stage,
                    "candidate_id": job["candidate"]["candidate_id"],
                    "channel": job["channel"],
                    "seed": int(job["seed"]),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                })
                print(
                    f"[FAIL] {stage} | "
                    f"{job['candidate']['candidate_id']} | "
                    f"{job['channel']} seed={job['seed']} | {exc}",
                    flush=True,
                )

    print()
    print(f"New jobs completed: {completed}")
    print(f"Failures: {len(failures)}")

    if failures:
        fail_path = RESULTS_ROOT / stage / "failures.json"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        fail_path.write_text(json.dumps(failures, indent=2))

    return failures


def collect_stage(stage):
    run_dir = RESULTS_ROOT / stage / "runs"
    rows = []
    if run_dir.exists():
        for path in sorted(run_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            if len(frame) != 1:
                raise RuntimeError(
                    f"Expected one row in {path}, found {len(frame)}"
                )
            rows.append(frame.iloc[0].to_dict())
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarise_screen(config):
    df = collect_stage("screen")
    expected = (
        len(config["candidates"])
        * sum(len(v) for v in config["screening"]["channels"].values())
        * len(config["screening"]["seeds"])
    )
    if len(df) != expected:
        raise RuntimeError(
            f"Screening incomplete: found {len(df)} rows, expected {expected}"
        )

    if df.duplicated(["candidate_id", "channel", "seed"]).any():
        raise RuntimeError("Duplicate screening candidate/channel/seed rows")
    if not np.isfinite(df["validation_mse"].astype(float)).all():
        raise RuntimeError("Non-finite screening validation MSE")

    summary = (
        df.groupby(["mission", "candidate_id"], as_index=False)
        .agg(
            channels=("channel", "nunique"),
            runs=("validation_mse", "size"),
            validation_mse_mean=("validation_mse", "mean"),
            validation_mse_median=("validation_mse", "median"),
            elapsed_seconds_total=("elapsed_seconds", "sum"),
        )
        .sort_values(["mission", "validation_mse_mean", "candidate_id"])
        .reset_index(drop=True)
    )

    summary["rank_within_mission"] = (
        summary.groupby("mission")["validation_mse_mean"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    out_dir = RESULTS_ROOT / "screen"
    df.sort_values(["mission", "candidate_id", "channel", "seed"]).to_csv(
        out_dir / "all_runs.csv", index=False
    )
    summary.to_csv(out_dir / "candidate_summary.csv", index=False)

    top_n = int(config["screening"]["select_top_per_mission"])
    selected = {}
    for mission in config["dataset"]["missions"]:
        part = summary[summary["mission"] == mission].head(top_n)
        if len(part) != top_n:
            raise RuntimeError(f"Could not select {top_n} {mission} candidates")
        selected[mission] = part["candidate_id"].tolist()

    (out_dir / "selected_top2.json").write_text(
        json.dumps(selected, indent=2) + "\n"
    )

    print("\nSCREENING RANKING")
    for mission in config["dataset"]["missions"]:
        print(f"\n{mission}")
        print(
            summary[summary["mission"] == mission]
            [["rank_within_mission", "candidate_id", "validation_mse_mean"]]
            .to_string(index=False)
        )
        print(f"Selected top 2: {selected[mission]}")

    return selected


def build_screen_jobs(config):
    jobs = []
    for candidate in config["candidates"]:
        for mission, channels in config["screening"]["channels"].items():
            for channel in channels:
                for seed in config["screening"]["seeds"]:
                    jobs.append({
                        "stage": "screen",
                        "channel": channel,
                        "seed": int(seed),
                        "candidate": candidate,
                        "config": config,
                    })
    return jobs


def build_confirm_jobs(config, selected):
    lookup = candidate_lookup(config)
    channels = list_channels()

    # Mission is known from the fixed screening manifest for screening, but
    # confirmation needs all channels. Infer mission without touching TEST or
    # anomaly labels via the training function's normal data loader. To avoid
    # an extra load here, use mission-prefix membership from a tiny cached
    # helper run is unnecessary: selected candidates are submitted for every
    # channel and train_one reports mission. Instead we need mission-specific
    # candidate sets before submission, so infer using TRAIN array dimensions.
    from v2_data import TRAIN_DIR, infer_mission

    jobs = []
    for channel in channels:
        raw = np.load(TRAIN_DIR / f"{channel}.npy", mmap_mode="r")
        mission = infer_mission(raw)
        for candidate_id in selected[mission]:
            candidate = lookup[candidate_id]
            for seed in config["confirmation"]["seeds"]:
                jobs.append({
                    "stage": "confirm",
                    "channel": channel,
                    "seed": int(seed),
                    "candidate": candidate,
                    "config": config,
                })
    return jobs


def summarise_confirmation(config, selected):
    df = collect_stage("confirm")
    expected = 0
    from v2_data import TRAIN_DIR, infer_mission
    for channel in list_channels():
        raw = np.load(TRAIN_DIR / f"{channel}.npy", mmap_mode="r")
        mission = infer_mission(raw)
        expected += (
            len(selected[mission])
            * len(config["confirmation"]["seeds"])
        )

    if len(df) != expected:
        raise RuntimeError(
            f"Confirmation incomplete: found {len(df)} rows, expected {expected}"
        )
    if df.duplicated(["candidate_id", "channel", "seed"]).any():
        raise RuntimeError("Duplicate confirmation candidate/channel/seed rows")
    if not np.isfinite(df["validation_mse"].astype(float)).all():
        raise RuntimeError("Non-finite confirmation validation MSE")

    summary = (
        df.groupby(["mission", "candidate_id"], as_index=False)
        .agg(
            channels=("channel", "nunique"),
            runs=("validation_mse", "size"),
            validation_mse_mean=("validation_mse", "mean"),
            validation_mse_median=("validation_mse", "median"),
            persistence_mse_mean=("persistence_validation_mse", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            epochs_ran_mean=("epochs_ran", "mean"),
            elapsed_seconds_total=("elapsed_seconds", "sum"),
        )
        .sort_values(["mission", "validation_mse_mean", "candidate_id"])
        .reset_index(drop=True)
    )

    summary["macro_mse_reduction_pct"] = (
        100.0
        * (summary["persistence_mse_mean"] - summary["validation_mse_mean"])
        / summary["persistence_mse_mean"]
    )
    summary["rank_within_mission"] = (
        summary.groupby("mission")["validation_mse_mean"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    final_selected = {}
    for mission in config["dataset"]["missions"]:
        part = summary[summary["mission"] == mission]
        if len(part) != len(selected[mission]):
            raise RuntimeError(f"Unexpected confirmation candidate count for {mission}")
        final_selected[mission] = part.iloc[0]["candidate_id"]

    final_parts = []
    for mission, cid in final_selected.items():
        final_parts.append(
            df[(df["mission"] == mission) & (df["candidate_id"] == cid)].copy()
        )
    final_df = pd.concat(final_parts, ignore_index=True)

    if len(final_df) != 243:
        raise RuntimeError(
            f"Final selected rows should be 243, found {len(final_df)}"
        )

    mission_seed_summary = (
        final_df.groupby(["mission", "seed"], as_index=False)
        .agg(
            channels=("channel", "nunique"),
            runs=("validation_mse", "size"),
            validation_mse_mean=("validation_mse", "mean"),
            persistence_mse_mean=("persistence_validation_mse", "mean"),
        )
    )
    mission_seed_summary["macro_mse_reduction_pct"] = (
        100.0
        * (
            mission_seed_summary["persistence_mse_mean"]
            - mission_seed_summary["validation_mse_mean"]
        )
        / mission_seed_summary["persistence_mse_mean"]
    )

    mission_summary = (
        final_df.groupby("mission", as_index=False)
        .agg(
            channels=("channel", "nunique"),
            runs=("validation_mse", "size"),
            validation_mse_mean=("validation_mse", "mean"),
            validation_mse_median=("validation_mse", "median"),
            persistence_mse_mean=("persistence_validation_mse", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            epochs_ran_mean=("epochs_ran", "mean"),
            elapsed_seconds_total=("elapsed_seconds", "sum"),
        )
    )
    mission_summary["macro_mse_reduction_pct"] = (
        100.0
        * (mission_summary["persistence_mse_mean"] - mission_summary["validation_mse_mean"])
        / mission_summary["persistence_mse_mean"]
    )

    confirm_dir = RESULTS_ROOT / "confirm"
    final_dir = RESULTS_ROOT / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    df.sort_values(["mission", "candidate_id", "channel", "seed"]).to_csv(
        confirm_dir / "all_runs.csv", index=False
    )
    summary.to_csv(confirm_dir / "candidate_summary.csv", index=False)
    (confirm_dir / "final_selected_by_mission.json").write_text(
        json.dumps(final_selected, indent=2) + "\n"
    )

    final_df.sort_values(["mission", "channel", "seed"]).to_csv(
        final_dir / "all_runs.csv", index=False
    )
    mission_seed_summary.to_csv(
        final_dir / "mission_seed_summary.csv", index=False
    )
    mission_summary.to_csv(
        final_dir / "mission_summary.csv", index=False
    )
    (final_dir / "selected_config_by_mission.json").write_text(
        json.dumps(final_selected, indent=2) + "\n"
    )

    print("\nCONFIRMATION SUMMARY")
    print(summary.to_string(index=False))
    print("\nFINAL SELECTED CONFIGURATION BY MISSION")
    print(json.dumps(final_selected, indent=2))
    print("\nFINAL EXP005 MISSION SUMMARY")
    print(mission_summary.to_string(index=False))

    return final_selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "screen", "confirm"],
        default="all",
        help="Run screening only, confirmation only, or both sequentially.",
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    config = load_config()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.stage in {"all", "screen"}:
        screen_jobs = build_screen_jobs(config)
        failures = run_jobs("screen", screen_jobs, args.workers)
        if failures:
            raise RuntimeError(
                "Screening had failures. Re-run the same command to resume; "
                "do not proceed to confirmation until all screening jobs pass."
            )
        selected = summarise_screen(config)
    else:
        selected_path = RESULTS_ROOT / "screen" / "selected_top2.json"
        if not selected_path.exists():
            raise FileNotFoundError(
                "Missing screening selection. Run --stage screen first."
            )
        selected = json.loads(selected_path.read_text())

    if args.stage in {"all", "confirm"}:
        confirm_jobs = build_confirm_jobs(config, selected)
        failures = run_jobs("confirm", confirm_jobs, args.workers)
        if failures:
            raise RuntimeError(
                "Confirmation had failures. Re-run the same command to resume."
            )
        summarise_confirmation(config, selected)

    print("\n" + "=" * 76)
    print("EXP005 REQUESTED STAGE(S) COMPLETE")
    print("=" * 76)
    print("No NASA TEST arrays or anomaly labels were accessed by this runner.")


if __name__ == "__main__":
    main()
