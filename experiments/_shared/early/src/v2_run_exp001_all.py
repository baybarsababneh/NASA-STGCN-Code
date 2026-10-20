import argparse
import json
import traceback
from pathlib import Path

import pandas as pd
import torch

from v2_data import list_channels
from v2_train_exp001 import train_one


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "v2_exp001_tcn_baseline.json"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "v2_exp001_tcn_baseline"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "v2_exp001_tcn_baseline"
)


def save_run(model, result, config):
    channel = result["channel"]
    seed = result["seed"]

    stem = f"{channel}_seed{seed}"

    result_path = RESULTS_DIR / f"{stem}.csv"
    checkpoint_path = CHECKPOINT_DIR / f"{stem}.pt"

    pd.DataFrame(
        [result]
    ).to_csv(
        result_path,
        index=False,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "result": result,
            "config": config,
        },
        checkpoint_path,
    )

    return result_path, checkpoint_path


def run_complete(channel, seed):
    stem = f"{channel}_seed{seed}"

    result_path = RESULTS_DIR / f"{stem}.csv"
    checkpoint_path = CHECKPOINT_DIR / f"{stem}.pt"

    return (
        result_path.exists()
        and checkpoint_path.exists()
    )


def load_completed_results(channels, seeds):
    rows = []

    for channel in channels:
        for seed in seeds:
            path = (
                RESULTS_DIR
                / f"{channel}_seed{seed}.csv"
            )

            if path.exists():
                frame = pd.read_csv(path)

                if len(frame) != 1:
                    raise RuntimeError(
                        f"Expected one result row in {path}"
                    )

                rows.append(frame.iloc[0].to_dict())

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def build_summaries(channels, seeds):
    all_runs = load_completed_results(
        channels,
        seeds,
    )

    if all_runs.empty:
        print("No completed runs available for summary.")
        return

    all_runs = all_runs.sort_values(
        ["mission", "channel", "seed"]
    ).reset_index(drop=True)

    all_runs.to_csv(
        RESULTS_DIR / "all_runs.csv",
        index=False,
    )

    numeric_metrics = [
        "validation_mse",
        "persistence_validation_mse",
        "relative_mse_reduction_pct",
        "best_epoch",
        "epochs_ran",
        "elapsed_seconds",
    ]

    mission_seed_summary = (
        all_runs
        .groupby(
            ["mission", "seed"],
            as_index=False,
        )[numeric_metrics]
        .mean()
    )

    mission_seed_summary.to_csv(
        RESULTS_DIR / "mission_seed_summary.csv",
        index=False,
    )

    mission_summary = (
        all_runs
        .groupby(
            "mission",
            as_index=False,
        )
        .agg(
            channels=("channel", "nunique"),
            runs=("channel", "size"),
            validation_mse_mean=(
                "validation_mse",
                "mean",
            ),
            validation_mse_median=(
                "validation_mse",
                "median",
            ),
            persistence_mse_mean=(
                "persistence_validation_mse",
                "mean",
            ),
            relative_reduction_mean_pct=(
                "relative_mse_reduction_pct",
                "mean",
            ),
            relative_reduction_median_pct=(
                "relative_mse_reduction_pct",
                "median",
            ),
            best_epoch_mean=(
                "best_epoch",
                "mean",
            ),
            epochs_ran_mean=(
                "epochs_ran",
                "mean",
            ),
            elapsed_seconds_total=(
                "elapsed_seconds",
                "sum",
            ),
        )
    )

    mission_summary.to_csv(
        RESULTS_DIR / "mission_summary.csv",
        index=False,
    )

    expected_runs = (
        len(channels)
        * len(seeds)
    )

    print("\n" + "=" * 70)
    print("V2 EXP001 SUMMARY")
    print("=" * 70)

    print(
        f"Completed runs: "
        f"{len(all_runs)} / {expected_runs}"
    )

    print("\nMission summary:")
    print(
        mission_summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
        "\n  all_runs.csv"
        "\n  mission_seed_summary.csv"
        "\n  mission_summary.csv"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Optional maximum number of NEW fits. "
            "Useful for testing the runner."
        ),
    )

    args = parser.parse_args()

    with open(CONFIG_PATH, "r") as handle:
        config = json.load(handle)

    channels = list_channels()
    seeds = [
        int(seed)
        for seed in config["training"]["seeds"]
    ]

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    failures = []
    new_runs = 0

    total_expected = (
        len(channels)
        * len(seeds)
    )

    completed_before = sum(
        run_complete(channel, seed)
        for channel in channels
        for seed in seeds
    )

    print("=" * 70)
    print("V2 EXP001: FULL TCN BASELINE")
    print("=" * 70)

    print(f"Channels:        {len(channels)}")
    print(f"Seeds:           {seeds}")
    print(f"Expected fits:   {total_expected}")
    print(f"Already complete:{completed_before}")
    print()

    for channel_index, channel in enumerate(
        channels,
        start=1,
    ):
        for seed in seeds:
            if run_complete(channel, seed):
                print(
                    f"[SKIP] "
                    f"{channel} seed={seed}"
                )
                continue

            if (
                args.max_runs is not None
                and new_runs >= args.max_runs
            ):
                print(
                    "\nReached --max-runs limit."
                )

                build_summaries(
                    channels,
                    seeds,
                )
                return

            print("\n" + "-" * 70)

            print(
                f"[RUN] channel "
                f"{channel_index}/{len(channels)} "
                f"| {channel} "
                f"| seed={seed}"
            )

            print("-" * 70)

            try:
                model, result = train_one(
                    channel=channel,
                    seed=seed,
                    config=config,
                    smoke_epochs=None,
                )

                result_path, checkpoint_path = (
                    save_run(
                        model,
                        result,
                        config,
                    )
                )

                new_runs += 1

                print(
                    f"\n[SAVED] {result_path.name}"
                )

                print(
                    f"[SAVED] {checkpoint_path.name}"
                )

            except Exception as exc:
                print(
                    f"\n[FAILED] "
                    f"{channel} seed={seed}"
                )

                print(
                    f"{type(exc).__name__}: {exc}"
                )

                failures.append(
                    {
                        "channel": channel,
                        "seed": seed,
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error": str(exc),
                        "traceback": (
                            traceback.format_exc()
                        ),
                    }
                )

    if failures:
        pd.DataFrame(
            failures
        ).to_csv(
            RESULTS_DIR / "failures.csv",
            index=False,
        )

        print(
            f"\nWARNING: "
            f"{len(failures)} fits failed."
        )

        print(
            "See results/"
            "v2_exp001_tcn_baseline/"
            "failures.csv"
        )

    build_summaries(
        channels,
        seeds,
    )


if __name__ == "__main__":
    main()