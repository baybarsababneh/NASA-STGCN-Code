import argparse
import json
import traceback
from pathlib import Path

import pandas as pd
import torch

from v2_data import list_channels
from v2_train_exp002 import train_one


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "v2_exp002_tcn_gcn_fixed.json"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "v2_exp002_tcn_gcn_fixed"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "v2_exp002_tcn_gcn_fixed"
)


def result_path(
    channel,
    seed,
):
    return (
        RESULTS_DIR
        / f"{channel}_seed{seed}.csv"
    )


def checkpoint_path(
    channel,
    seed,
):
    return (
        CHECKPOINT_DIR
        / f"{channel}_seed{seed}.pt"
    )


def run_complete(
    channel,
    seed,
):
    return (
        result_path(
            channel,
            seed,
        ).exists()
        and
        checkpoint_path(
            channel,
            seed,
        ).exists()
    )


def save_run(
    channel,
    seed,
    model,
    result,
    config,
):
    csv_path = result_path(
        channel,
        seed,
    )

    pt_path = checkpoint_path(
        channel,
        seed,
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        csv_path,
        index=False,
    )

    torch.save(
        {
            "state_dict": (
                model.state_dict()
            ),
            "result": result,
            "config": config,
        },
        pt_path,
    )

    print(
        f"\n[SAVED] {csv_path.name}"
    )

    print(
        f"[SAVED] {pt_path.name}"
    )


def load_completed_results(
    channels,
    seeds,
):
    rows = []

    for channel in channels:
        for seed in seeds:
            path = result_path(
                channel,
                seed,
            )

            checkpoint = (
                checkpoint_path(
                    channel,
                    seed,
                )
            )

            if (
                path.exists()
                and
                checkpoint.exists()
            ):
                frame = pd.read_csv(
                    path
                )

                if len(frame) != 1:
                    raise RuntimeError(
                        f"{path.name} should contain "
                        "exactly one row."
                    )

                rows.append(
                    frame.iloc[0].to_dict()
                )

    return pd.DataFrame(
        rows
    )


def build_summaries(
    channels,
    seeds,
):
    runs = load_completed_results(
        channels,
        seeds,
    )

    if len(runs) == 0:
        return

    runs = runs.sort_values(
        [
            "mission",
            "channel",
            "seed",
        ]
    ).reset_index(
        drop=True
    )

    runs.to_csv(
        RESULTS_DIR
        / "all_runs.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Seed-level macro MSE.
    #
    # This uses ratio-of-means rather than averaging unstable
    # per-channel percentage reductions.
    # ---------------------------------------------------------

    seed_rows = []

    for (
        mission,
        seed,
    ), frame in runs.groupby(
        [
            "mission",
            "seed",
        ]
    ):
        model_mse = float(
            frame[
                "validation_mse"
            ].mean()
        )

        persistence_mse = float(
            frame[
                "persistence_validation_mse"
            ].mean()
        )

        if persistence_mse > 0:
            macro_reduction = (
                100.0
                * (
                    persistence_mse
                    - model_mse
                )
                / persistence_mse
            )
        else:
            macro_reduction = float(
                "nan"
            )

        seed_rows.append(
            {
                "mission": mission,
                "seed": int(seed),

                "channels": int(
                    frame[
                        "channel"
                    ].nunique()
                ),

                "runs": len(
                    frame
                ),

                "validation_mse_mean": (
                    model_mse
                ),

                "persistence_mse_mean": (
                    persistence_mse
                ),

                "macro_mse_reduction_pct": (
                    macro_reduction
                ),
            }
        )

    seed_summary = pd.DataFrame(
        seed_rows
    ).sort_values(
        [
            "mission",
            "seed",
        ]
    )

    seed_summary.to_csv(
        RESULTS_DIR
        / "mission_seed_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Mission-level current-run summary.
    #
    # We do NOT average the stored per-channel percentage
    # reductions.
    # ---------------------------------------------------------

    mission_rows = []

    for mission, frame in runs.groupby(
        "mission"
    ):
        model_mse = float(
            frame[
                "validation_mse"
            ].mean()
        )

        persistence_mse = float(
            frame[
                "persistence_validation_mse"
            ].mean()
        )

        if persistence_mse > 0:
            macro_reduction = (
                100.0
                * (
                    persistence_mse
                    - model_mse
                )
                / persistence_mse
            )
        else:
            macro_reduction = float(
                "nan"
            )

        mission_rows.append(
            {
                "mission": mission,

                "channels": int(
                    frame[
                        "channel"
                    ].nunique()
                ),

                "runs": len(
                    frame
                ),

                "validation_mse_mean": (
                    model_mse
                ),

                "validation_mse_median": float(
                    frame[
                        "validation_mse"
                    ].median()
                ),

                "persistence_mse_mean": (
                    persistence_mse
                ),

                "macro_mse_reduction_pct": (
                    macro_reduction
                ),

                "best_epoch_mean": float(
                    frame[
                        "best_epoch"
                    ].mean()
                ),

                "epochs_ran_mean": float(
                    frame[
                        "epochs_ran"
                    ].mean()
                ),

                "elapsed_seconds_total": float(
                    frame[
                        "elapsed_seconds"
                    ].sum()
                ),
            }
        )

    mission_summary = pd.DataFrame(
        mission_rows
    ).sort_values(
        "mission"
    )

    mission_summary.to_csv(
        RESULTS_DIR
        / "mission_summary.csv",
        index=False,
    )

    expected = (
        len(channels)
        * len(seeds)
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "V2 EXP002 SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Completed runs: "
        f"{len(runs)} / {expected}"
    )

    print(
        "\nMission summary:"
    )

    print(
        mission_summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved:"
    )

    print(
        "  all_runs.csv"
    )

    print(
        "  mission_seed_summary.csv"
    )

    print(
        "  mission_summary.csv"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Optional maximum number of new runs. "
            "Useful for testing the runner."
        ),
    )

    args = parser.parse_args()

    with open(
        CONFIG_PATH,
        "r",
    ) as handle:
        config = json.load(
            handle
        )

    channels = list_channels()

    seeds = list(
        config[
            "training"
        ][
            "seeds"
        ]
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_runs = (
        len(channels)
        * len(seeds)
    )

    already_complete = sum(
        run_complete(
            channel,
            seed,
        )
        for channel in channels
        for seed in seeds
    )

    print(
        "=" * 70
    )

    print(
        "V2 EXP002: FULL FIXED-GRAPH TCN-GCN"
    )

    print(
        "=" * 70
    )

    print(
        f"Channels:        "
        f"{len(channels)}"
    )

    print(
        f"Seeds:           "
        f"{seeds}"
    )

    print(
        f"Expected fits:   "
        f"{expected_runs}"
    )

    print(
        f"Already complete:"
        f"{already_complete}"
    )

    new_runs = 0
    failures = []

    stop_requested = False

    for channel_index, channel in enumerate(
        channels,
        start=1,
    ):
        for seed in seeds:
            if run_complete(
                channel,
                seed,
            ):
                continue

            if (
                args.max_runs is not None
                and
                new_runs >= args.max_runs
            ):
                stop_requested = True
                break

            print(
                "\n"
                + "-" * 70
            )

            print(
                f"[RUN] channel "
                f"{channel_index}/{len(channels)} "
                f"| {channel} "
                f"| seed={seed}"
            )

            print(
                "-" * 70
            )

            try:
                model, result = train_one(
                    channel=channel,
                    seed=seed,
                    config=config,
                    smoke_epochs=None,
                )

                save_run(
                    channel=channel,
                    seed=seed,
                    model=model,
                    result=result,
                    config=config,
                )

                new_runs += 1

            except Exception as exc:
                print(
                    f"\n[FAILED] "
                    f"{channel} "
                    f"seed={seed}"
                )

                print(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                failures.append(
                    {
                        "channel": channel,
                        "seed": seed,
                        "error_type": (
                            type(exc).__name__
                        ),
                        "error": str(
                            exc
                        ),
                        "traceback": (
                            traceback.format_exc()
                        ),
                    }
                )

        if stop_requested:
            break

    # ---------------------------------------------------------
    # Failure log.
    # ---------------------------------------------------------

    failure_path = (
        RESULTS_DIR
        / "failures.csv"
    )

    if failures:
        pd.DataFrame(
            failures
        ).to_csv(
            failure_path,
            index=False,
        )

        print(
            f"\nWARNING: "
            f"{len(failures)} fits failed."
        )

        print(
            f"See {failure_path}"
        )

    else:
        # Remove an obsolete failure file from an earlier
        # interrupted/resumed execution.
        if failure_path.exists():
            failure_path.unlink()

    # Always rebuild summaries from canonical completed runs.
    build_summaries(
        channels,
        seeds,
    )


if __name__ == "__main__":
    main()