from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "results"
    / "v2_exp001_tcn_baseline"
)

ALL_RUNS_PATH = RESULTS_DIR / "all_runs.csv"

RELATIVE_EPS = 1e-8


def classify_pair(model_mse, baseline_mse):
    """
    Compare TCN against persistence using a small numerical
    tolerance.

    Returns:
        win  -> TCN lower MSE
        loss -> TCN higher MSE
        tie  -> numerically indistinguishable
    """

    tolerance = max(
        1e-12,
        1e-6 * max(
            abs(model_mse),
            abs(baseline_mse),
        ),
    )

    delta = model_mse - baseline_mse

    if delta < -tolerance:
        return "win"

    if delta > tolerance:
        return "loss"

    return "tie"


def main():
    if not ALL_RUNS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ALL_RUNS_PATH}"
        )

    runs = pd.read_csv(ALL_RUNS_PATH)

    # ---------------------------------------------------------
    # Integrity checks.
    # ---------------------------------------------------------

    expected_runs = 243
    expected_channels = 81
    expected_seeds = {17, 29, 43}

    if len(runs) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} runs, "
            f"found {len(runs)}."
        )

    if runs["channel"].nunique() != expected_channels:
        raise RuntimeError(
            f"Expected {expected_channels} channels, "
            f"found {runs['channel'].nunique()}."
        )

    duplicates = runs.duplicated(
        ["channel", "seed"]
    )

    if duplicates.any():
        raise RuntimeError(
            "Duplicate channel/seed runs found."
        )

    seed_sets = (
        runs
        .groupby("channel")["seed"]
        .apply(set)
    )

    invalid_seed_sets = seed_sets[
        seed_sets != expected_seeds
    ]

    if len(invalid_seed_sets):
        raise RuntimeError(
            "Some channels do not contain seeds "
            "17, 29 and 43."
        )

    # ---------------------------------------------------------
    # Collapse three seeds into one row per channel.
    # ---------------------------------------------------------

    per_channel = (
        runs
        .groupby(
            ["mission", "channel"],
            as_index=False,
        )
        .agg(
            validation_mse_mean=(
                "validation_mse",
                "mean",
            ),
            validation_mse_std=(
                "validation_mse",
                "std",
            ),
            persistence_mse=(
                "persistence_validation_mse",
                "mean",
            ),
            persistence_mse_min=(
                "persistence_validation_mse",
                "min",
            ),
            persistence_mse_max=(
                "persistence_validation_mse",
                "max",
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

    # Persistence is deterministic and therefore should be
    # identical across the three seeds.
    persistence_spread = (
        per_channel["persistence_mse_max"]
        - per_channel["persistence_mse_min"]
    )

    if not np.allclose(
        persistence_spread,
        0.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Persistence MSE unexpectedly differs "
            "between seeds."
        )

    per_channel["absolute_mse_change"] = (
        per_channel["validation_mse_mean"]
        - per_channel["persistence_mse"]
    )

    per_channel[
        "relative_mse_reduction_pct"
    ] = np.where(
        per_channel["persistence_mse"]
        > RELATIVE_EPS,
        100.0
        * (
            per_channel["persistence_mse"]
            - per_channel["validation_mse_mean"]
        )
        / per_channel["persistence_mse"],
        np.nan,
    )

    per_channel["comparison"] = [
        classify_pair(model_mse, baseline_mse)
        for model_mse, baseline_mse in zip(
            per_channel["validation_mse_mean"],
            per_channel["persistence_mse"],
        )
    ]

    per_channel = per_channel.drop(
        columns=[
            "persistence_mse_min",
            "persistence_mse_max",
        ]
    )

    per_channel = per_channel.sort_values(
        ["mission", "channel"]
    ).reset_index(drop=True)

    per_channel.to_csv(
        RESULTS_DIR
        / "per_channel_summary_corrected.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Mission-level scientific summary.
    # ---------------------------------------------------------

    mission_rows = []

    for mission, frame in per_channel.groupby(
        "mission"
    ):
        model_macro_mse = float(
            frame["validation_mse_mean"].mean()
        )

        persistence_macro_mse = float(
            frame["persistence_mse"].mean()
        )

        macro_reduction = (
            100.0
            * (
                persistence_macro_mse
                - model_macro_mse
            )
            / persistence_macro_mse
        )

        valid_relative = (
            frame[
                "relative_mse_reduction_pct"
            ]
            .dropna()
        )

        comparisons = (
            frame["comparison"]
            .value_counts()
        )

        mission_rows.append(
            {
                "mission": mission,
                "channels": len(frame),

                "tcn_macro_validation_mse": (
                    model_macro_mse
                ),

                "persistence_macro_validation_mse": (
                    persistence_macro_mse
                ),

                "macro_mse_reduction_pct": (
                    macro_reduction
                ),

                "median_defined_paired_reduction_pct": (
                    float(valid_relative.median())
                    if len(valid_relative)
                    else np.nan
                ),

                "relative_pct_defined_channels": (
                    len(valid_relative)
                ),

                "relative_pct_undefined_channels": (
                    int(
                        frame[
                            "relative_mse_reduction_pct"
                        ]
                        .isna()
                        .sum()
                    )
                ),

                "wins": int(
                    comparisons.get("win", 0)
                ),

                "losses": int(
                    comparisons.get("loss", 0)
                ),

                "ties": int(
                    comparisons.get("tie", 0)
                ),

                "mean_seed_mse_std": float(
                    frame[
                        "validation_mse_std"
                    ].mean()
                ),

                "best_epoch_mean": float(
                    frame[
                        "best_epoch_mean"
                    ].mean()
                ),

                "epochs_ran_mean": float(
                    frame[
                        "epochs_ran_mean"
                    ].mean()
                ),

                "elapsed_seconds_total": float(
                    frame[
                        "elapsed_seconds_total"
                    ].sum()
                ),
            }
        )

    mission_summary = pd.DataFrame(
        mission_rows
    )

    mission_summary.to_csv(
        RESULTS_DIR
        / "mission_summary_corrected.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Seed-level macro results.
    # ---------------------------------------------------------

    seed_rows = []

    for (
        mission,
        seed,
    ), frame in runs.groupby(
        ["mission", "seed"]
    ):
        model_mse = float(
            frame["validation_mse"].mean()
        )

        persistence_mse = float(
            frame[
                "persistence_validation_mse"
            ].mean()
        )

        reduction = (
            100.0
            * (
                persistence_mse
                - model_mse
            )
            / persistence_mse
        )

        seed_rows.append(
            {
                "mission": mission,
                "seed": int(seed),
                "channels": (
                    frame["channel"].nunique()
                ),
                "tcn_macro_validation_mse": (
                    model_mse
                ),
                "persistence_macro_validation_mse": (
                    persistence_mse
                ),
                "macro_mse_reduction_pct": (
                    reduction
                ),
            }
        )

    seed_summary = pd.DataFrame(
        seed_rows
    ).sort_values(
        ["mission", "seed"]
    )

    seed_summary.to_csv(
        RESULTS_DIR
        / "mission_seed_summary_corrected.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Console output.
    # ---------------------------------------------------------

    print("=" * 76)
    print("V2 EXP001 — CORRECTED SCIENTIFIC SUMMARY")
    print("=" * 76)

    print(
        f"Runs verified:     {len(runs)}"
    )

    print(
        f"Channels verified: "
        f"{runs['channel'].nunique()}"
    )

    print(
        "Seeds verified:    "
        "17, 29, 43"
    )

    print()

    display_columns = [
        "mission",
        "channels",
        "tcn_macro_validation_mse",
        "persistence_macro_validation_mse",
        "macro_mse_reduction_pct",
        "median_defined_paired_reduction_pct",
        "relative_pct_defined_channels",
        "relative_pct_undefined_channels",
        "wins",
        "losses",
        "ties",
    ]

    print(
        mission_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\nSeed-level macro results:")

    print(
        seed_summary.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(
        "  per_channel_summary_corrected.csv"
    )
    print(
        "  mission_summary_corrected.csv"
    )
    print(
        "  mission_seed_summary_corrected.csv"
    )


if __name__ == "__main__":
    main()