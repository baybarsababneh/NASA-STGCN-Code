from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EXP002 = "v2_exp002_tcn_gcn_fixed"
EXP003 = "v2_exp003_tcn_no_graph"

RESULT002 = ROOT / "results" / EXP002
RESULT003 = ROOT / "results" / EXP003

CHECK002 = ROOT / "checkpoints" / EXP002
CHECK003 = ROOT / "checkpoints" / EXP003

AUDIT_DIR = ROOT / "audits"
COMPARE_DIR = ROOT / "comparisons" / "v2_exp002_vs_exp003"

EXPECTED_CHANNELS = 81
EXPECTED_RUNS = 243
EXPECTED_SEEDS = {17, 29, 43}

EXPECTED_MISSION_CHANNELS = {
    "MSL": 27,
    "SMAP": 54,
}

EXPECTED_MISSION_RUNS = {
    "MSL": 81,
    "SMAP": 162,
}

RELATIVE_EPS = 1e-8
BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20261001


failures = []
report = []


def log(message=""):
    print(message)
    report.append(str(message))


def require(condition, message):
    if not condition:
        failures.append(message)
        log(f"[FAIL] {message}")


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def canonical_key(path, extension):
    pattern = (
        rf"^(.+)_seed(\d+)\.{extension}$"
    )

    match = re.fullmatch(
        pattern,
        path.name,
    )

    if match is None:
        return None

    return (
        match.group(1),
        int(match.group(2)),
    )


def load_runs(result_dir):
    path = result_dir / "all_runs.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def audit_experiment(
    name,
    result_dir,
    checkpoint_dir,
):
    log()
    log("=" * 72)
    log(f"AUDIT: {name}")
    log("=" * 72)

    df = load_runs(
        result_dir
    )

    required = {
        "experiment_id",
        "channel",
        "mission",
        "seed",
        "validation_mse",
        "persistence_validation_mse",
        "best_epoch",
        "epochs_ran",
        "elapsed_seconds",
        "graph_num_nodes",
        "hidden_size",
        "lookback",
        "parameters",
        "fit_samples",
        "validation_samples",
        "calibration_samples",
    }

    missing = (
        required
        - set(df.columns)
    )

    require(
        not missing,
        f"{name}: missing columns "
        f"{sorted(missing)}",
    )

    require(
        len(df) == EXPECTED_RUNS,
        f"{name}: expected "
        f"{EXPECTED_RUNS} rows, "
        f"found {len(df)}",
    )

    df["channel"] = (
        df["channel"]
        .astype(str)
    )

    df["mission"] = (
        df["mission"]
        .astype(str)
    )

    df["seed"] = (
        df["seed"]
        .astype(int)
    )

    duplicate_mask = (
        df.duplicated(
            ["channel", "seed"],
            keep=False,
        )
    )

    require(
        not duplicate_mask.any(),
        f"{name}: duplicate "
        "channel/seed rows found",
    )

    channels = (
        df["channel"]
        .unique()
    )

    require(
        len(channels)
        == EXPECTED_CHANNELS,
        f"{name}: expected "
        f"{EXPECTED_CHANNELS} channels, "
        f"found {len(channels)}",
    )

    seeds = set(
        df["seed"].unique()
    )

    require(
        seeds == EXPECTED_SEEDS,
        f"{name}: expected seeds "
        f"{sorted(EXPECTED_SEEDS)}, "
        f"found {sorted(seeds)}",
    )

    for channel, frame in (
        df.groupby("channel")
    ):
        channel_seeds = set(
            frame["seed"]
            .astype(int)
        )

        require(
            len(frame) == 3
            and
            channel_seeds
            == EXPECTED_SEEDS,
            f"{name}: {channel} "
            "does not contain exactly "
            "three expected seeds",
        )

    for mission in [
        "MSL",
        "SMAP",
    ]:
        frame = df[
            df["mission"]
            == mission
        ]

        channels_mission = (
            frame["channel"]
            .nunique()
        )

        require(
            channels_mission
            ==
            EXPECTED_MISSION_CHANNELS[
                mission
            ],
            f"{name}: {mission} "
            "channel count mismatch",
        )

        require(
            len(frame)
            ==
            EXPECTED_MISSION_RUNS[
                mission
            ],
            f"{name}: {mission} "
            "run count mismatch",
        )

    numeric_columns = [
        "validation_mse",
        "persistence_validation_mse",
        "best_epoch",
        "epochs_ran",
        "elapsed_seconds",
        "graph_num_nodes",
        "hidden_size",
        "lookback",
        "parameters",
        "fit_samples",
        "validation_samples",
        "calibration_samples",
    ]

    for column in numeric_columns:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        require(
            np.isfinite(
                values
            ).all(),
            f"{name}: non-finite "
            f"value in {column}",
        )

    require(
        (
            df["validation_mse"]
            >= 0
        ).all(),
        f"{name}: negative "
        "validation MSE",
    )

    require(
        (
            df[
                "persistence_validation_mse"
            ]
            >= 0
        ).all(),
        f"{name}: negative "
        "persistence MSE",
    )

    require(
        (
            df["elapsed_seconds"]
            > 0
        ).all(),
        f"{name}: non-positive "
        "elapsed time",
    )

    require(
        (
            df["best_epoch"]
            >= 1
        ).all(),
        f"{name}: invalid best_epoch",
    )

    require(
        (
            df["epochs_ran"]
            >= df["best_epoch"]
        ).all(),
        f"{name}: best_epoch exceeds "
        "epochs_ran",
    )

    require(
        (
            df["epochs_ran"]
            <= 80
        ).all(),
        f"{name}: epochs_ran exceeds 80",
    )

    result_files = []

    for path in result_dir.glob(
        "*.csv"
    ):
        key = canonical_key(
            path,
            "csv",
        )

        if key is not None:
            result_files.append(
                (path, key)
            )

    checkpoint_files = []

    for path in checkpoint_dir.glob(
        "*.pt"
    ):
        key = canonical_key(
            path,
            "pt",
        )

        if key is not None:
            checkpoint_files.append(
                (path, key)
            )

    canonical_keys = {
        (
            str(row.channel),
            int(row.seed),
        )
        for row in df.itertuples(
            index=False
        )
    }

    result_keys = {
        key
        for _, key
        in result_files
    }

    checkpoint_keys = {
        key
        for _, key
        in checkpoint_files
    }

    require(
        len(result_files)
        == EXPECTED_RUNS,
        f"{name}: expected "
        f"{EXPECTED_RUNS} canonical "
        f"result CSVs, found "
        f"{len(result_files)}",
    )

    require(
        len(checkpoint_files)
        == EXPECTED_RUNS,
        f"{name}: expected "
        f"{EXPECTED_RUNS} checkpoints, "
        f"found "
        f"{len(checkpoint_files)}",
    )

    require(
        result_keys
        == canonical_keys,
        f"{name}: canonical result "
        "file keys disagree with "
        "all_runs.csv",
    )

    require(
        checkpoint_keys
        == canonical_keys,
        f"{name}: checkpoint keys "
        "disagree with all_runs.csv",
    )

    log(
        f"Rows:              "
        f"{len(df)}"
    )

    log(
        f"Unique channels:   "
        f"{df['channel'].nunique()}"
    )

    log(
        f"Canonical CSVs:    "
        f"{len(result_files)}"
    )

    log(
        f"Checkpoints:       "
        f"{len(checkpoint_files)}"
    )

    log(
        f"Total fit hours:   "
        f"{df['elapsed_seconds'].sum()/3600:.3f}"
    )

    return df


def assert_matched_design(
    exp002,
    exp003,
):
    log()
    log("=" * 72)
    log("MATCHED DESIGN CHECK")
    log("=" * 72)

    merge = exp002.merge(
        exp003,
        on=[
            "channel",
            "mission",
            "seed",
        ],
        suffixes=(
            "_graph",
            "_no_graph",
        ),
        how="outer",
        indicator=True,
    )

    require(
        len(merge)
        == EXPECTED_RUNS,
        "Merged comparison does not "
        f"contain {EXPECTED_RUNS} rows",
    )

    require(
        (
            merge["_merge"]
            == "both"
        ).all(),
        "EXP002 and EXP003 "
        "channel/mission/seed keys "
        "do not match exactly",
    )

    exact_fields = [
        "original_num_features",
        "graph_num_nodes",
        "num_values",
        "fit_samples",
        "validation_samples",
        "calibration_samples",
        "hidden_size",
        "lookback",
        "parameters",
    ]

    for field in exact_fields:
        left = merge[
            f"{field}_graph"
        ]

        right = merge[
            f"{field}_no_graph"
        ]

        require(
            (
                left
                == right
            ).all(),
            f"Matched-design field "
            f"differs: {field}",
        )

    graph_feature_left = (
        "graph_feature_indices_graph"
    )

    graph_feature_right = (
        "graph_feature_indices_no_graph"
    )

    if (
        graph_feature_left
        in merge.columns
        and
        graph_feature_right
        in merge.columns
    ):
        require(
            (
                merge[
                    graph_feature_left
                ].astype(str)
                ==
                merge[
                    graph_feature_right
                ].astype(str)
            ).all(),
            "Retained feature indices "
            "differ between EXP002 "
            "and EXP003",
        )

    p_graph = merge[
        "persistence_validation_mse_graph"
    ].to_numpy(
        dtype=float
    )

    p_no_graph = merge[
        "persistence_validation_mse_no_graph"
    ].to_numpy(
        dtype=float
    )

    require(
        np.allclose(
            p_graph,
            p_no_graph,
            rtol=1e-10,
            atol=1e-12,
        ),
        "Persistence validation MSE "
        "differs between matched runs",
    )

    if (
        "reference_graph_mode"
        in exp003.columns
    ):
        expected_mode = merge[
            "graph_mode_graph"
        ].astype(str)

        observed_mode = merge[
            "reference_graph_mode"
        ].astype(str)

        require(
            (
                expected_mode
                == observed_mode
            ).all(),
            "EXP003 reference graph mode "
            "does not match EXP002",
        )

    if (
        "reference_graph_num_edges"
        in exp003.columns
    ):
        expected_edges = merge[
            "graph_num_edges_graph"
        ].astype(int)

        observed_edges = merge[
            "reference_graph_num_edges"
        ].astype(int)

        require(
            (
                expected_edges
                == observed_edges
            ).all(),
            "EXP003 reference edge count "
            "does not match EXP002",
        )

    require(
        (
            merge[
                "graph_num_edges_no_graph"
            ]
            == 0
        ).all(),
        "EXP003 contains a non-zero "
        "active graph edge count",
    )

    log(
        "243/243 channel-seed pairs matched."
    )

    log(
        "Data sizes, retained nodes, "
        "hidden sizes, lookback, parameter "
        "counts and persistence baseline "
        "match exactly."
    )

    log(
        "EXP003 active edge count = 0 "
        "for all runs."
    )

    return merge


def bootstrap_channel_ci(
    channel_frame,
):
    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    graph = channel_frame[
        "graph_mse"
    ].to_numpy(
        dtype=float
    )

    no_graph = channel_frame[
        "no_graph_mse"
    ].to_numpy(
        dtype=float
    )

    n = len(channel_frame)

    indices = rng.integers(
        0,
        n,
        size=(
            BOOTSTRAP_SAMPLES,
            n,
        ),
    )

    graph_boot = (
        graph[indices]
        .mean(axis=1)
    )

    no_graph_boot = (
        no_graph[indices]
        .mean(axis=1)
    )

    delta_boot = (
        no_graph_boot
        - graph_boot
    )

    valid = (
        no_graph_boot
        > RELATIVE_EPS
    )

    reduction_boot = np.full(
        BOOTSTRAP_SAMPLES,
        np.nan,
        dtype=float,
    )

    reduction_boot[
        valid
    ] = (
        100.0
        * delta_boot[valid]
        / no_graph_boot[valid]
    )

    delta_ci = np.quantile(
        delta_boot,
        [0.025, 0.975],
    )

    finite_reduction = (
        reduction_boot[
            np.isfinite(
                reduction_boot
            )
        ]
    )

    if len(
        finite_reduction
    ) > 0:
        reduction_ci = np.quantile(
            finite_reduction,
            [0.025, 0.975],
        )
    else:
        reduction_ci = [
            np.nan,
            np.nan,
        ]

    return (
        float(delta_ci[0]),
        float(delta_ci[1]),
        float(reduction_ci[0]),
        float(reduction_ci[1]),
    )


def compare_models(
    merged,
):
    log()
    log("=" * 72)
    log("EXP002 FIXED GRAPH vs EXP003 NO GRAPH")
    log("=" * 72)

    paired_runs = pd.DataFrame(
        {
            "channel": (
                merged["channel"]
            ),
            "mission": (
                merged["mission"]
            ),
            "seed": (
                merged["seed"]
                .astype(int)
            ),
            "graph_mse": (
                merged[
                    "validation_mse_graph"
                ].astype(float)
            ),
            "no_graph_mse": (
                merged[
                    "validation_mse_no_graph"
                ].astype(float)
            ),
        }
    )

    paired_runs[
        "delta_no_graph_minus_graph"
    ] = (
        paired_runs[
            "no_graph_mse"
        ]
        -
        paired_runs[
            "graph_mse"
        ]
    )

    paired_runs[
        "graph_reduction_pct"
    ] = np.where(
        paired_runs[
            "no_graph_mse"
        ]
        > RELATIVE_EPS,
        (
            100.0
            *
            paired_runs[
                "delta_no_graph_minus_graph"
            ]
            /
            paired_runs[
                "no_graph_mse"
            ]
        ),
        np.nan,
    )

    # ---------------------------------------------------------
    # Average each channel across the three seeds first.
    # Channels are the resampling unit for inference.
    # ---------------------------------------------------------

    channel_summary = (
        paired_runs
        .groupby(
            [
                "mission",
                "channel",
            ],
            as_index=False,
        )
        .agg(
            graph_mse=(
                "graph_mse",
                "mean",
            ),
            no_graph_mse=(
                "no_graph_mse",
                "mean",
            ),
        )
    )

    channel_summary[
        "delta_no_graph_minus_graph"
    ] = (
        channel_summary[
            "no_graph_mse"
        ]
        -
        channel_summary[
            "graph_mse"
        ]
    )

    channel_summary[
        "graph_reduction_pct"
    ] = np.where(
        channel_summary[
            "no_graph_mse"
        ]
        > RELATIVE_EPS,
        (
            100.0
            *
            channel_summary[
                "delta_no_graph_minus_graph"
            ]
            /
            channel_summary[
                "no_graph_mse"
            ]
        ),
        np.nan,
    )

    tolerance = (
        1e-10
        +
        1e-6
        *
        np.maximum(
            np.abs(
                channel_summary[
                    "graph_mse"
                ]
            ),
            np.abs(
                channel_summary[
                    "no_graph_mse"
                ]
            ),
        )
    )

    delta = channel_summary[
        "delta_no_graph_minus_graph"
    ]

    channel_summary[
        "outcome"
    ] = np.where(
        delta > tolerance,
        "graph_win",
        np.where(
            delta < -tolerance,
            "graph_loss",
            "tie",
        ),
    )

    mission_rows = []

    for mission, frame in (
        channel_summary.groupby(
            "mission"
        )
    ):
        graph_mean = float(
            frame[
                "graph_mse"
            ].mean()
        )

        no_graph_mean = float(
            frame[
                "no_graph_mse"
            ].mean()
        )

        mean_delta = (
            no_graph_mean
            - graph_mean
        )

        if (
            no_graph_mean
            > RELATIVE_EPS
        ):
            aggregate_reduction = (
                100.0
                * mean_delta
                / no_graph_mean
            )
        else:
            aggregate_reduction = (
                np.nan
            )

        defined_reductions = (
            frame[
                "graph_reduction_pct"
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

        median_paired_reduction = (
            float(
                defined_reductions
                .median()
            )
            if len(
                defined_reductions
            )
            else np.nan
        )

        wins = int(
            (
                frame["outcome"]
                == "graph_win"
            ).sum()
        )

        losses = int(
            (
                frame["outcome"]
                == "graph_loss"
            ).sum()
        )

        ties = int(
            (
                frame["outcome"]
                == "tie"
            ).sum()
        )

        (
            delta_low,
            delta_high,
            reduction_low,
            reduction_high,
        ) = bootstrap_channel_ci(
            frame
        )

        mission_rows.append(
            {
                "mission": mission,
                "channels": len(frame),

                "graph_validation_mse_mean": (
                    graph_mean
                ),

                "no_graph_validation_mse_mean": (
                    no_graph_mean
                ),

                "mean_delta_no_graph_minus_graph": (
                    mean_delta
                ),

                "aggregate_graph_reduction_pct": (
                    aggregate_reduction
                ),

                "median_defined_paired_reduction_pct": (
                    median_paired_reduction
                ),

                "defined_relative_channels": int(
                    defined_reductions
                    .shape[0]
                ),

                "undefined_relative_channels": int(
                    len(frame)
                    -
                    defined_reductions
                    .shape[0]
                ),

                "graph_wins": wins,
                "graph_losses": losses,
                "ties": ties,

                "bootstrap_mean_delta_ci95_low": (
                    delta_low
                ),

                "bootstrap_mean_delta_ci95_high": (
                    delta_high
                ),

                "bootstrap_reduction_pct_ci95_low": (
                    reduction_low
                ),

                "bootstrap_reduction_pct_ci95_high": (
                    reduction_high
                ),
            }
        )

    mission_summary = pd.DataFrame(
        mission_rows
    ).sort_values(
        "mission"
    )

    seed_rows = []

    for (
        mission,
        seed,
    ), frame in paired_runs.groupby(
        [
            "mission",
            "seed",
        ]
    ):
        graph_mean = float(
            frame[
                "graph_mse"
            ].mean()
        )

        no_graph_mean = float(
            frame[
                "no_graph_mse"
            ].mean()
        )

        delta = (
            no_graph_mean
            - graph_mean
        )

        reduction = (
            100.0
            * delta
            / no_graph_mean
            if no_graph_mean
            > RELATIVE_EPS
            else np.nan
        )

        tol = (
            1e-10
            +
            1e-6
            *
            np.maximum(
                np.abs(
                    frame["graph_mse"]
                ),
                np.abs(
                    frame["no_graph_mse"]
                ),
            )
        )

        pair_delta = (
            frame[
                "no_graph_mse"
            ]
            -
            frame[
                "graph_mse"
            ]
        )

        seed_rows.append(
            {
                "mission": mission,
                "seed": int(seed),
                "channels": (
                    frame[
                        "channel"
                    ].nunique()
                ),

                "graph_validation_mse_mean": (
                    graph_mean
                ),

                "no_graph_validation_mse_mean": (
                    no_graph_mean
                ),

                "graph_reduction_pct": (
                    reduction
                ),

                "graph_wins": int(
                    (
                        pair_delta
                        > tol
                    ).sum()
                ),

                "graph_losses": int(
                    (
                        pair_delta
                        < -tol
                    ).sum()
                ),

                "ties": int(
                    (
                        np.abs(
                            pair_delta
                        )
                        <= tol
                    ).sum()
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

    COMPARE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paired_runs.to_csv(
        COMPARE_DIR
        / "paired_run_results.csv",
        index=False,
    )

    channel_summary.to_csv(
        COMPARE_DIR
        / "paired_channel_summary.csv",
        index=False,
    )

    mission_summary.to_csv(
        COMPARE_DIR
        / "mission_comparison.csv",
        index=False,
    )

    seed_summary.to_csv(
        COMPARE_DIR
        / "seed_comparison.csv",
        index=False,
    )

    top_bottom = []

    for mission, frame in (
        channel_summary.groupby(
            "mission"
        )
    ):
        ordered = frame.sort_values(
            "delta_no_graph_minus_graph",
            ascending=False,
        )

        best = ordered.head(
            min(5, len(ordered))
        ).copy()

        best[
            "group"
        ] = "largest_graph_benefit"

        worst = ordered.tail(
            min(5, len(ordered))
        ).sort_values(
            "delta_no_graph_minus_graph"
        ).copy()

        worst[
            "group"
        ] = "largest_graph_harm"

        top_bottom.extend(
            [
                best,
                worst,
            ]
        )

    extremes = pd.concat(
        top_bottom,
        ignore_index=True,
    )

    extremes.to_csv(
        COMPARE_DIR
        / "channel_extremes.csv",
        index=False,
    )

    log()
    log("Mission-level paired comparison:")
    log(
        mission_summary.to_string(
            index=False
        )
    )

    log()
    log("Seed-level stability:")
    log(
        seed_summary.to_string(
            index=False
        )
    )

    return (
        mission_summary,
        seed_summary,
        channel_summary,
    )


def provenance():
    files = [
        ROOT
        / "configs"
        / "v2_exp002_tcn_gcn_fixed.json",

        ROOT
        / "configs"
        / "v2_exp003_tcn_no_graph.json",

        ROOT
        / "src"
        / "v2_data.py",

        ROOT
        / "src"
        / "v2_graph.py",

        ROOT
        / "src"
        / "v2_models_exp002.py",

        ROOT
        / "src"
        / "v2_models_exp003.py",

        ROOT
        / "src"
        / "v2_train_exp002.py",

        ROOT
        / "src"
        / "v2_train_exp003.py",

        ROOT
        / "src"
        / "v2_run_exp003_parallel.py",

        ROOT
        / "src"
        / "v2_verify_exp003.py",
    ]

    log()
    log("=" * 72)
    log("PROVENANCE SHA256")
    log("=" * 72)

    for path in files:
        if path.exists():
            log(
                f"{path.relative_to(ROOT)}"
            )

            log(
                f"  {sha256_file(path)}"
            )
        else:
            require(
                False,
                f"Missing provenance file: "
                f"{path}",
            )


def main():
    log("=" * 72)
    log("V2 EXP003 AUDIT + EXP002/EXP003 PAIRED COMPARISON")
    log("=" * 72)

    exp002 = audit_experiment(
        EXP002,
        RESULT002,
        CHECK002,
    )

    exp003 = audit_experiment(
        EXP003,
        RESULT003,
        CHECK003,
    )

    merged = assert_matched_design(
        exp002,
        exp003,
    )

    (
        mission_summary,
        seed_summary,
        channel_summary,
    ) = compare_models(
        merged
    )

    provenance()

    log()
    log("=" * 72)

    if failures:
        log(
            f"AUDIT/COMPARISON FAILED: "
            f"{len(failures)} issue(s)"
        )

        for index, failure in enumerate(
            failures,
            start=1,
        ):
            log(
                f"{index}. {failure}"
            )

        status = 1

    else:
        log("EXP003 AUDIT PASSED")

        log(
            "243/243 EXP003 canonical runs "
            "are internally consistent."
        )

        log(
            "EXP002 and EXP003 are matched "
            "for the intended graph ablation."
        )

        log(
            "Paired channel-level comparison "
            "completed successfully."
        )

        status = 0

    log("=" * 72)

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_path = (
        AUDIT_DIR
        /
        "v2_exp003_audit_and_exp002_vs_exp003_2026-10-01.txt"
    )

    audit_path.write_text(
        "\n".join(report)
        + "\n"
    )

    print()
    print(
        f"Audit report saved to: "
        f"{audit_path}"
    )

    print(
        f"Comparison tables saved to: "
        f"{COMPARE_DIR}"
    )

    raise SystemExit(
        status
    )


if __name__ == "__main__":
    main()
