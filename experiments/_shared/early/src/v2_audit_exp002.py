from pathlib import Path
import hashlib
import re
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EXP_ID = "v2_exp002_tcn_gcn_fixed"

RESULT_DIR = ROOT / "results" / EXP_ID
CHECKPOINT_DIR = ROOT / "checkpoints" / EXP_ID
AUDIT_DIR = ROOT / "audits"

ALL_RUNS_PATH = RESULT_DIR / "all_runs.csv"
MISSION_SUMMARY_PATH = RESULT_DIR / "mission_summary.csv"
MISSION_SEED_SUMMARY_PATH = RESULT_DIR / "mission_seed_summary.csv"

EXPECTED_SEEDS = {17, 29, 43}
EXPECTED_TOTAL_CHANNELS = 81
EXPECTED_TOTAL_RUNS = 243
EXPECTED_MISSION_CHANNELS = {
    "MSL": 27,
    "SMAP": 54,
}
EXPECTED_MISSION_RUNS = {
    "MSL": 81,
    "SMAP": 162,
}

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
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_key(path, extension):
    pattern = rf"(.+)_seed(\d+)\.{extension}"
    match = re.fullmatch(pattern, path.name)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


log("=" * 72)
log("V2 EXP002 INTEGRITY AUDIT")
log("=" * 72)
log(f"Project root: {ROOT}")
log(f"Experiment:   {EXP_ID}")
log()

# ---------------------------------------------------------------------
# 1. Required files
# ---------------------------------------------------------------------

for path in [
    ALL_RUNS_PATH,
    MISSION_SUMMARY_PATH,
    MISSION_SEED_SUMMARY_PATH,
]:
    require(path.exists(), f"Missing required file: {path}")

require(RESULT_DIR.exists(), f"Missing result directory: {RESULT_DIR}")
require(
    CHECKPOINT_DIR.exists(),
    f"Missing checkpoint directory: {CHECKPOINT_DIR}",
)

if failures:
    log()
    log("Cannot continue because required files/directories are missing.")
    sys.exit(1)

# ---------------------------------------------------------------------
# 2. all_runs.csv structure
# ---------------------------------------------------------------------

df = pd.read_csv(ALL_RUNS_PATH)

required_columns = {
    "channel",
    "mission",
    "seed",
    "validation_mse",
    "persistence_validation_mse",
    "best_epoch",
    "epochs_ran",
    "elapsed_seconds",
}

missing_columns = required_columns - set(df.columns)
require(
    not missing_columns,
    f"all_runs.csv missing columns: {sorted(missing_columns)}",
)

log(f"Rows in all_runs.csv: {len(df)}")
require(
    len(df) == EXPECTED_TOTAL_RUNS,
    f"Expected {EXPECTED_TOTAL_RUNS} rows, found {len(df)}",
)

df["channel"] = df["channel"].astype(str)
df["mission"] = df["mission"].astype(str)
df["seed"] = df["seed"].astype(int)

# ---------------------------------------------------------------------
# 3. Channel / seed integrity
# ---------------------------------------------------------------------

duplicate_mask = df.duplicated(["channel", "seed"], keep=False)
require(
    not duplicate_mask.any(),
    "Duplicate channel/seed combinations found in all_runs.csv",
)

channels = sorted(df["channel"].unique())
log(f"Unique channels: {len(channels)}")

require(
    len(channels) == EXPECTED_TOTAL_CHANNELS,
    f"Expected {EXPECTED_TOTAL_CHANNELS} unique channels, "
    f"found {len(channels)}",
)

seed_set = set(df["seed"].unique())
log(f"Seeds: {sorted(seed_set)}")

require(
    seed_set == EXPECTED_SEEDS,
    f"Expected seeds {sorted(EXPECTED_SEEDS)}, found {sorted(seed_set)}",
)

bad_seed_channels = []

for channel, group in df.groupby("channel"):
    channel_seeds = set(group["seed"].astype(int))
    if channel_seeds != EXPECTED_SEEDS or len(group) != 3:
        bad_seed_channels.append(
            (channel, sorted(channel_seeds), len(group))
        )

require(
    not bad_seed_channels,
    f"Channels without exactly seeds 17/29/43: {bad_seed_channels}",
)

# ---------------------------------------------------------------------
# 4. Mission integrity
# ---------------------------------------------------------------------

mission_per_channel = df.groupby("channel")["mission"].nunique()

require(
    (mission_per_channel == 1).all(),
    "At least one channel is assigned to multiple missions",
)

channel_missions = (
    df[["channel", "mission"]]
    .drop_duplicates()
    .groupby("mission")["channel"]
    .nunique()
)

run_missions = df.groupby("mission").size()

log()
log("Mission counts:")

for mission in ["MSL", "SMAP"]:
    channel_count = int(channel_missions.get(mission, 0))
    run_count = int(run_missions.get(mission, 0))

    log(
        f"  {mission}: "
        f"{channel_count} channels, "
        f"{run_count} runs"
    )

    require(
        channel_count == EXPECTED_MISSION_CHANNELS[mission],
        f"{mission}: expected "
        f"{EXPECTED_MISSION_CHANNELS[mission]} channels, "
        f"found {channel_count}",
    )

    require(
        run_count == EXPECTED_MISSION_RUNS[mission],
        f"{mission}: expected "
        f"{EXPECTED_MISSION_RUNS[mission]} runs, "
        f"found {run_count}",
    )

# ---------------------------------------------------------------------
# 5. Numeric sanity checks
# ---------------------------------------------------------------------

critical_numeric = [
    "validation_mse",
    "persistence_validation_mse",
    "best_epoch",
    "epochs_ran",
    "elapsed_seconds",
]

for column in critical_numeric:
    values = pd.to_numeric(df[column], errors="coerce")

    require(
        np.isfinite(values).all(),
        f"Non-finite value found in {column}",
    )

require(
    (df["validation_mse"] >= 0).all(),
    "Negative validation MSE found",
)

require(
    (df["persistence_validation_mse"] >= 0).all(),
    "Negative persistence MSE found",
)

require(
    (df["elapsed_seconds"] > 0).all(),
    "Non-positive elapsed time found",
)

require(
    (df["best_epoch"] >= 1).all(),
    "best_epoch below 1 found",
)

require(
    (df["epochs_ran"] >= df["best_epoch"]).all(),
    "best_epoch exceeds epochs_ran for at least one run",
)

require(
    (df["epochs_ran"] <= 80).all(),
    "epochs_ran exceeds configured maximum of 80",
)

# ---------------------------------------------------------------------
# 6. Canonical result CSV and checkpoint coverage
# ---------------------------------------------------------------------

result_files = []
for path in RESULT_DIR.glob("*.csv"):
    key = canonical_key(path, "csv")
    if key is not None:
        result_files.append((path, key))

checkpoint_files = []
for path in CHECKPOINT_DIR.glob("*.pt"):
    key = canonical_key(path, "pt")
    if key is not None:
        checkpoint_files.append((path, key))

result_keys = {key for _, key in result_files}
checkpoint_keys = {key for _, key in checkpoint_files}

all_run_keys = {
    (str(row.channel), int(row.seed))
    for row in df.itertuples(index=False)
}

log()
log(f"Canonical per-run CSVs: {len(result_files)}")
log(f"Canonical checkpoints:  {len(checkpoint_files)}")

require(
    len(result_files) == EXPECTED_TOTAL_RUNS,
    f"Expected {EXPECTED_TOTAL_RUNS} canonical result CSVs, "
    f"found {len(result_files)}",
)

require(
    len(checkpoint_files) == EXPECTED_TOTAL_RUNS,
    f"Expected {EXPECTED_TOTAL_RUNS} canonical checkpoints, "
    f"found {len(checkpoint_files)}",
)

require(
    result_keys == all_run_keys,
    "Canonical result-file channel/seed set does not match all_runs.csv",
)

require(
    checkpoint_keys == all_run_keys,
    "Checkpoint channel/seed set does not match all_runs.csv",
)

# ---------------------------------------------------------------------
# 7. Verify every individual result CSV
# ---------------------------------------------------------------------

all_runs_lookup = {
    (str(row.channel), int(row.seed)): row
    for row in df.itertuples(index=False)
}

bad_result_files = []

for path, key in result_files:
    try:
        one = pd.read_csv(path)
    except Exception as exc:
        bad_result_files.append((path.name, f"read failure: {exc}"))
        continue

    if len(one) != 1:
        bad_result_files.append(
            (path.name, f"expected 1 row, found {len(one)}")
        )
        continue

    row = one.iloc[0]

    try:
        file_channel = str(row["channel"])
        file_seed = int(row["seed"])
        file_mission = str(row["mission"])
        file_mse = float(row["validation_mse"])
    except Exception as exc:
        bad_result_files.append(
            (path.name, f"missing/invalid fields: {exc}")
        )
        continue

    if (file_channel, file_seed) != key:
        bad_result_files.append(
            (
                path.name,
                f"filename key {key} != row key "
                f"{(file_channel, file_seed)}",
            )
        )
        continue

    master = all_runs_lookup[key]

    if file_mission != str(master.mission):
        bad_result_files.append(
            (path.name, "mission differs from all_runs.csv")
        )
        continue

    if not np.isclose(
        file_mse,
        float(master.validation_mse),
        rtol=1e-10,
        atol=1e-12,
    ):
        bad_result_files.append(
            (path.name, "validation_mse differs from all_runs.csv")
        )

require(
    not bad_result_files,
    f"Problems found in individual result CSVs: "
    f"{bad_result_files[:10]}",
)

# ---------------------------------------------------------------------
# 8. Recompute mission summary independently
# ---------------------------------------------------------------------

summary_rows = []

for mission, group in df.groupby("mission"):
    val_mean = float(group["validation_mse"].mean())
    val_median = float(group["validation_mse"].median())
    persistence_mean = float(
        group["persistence_validation_mse"].mean()
    )

    if persistence_mean > 0:
        macro_reduction = (
            100.0 * (1.0 - val_mean / persistence_mean)
        )
    else:
        macro_reduction = np.nan

    summary_rows.append(
        {
            "mission": mission,
            "channels": int(group["channel"].nunique()),
            "runs": int(len(group)),
            "validation_mse_mean": val_mean,
            "validation_mse_median": val_median,
            "persistence_mse_mean": persistence_mean,
            "macro_mse_reduction_pct": macro_reduction,
            "best_epoch_mean": float(group["best_epoch"].mean()),
            "epochs_ran_mean": float(group["epochs_ran"].mean()),
            "elapsed_seconds_total": float(
                group["elapsed_seconds"].sum()
            ),
        }
    )

recomputed = pd.DataFrame(summary_rows).sort_values("mission")

saved_summary = pd.read_csv(MISSION_SUMMARY_PATH).sort_values("mission")

summary_columns = [
    "channels",
    "runs",
    "validation_mse_mean",
    "validation_mse_median",
    "persistence_mse_mean",
    "macro_mse_reduction_pct",
    "best_epoch_mean",
    "epochs_ran_mean",
    "elapsed_seconds_total",
]

require(
    set(recomputed["mission"]) == set(saved_summary["mission"]),
    "Saved mission_summary.csv mission set is incorrect",
)

for mission in recomputed["mission"]:
    a = recomputed[recomputed["mission"] == mission].iloc[0]
    b = saved_summary[saved_summary["mission"] == mission].iloc[0]

    for column in summary_columns:
        av = float(a[column])
        bv = float(b[column])

        require(
            np.isclose(av, bv, rtol=1e-8, atol=1e-10),
            f"{mission} mission_summary mismatch in {column}: "
            f"recomputed={av}, saved={bv}",
        )

# ---------------------------------------------------------------------
# 9. Mission/seed summary structural check
# ---------------------------------------------------------------------

seed_summary = pd.read_csv(MISSION_SEED_SUMMARY_PATH)

require(
    len(seed_summary) == 6,
    f"Expected 6 mission/seed summary rows, found {len(seed_summary)}",
)

if {"mission", "seed"}.issubset(seed_summary.columns):
    observed_pairs = {
        (str(row.mission), int(row.seed))
        for row in seed_summary.itertuples(index=False)
    }

    expected_pairs = {
        (mission, seed)
        for mission in ["MSL", "SMAP"]
        for seed in EXPECTED_SEEDS
    }

    require(
        observed_pairs == expected_pairs,
        "mission_seed_summary.csv does not contain exactly "
        "MSL/SMAP x seeds 17/29/43",
    )

# ---------------------------------------------------------------------
# 10. Provenance hashes for EXP002 code/config
# ---------------------------------------------------------------------

provenance_files = [
    ROOT / "configs" / "v2_exp002_tcn_gcn_fixed.json",
    ROOT / "src" / "v2_data.py",
    ROOT / "src" / "v2_graph.py",
    ROOT / "src" / "v2_models_exp002.py",
    ROOT / "src" / "v2_train_exp002.py",
    ROOT / "src" / "v2_run_exp002_all.py",
]

log()
log("EXP002 provenance SHA256:")

for path in provenance_files:
    if path.exists():
        digest = sha256_file(path)
        log(f"  {path.relative_to(ROOT)}")
        log(f"    {digest}")
    else:
        require(False, f"Missing provenance file: {path}")

# ---------------------------------------------------------------------
# 11. Final scientific summary
# ---------------------------------------------------------------------

log()
log("=" * 72)
log("RECOMPUTED EXP002 SUMMARY")
log("=" * 72)

for _, row in recomputed.iterrows():
    log(
        f"{row['mission']}: "
        f"channels={int(row['channels'])}, "
        f"runs={int(row['runs'])}, "
        f"validation_mse_mean={row['validation_mse_mean']:.6f}, "
        f"validation_mse_median={row['validation_mse_median']:.6f}, "
        f"persistence_mse_mean={row['persistence_mse_mean']:.6f}, "
        f"macro_mse_reduction_pct="
        f"{row['macro_mse_reduction_pct']:.6f}, "
        f"best_epoch_mean={row['best_epoch_mean']:.6f}, "
        f"epochs_ran_mean={row['epochs_ran_mean']:.6f}, "
        f"elapsed_seconds_total="
        f"{row['elapsed_seconds_total']:.6f}"
    )

total_seconds = float(df["elapsed_seconds"].sum())

log()
log(f"Total training seconds across all runs: {total_seconds:.3f}")
log(f"Total training hours across all runs:   {total_seconds / 3600:.3f}")

log()
log("=" * 72)

if failures:
    log(f"EXP002 AUDIT FAILED: {len(failures)} issue(s)")
    log("=" * 72)

    for i, failure in enumerate(failures, 1):
        log(f"{i}. {failure}")

    status = 1
else:
    log("EXP002 AUDIT PASSED")
    log("243/243 canonical runs are internally consistent.")
    log("EXP002 is safe to treat as frozen.")
    log("=" * 72)
    status = 0
AUDIT_DIR.mkdir(exist_ok=True)
audit_path = AUDIT_DIR / "v2_exp002_audit_2026-10-01.txt"
audit_path.write_text("\n".join(report) + "\n")

print()
print(f"Audit report saved to: {audit_path}")

sys.exit(status)
