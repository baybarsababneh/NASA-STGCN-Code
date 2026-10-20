import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from v2_data import list_channels
from v2_run_exp003_all import build_summaries


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "results"
    / "v2_exp003_tcn_no_graph"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "v2_exp003_tcn_no_graph"
)

LOG_DIR = (
    RESULTS_DIR
    / "parallel_logs"
)

SEEDS = [17, 29, 43]


def result_path(channel, seed):
    return (
        RESULTS_DIR
        / f"{channel}_seed{seed}.csv"
    )


def checkpoint_path(channel, seed):
    return (
        CHECKPOINT_DIR
        / f"{channel}_seed{seed}.pt"
    )


def run_complete(channel, seed):
    return (
        result_path(channel, seed).exists()
        and
        checkpoint_path(channel, seed).exists()
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help=(
            "Maximum number of new canonical runs "
            "to launch during this invocation."
        ),
    )

    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError(
            "--workers must be at least 1."
        )

    channels = list_channels()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_tasks = [
        (channel, seed)
        for channel in channels
        for seed in SEEDS
    ]

    completed_before = [
        task
        for task in all_tasks
        if run_complete(*task)
    ]

    pending = [
        task
        for task in all_tasks
        if not run_complete(*task)
    ]

    if args.max_runs is not None:
        pending = pending[
            :args.max_runs
        ]

    print("=" * 72)
    print("V2 EXP003 PARALLEL CPU RUNNER")
    print("=" * 72)

    print(
        f"Channels:          {len(channels)}"
    )

    print(
        f"Total expected:    {len(all_tasks)}"
    )

    print(
        f"Already complete:  {len(completed_before)}"
    )

    print(
        f"Runs this launch:  {len(pending)}"
    )

    print(
        f"Parallel workers:  {args.workers}"
    )

    if not pending:
        print(
            "\nNo new runs requested."
        )

        build_summaries(
            channels,
            SEEDS,
        )

        return

    queue = list(pending)

    active = {}
    finished_rows = []
    failures = []

    wall_start = time.perf_counter()

    try:
        while queue or active:

            # -------------------------------------------------
            # Fill available worker slots.
            # -------------------------------------------------

            while (
                queue
                and
                len(active) < args.workers
            ):
                channel, seed = queue.pop(0)

                log_path = (
                    LOG_DIR
                    / f"{channel}_seed{seed}.log"
                )

                log_handle = open(
                    log_path,
                    "w",
                )

                command = [
                    sys.executable,
                    "-u",
                    str(
                        ROOT
                        / "src"
                        / "v2_train_exp003.py"
                    ),
                    "--channel",
                    channel,
                    "--seed",
                    str(seed),
                ]

                print(
                    f"\n[START] "
                    f"{channel} seed={seed}"
                )

                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )

                active[
                    process.pid
                ] = {
                    "process": process,
                    "channel": channel,
                    "seed": seed,
                    "log_path": log_path,
                    "log_handle": log_handle,
                }

            # -------------------------------------------------
            # Check running workers.
            # -------------------------------------------------

            finished_pids = []

            for pid, info in active.items():
                process = info["process"]

                returncode = process.poll()

                if returncode is None:
                    continue

                info[
                    "log_handle"
                ].close()

                channel = info["channel"]
                seed = info["seed"]

                if (
                    returncode == 0
                    and
                    run_complete(
                        channel,
                        seed,
                    )
                ):
                    frame = pd.read_csv(
                        result_path(
                            channel,
                            seed,
                        )
                    )

                    if len(frame) != 1:
                        raise RuntimeError(
                            f"{channel} seed={seed}: "
                            "result CSV does not contain "
                            "exactly one row."
                        )

                    row = (
                        frame
                        .iloc[0]
                        .to_dict()
                    )

                    finished_rows.append(
                        row
                    )

                    print(
                        f"[DONE]  "
                        f"{channel} seed={seed} | "
                        f"epochs={int(row['epochs_ran'])} | "
                        f"best={int(row['best_epoch'])} | "
                        f"val_mse="
                        f"{float(row['validation_mse']):.8f} | "
                        f"train_time="
                        f"{float(row['elapsed_seconds']):.1f}s"
                    )

                else:
                    failures.append(
                        {
                            "channel": channel,
                            "seed": seed,
                            "returncode": (
                                returncode
                            ),
                            "log": str(
                                info["log_path"]
                            ),
                        }
                    )

                    print(
                        f"[FAILED] "
                        f"{channel} seed={seed} | "
                        f"returncode={returncode} | "
                        f"log={info['log_path']}"
                    )

                finished_pids.append(
                    pid
                )

            for pid in finished_pids:
                del active[pid]

            if active:
                time.sleep(0.25)

    except KeyboardInterrupt:
        print(
            "\nStopping active workers..."
        )

        for info in active.values():
            process = info["process"]

            if process.poll() is None:
                process.terminate()

        for info in active.values():
            process = info["process"]

            try:
                process.wait(
                    timeout=10
                )
            except subprocess.TimeoutExpired:
                process.kill()

            info[
                "log_handle"
            ].close()

        print(
            "Stopped. Completed canonical runs remain saved."
        )

        raise

    wall_seconds = (
        time.perf_counter()
        - wall_start
    )

    # ---------------------------------------------------------
    # Rebuild experiment summaries from every completed
    # canonical run.
    # ---------------------------------------------------------

    build_summaries(
        channels,
        SEEDS,
    )

    print()
    print("=" * 72)
    print("PARALLEL RUN REPORT")
    print("=" * 72)

    print(
        f"New runs completed: "
        f"{len(finished_rows)}"
    )

    print(
        f"Failures:           "
        f"{len(failures)}"
    )

    print(
        f"Wall time:          "
        f"{wall_seconds:.2f} s"
    )

    if finished_rows:
        summed_training_time = sum(
            float(
                row[
                    "elapsed_seconds"
                ]
            )
            for row in finished_rows
        )

        effective_speedup = (
            summed_training_time
            / wall_seconds
        )

        print(
            f"Summed fit time:    "
            f"{summed_training_time:.2f} s"
        )

        print(
            f"Parallel efficiency:"
            f" {effective_speedup:.2f}x "
            f"vs running those fits serially"
        )

    if failures:
        failure_path = (
            RESULTS_DIR
            / "parallel_failures.csv"
        )

        pd.DataFrame(
            failures
        ).to_csv(
            failure_path,
            index=False,
        )

        print(
            f"Failure details:    "
            f"{failure_path}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
