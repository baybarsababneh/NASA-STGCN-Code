from dataclasses import dataclass

import numpy as np

from v2_data import TRAIN_DIR, infer_mission


VARYING_EPS = 1e-8

TOP4_LAGS = (
    1,
    2,
    4,
    8,
    16,
)


@dataclass
class GraphSpec:
    channel: str
    mission: str

    # Original columns retained in the graph input.
    # Column 0 is always telemetry.
    feature_indices: np.ndarray

    # Adjacency after feature selection.
    adjacency: np.ndarray

    # Node indices in the reduced graph that send
    # messages into telemetry node 0.
    source_nodes: np.ndarray

    # Corresponding original data columns.
    source_original_columns: np.ndarray

    graph_mode: str
    num_nodes: int
    num_edges: int


def select_varying_features(
    raw,
    fit_end,
):
    """
    Fit-only command-feature selection.

    Telemetry column 0 is always retained.

    Command columns are retained only when they vary during
    the fitting block. Validation/calibration/test data are
    never involved in this decision.
    """

    fit = raw[:fit_end]

    command_std = fit[:, 1:].std(
        axis=0
    )

    varying_command_columns = (
        np.flatnonzero(
            command_std > VARYING_EPS
        )
        + 1
    )

    feature_indices = np.concatenate(
        [
            np.array(
                [0],
                dtype=np.int64,
            ),
            varying_command_columns.astype(
                np.int64
            ),
        ]
    )

    return feature_indices


def build_fixed_adjacency(
    fit,
    mode,
):
    """
    Reproduce the fixed graph logic used in the earlier study.

    fit:
        [time, nodes]

        Node 0 = telemetry target
        Nodes 1... = command-context nodes

    Adjacency convention:
        adjacency[target, source]

    Therefore:
        adjacency[0, j] > 0

    means command node j sends information to telemetry node 0.
    """

    n = fit.shape[1]

    adjacency = np.zeros(
        (n, n),
        dtype=np.float32,
    )

    if n < 2:
        return (
            adjacency,
            np.array(
                [],
                dtype=np.int64,
            ),
        )

    chosen = np.arange(
        1,
        n,
        dtype=np.int64,
    )

    if mode == "star":
        pass

    elif mode == "top4":
        # -----------------------------------------------------
        # Fit-only lagged association with telemetry changes.
        # -----------------------------------------------------

        dy = np.diff(
            fit[:, 0]
        )

        scores = []

        for node in chosen:
            lag_scores = []

            for lag in TOP4_LAGS:
                command = fit[
                    :-lag,
                    node,
                ]

                telemetry_change = dy[
                    lag - 1:
                ]

                if (
                    command.std() > VARYING_EPS
                    and telemetry_change.std() > VARYING_EPS
                ):
                    corr = abs(
                        np.corrcoef(
                            command,
                            telemetry_change,
                        )[0, 1]
                    )

                    if not np.isfinite(corr):
                        corr = 0.0

                else:
                    corr = 0.0

                lag_scores.append(
                    corr
                )

            scores.append(
                max(lag_scores)
            )

        scores = np.asarray(
            scores,
            dtype=np.float64,
        )

        order = np.argsort(
            scores,
            kind="stable",
        )

        number_to_keep = min(
            4,
            n - 1,
        )

        chosen = chosen[
            order[-number_to_keep:]
        ]

    else:
        raise ValueError(
            f"Unsupported graph mode: {mode}"
        )

    if len(chosen) > 0:
        adjacency[
            0,
            chosen,
        ] = (
            1.0
            / len(chosen)
        )

    return adjacency, chosen


def build_channel_graph(
    channel,
    fit_fraction=0.60,
):
    """
    Construct one channel's fixed graph using official TRAIN
    data only.

    Steps:
        1. Load TRAIN array.
        2. Define first 60% fitting block.
        3. Keep telemetry + commands varying in fit.
        4. SMAP -> star.
        5. MSL  -> top4.
    """

    path = (
        TRAIN_DIR
        / f"{channel}.npy"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    raw = np.load(
        path
    ).astype(
        np.float32
    )

    if raw.ndim != 2:
        raise ValueError(
            f"{channel}: expected 2D array, "
            f"got {raw.shape}."
        )

    mission = infer_mission(
        raw
    )

    fit_end = int(
        np.floor(
            len(raw)
            * fit_fraction
        )
    )

    feature_indices = (
        select_varying_features(
            raw,
            fit_end,
        )
    )

    reduced_fit = raw[
        :fit_end,
        feature_indices,
    ]

    if mission == "SMAP":
        mode = "star"

    elif mission == "MSL":
        mode = "top4"

    else:
        raise ValueError(
            f"Unsupported mission: {mission}"
        )

    adjacency, source_nodes = (
        build_fixed_adjacency(
            reduced_fit,
            mode,
        )
    )

    source_original_columns = (
        feature_indices[
            source_nodes
        ]
    )

    return GraphSpec(
        channel=channel,
        mission=mission,
        feature_indices=feature_indices,
        adjacency=adjacency,
        source_nodes=source_nodes,
        source_original_columns=(
            source_original_columns
        ),
        graph_mode=mode,
        num_nodes=len(
            feature_indices
        ),
        num_edges=len(
            source_nodes
        ),
    )


def audit_all_graphs():
    """
    TRAIN-only graph-construction audit across all 81 V2
    study channels.
    """

    from v2_data import list_channels

    channels = list_channels()

    records = []

    for channel in channels:
        graph = build_channel_graph(
            channel
        )

        # -----------------------------------------------------
        # Structural checks.
        # -----------------------------------------------------

        if graph.adjacency.shape != (
            graph.num_nodes,
            graph.num_nodes,
        ):
            raise RuntimeError(
                f"{channel}: adjacency shape mismatch."
            )

        if graph.num_nodes < 1:
            raise RuntimeError(
                f"{channel}: no telemetry node."
            )

        if graph.num_edges > 0:
            row_sum = float(
                graph.adjacency[
                    0
                ].sum()
            )

            if not np.isclose(
                row_sum,
                1.0,
                atol=1e-6,
            ):
                raise RuntimeError(
                    f"{channel}: incoming graph weights "
                    f"sum to {row_sum}, not 1."
                )

        if graph.mission == "MSL":
            if graph.num_edges > 4:
                raise RuntimeError(
                    f"{channel}: MSL graph has more "
                    f"than four command edges."
                )

        if graph.mission == "SMAP":
            expected_edges = max(
                0,
                graph.num_nodes - 1,
            )

            if (
                graph.num_edges
                != expected_edges
            ):
                raise RuntimeError(
                    f"{channel}: SMAP star edge "
                    f"count mismatch."
                )

        records.append(
            {
                "channel": channel,
                "mission": graph.mission,
                "mode": graph.graph_mode,
                "nodes": graph.num_nodes,
                "edges": graph.num_edges,
                "source_columns": (
                    graph
                    .source_original_columns
                    .tolist()
                ),
            }
        )

    return records


if __name__ == "__main__":
    records = audit_all_graphs()

    smap = [
        r
        for r in records
        if r["mission"] == "SMAP"
    ]

    msl = [
        r
        for r in records
        if r["mission"] == "MSL"
    ]

    print("=" * 60)
    print("V2 FIXED GRAPH AUDIT")
    print("=" * 60)

    print(
        f"Channels: {len(records)}"
    )

    print(
        f"SMAP: {len(smap)}"
    )

    print(
        f"MSL:  {len(msl)}"
    )

    print()

    print(
        "SMAP node range: "
        f"{min(r['nodes'] for r in smap)}"
        " to "
        f"{max(r['nodes'] for r in smap)}"
    )

    print(
        "SMAP edge range: "
        f"{min(r['edges'] for r in smap)}"
        " to "
        f"{max(r['edges'] for r in smap)}"
    )

    print(
        "MSL node range:  "
        f"{min(r['nodes'] for r in msl)}"
        " to "
        f"{max(r['nodes'] for r in msl)}"
    )

    print(
        "MSL edge range:  "
        f"{min(r['edges'] for r in msl)}"
        " to "
        f"{max(r['edges'] for r in msl)}"
    )

    print()

    print("Example MSL graphs:")

    for record in msl[:5]:
        print(
            f"{record['channel']}: "
            f"nodes={record['nodes']}, "
            f"edges={record['edges']}, "
            f"source_columns="
            f"{record['source_columns']}"
        )

    print()

    print(
        "V2 fixed graph audit PASSED"
    )
    