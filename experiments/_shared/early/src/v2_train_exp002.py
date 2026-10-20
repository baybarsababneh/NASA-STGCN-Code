import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from v2_data import load_channel
from v2_graph import build_channel_graph
from v2_models_exp002 import TCNGCNFixed, count_parameters


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

RELATIVE_EPS = 1e-8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    x,
    y,
    batch_size,
    shuffle,
    seed,
):
    x_tensor = torch.from_numpy(
        x.astype(np.float32)
    )

    y_tensor = torch.from_numpy(
        y.astype(np.float32)
    )

    dataset = TensorDataset(
        x_tensor,
        y_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=(
            generator
            if shuffle
            else None
        ),
        num_workers=0,
    )


@torch.no_grad()
def evaluate_mse(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        prediction = model(
            x_batch
        )

        loss = criterion(
            prediction,
            y_batch,
        )

        batch_size = len(
            y_batch
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

    if total_samples == 0:
        raise RuntimeError(
            "Evaluation loader contains no samples."
        )

    return (
        total_loss
        / total_samples
    )


def subset_graph_features(
    x,
    feature_indices,
):
    """
    x shape:
        [samples, original_features, lookback]

    Return only the telemetry target and command-context
    features retained by the fit-only graph construction.
    """

    feature_indices = np.asarray(
        feature_indices,
        dtype=np.int64,
    )

    if feature_indices.ndim != 1:
        raise ValueError(
            "feature_indices must be one-dimensional."
        )

    if len(feature_indices) == 0:
        raise ValueError(
            "At least telemetry node 0 must be retained."
        )

    if feature_indices[0] != 0:
        raise ValueError(
            "Graph feature index 0 must be telemetry."
        )

    if feature_indices.max() >= x.shape[1]:
        raise ValueError(
            "Graph feature index exceeds available "
            "input features."
        )

    return x[
        :,
        feature_indices,
        :,
    ]


def train_one(
    channel,
    seed,
    config,
    smoke_epochs=None,
):
    set_seed(seed)

    device = torch.device(
        "cpu"
    )

    model_cfg = config["model"]
    train_cfg = config["training"]

    lookback = int(
        model_cfg["lookback"]
    )

    # ---------------------------------------------------------
    # Load exactly the same chronological TRAIN-only split
    # convention used in EXP001.
    # ---------------------------------------------------------

    data = load_channel(
        channel=channel,
        lookback=lookback,
        fit_fraction=(
            config["dataset"]["fit_fraction"]
        ),
        validation_end_fraction=(
            config["dataset"][
                "validation_end_fraction"
            ]
        ),
    )

    # ---------------------------------------------------------
    # Build graph using FIT data only.
    # ---------------------------------------------------------

    graph = build_channel_graph(
        channel=channel,
        fit_fraction=(
            config["dataset"]["fit_fraction"]
        ),
    )

    if graph.mission != data.mission:
        raise RuntimeError(
            f"{channel}: mission mismatch between "
            "data loader and graph builder."
        )

    if graph.num_nodes != len(
        graph.feature_indices
    ):
        raise RuntimeError(
            f"{channel}: graph node count mismatch."
        )

    # ---------------------------------------------------------
    # Select exactly the graph nodes from every split.
    #
    # The feature-selection decision itself was made using
    # FIT data only.
    # ---------------------------------------------------------

    x_fit = subset_graph_features(
        data.x_fit,
        graph.feature_indices,
    )

    x_val = subset_graph_features(
        data.x_val,
        graph.feature_indices,
    )

    x_cal = subset_graph_features(
        data.x_cal,
        graph.feature_indices,
    )

    if x_fit.shape[1] != graph.num_nodes:
        raise RuntimeError(
            f"{channel}: training node count mismatch."
        )

    if x_val.shape[1] != graph.num_nodes:
        raise RuntimeError(
            f"{channel}: validation node count mismatch."
        )

    if x_cal.shape[1] != graph.num_nodes:
        raise RuntimeError(
            f"{channel}: calibration node count mismatch."
        )

    hidden_size = int(
        model_cfg[
            "hidden_by_mission"
        ][data.mission]
    )

    model = TCNGCNFixed(
        num_nodes=graph.num_nodes,
        hidden_size=hidden_size,
        adjacency=graph.adjacency,
        kernel_size=int(
            model_cfg["kernel_size"]
        ),
        dilations=tuple(
            model_cfg["dilations"]
        ),
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            train_cfg[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            train_cfg[
                "weight_decay"
            ]
        ),
    )

    batch_size = int(
        train_cfg["batch_size"]
    )

    fit_loader = make_loader(
        x=x_fit,
        y=data.y_fit,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    val_loader = make_loader(
        x=x_val,
        y=data.y_val,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    max_epochs = int(
        train_cfg["max_epochs"]
    )

    if smoke_epochs is not None:
        max_epochs = min(
            max_epochs,
            int(smoke_epochs),
        )

    patience = int(
        train_cfg["patience"]
    )

    min_delta = float(
        train_cfg["min_delta"]
    )

    clip_norm = float(
        train_cfg[
            "gradient_clip_norm"
        ]
    )

    best_val_mse = float(
        "inf"
    )

    best_epoch = 0
    best_state = None

    epochs_without_improvement = 0
    epochs_ran = 0

    start_time = time.perf_counter()

    # ---------------------------------------------------------
    # Training.
    # ---------------------------------------------------------

    for epoch in range(
        1,
        max_epochs + 1,
    ):
        epochs_ran = epoch

        model.train()

        for x_batch, y_batch in fit_loader:
            x_batch = x_batch.to(
                device
            )

            y_batch = y_batch.to(
                device
            )

            optimizer.zero_grad()

            prediction = model(
                x_batch
            )

            loss = criterion(
                prediction,
                y_batch,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=clip_norm,
            )

            optimizer.step()

        val_mse = evaluate_mse(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if val_mse < (
            best_val_mse
            - min_delta
        ):
            best_val_mse = (
                val_mse
            )

            best_epoch = (
                epoch
            )

            best_state = deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        print(
            f"{channel} | "
            f"{data.mission} | "
            f"graph={graph.graph_mode} | "
            f"nodes={graph.num_nodes} | "
            f"edges={graph.num_edges} | "
            f"seed={seed} | "
            f"epoch={epoch:02d} | "
            f"val_mse={val_mse:.8f}"
        )

        if (
            smoke_epochs is None
            and
            epochs_without_improvement
            >= patience
        ):
            break

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    if best_state is None:
        raise RuntimeError(
            "Training completed without a "
            "valid checkpoint."
        )

    model.load_state_dict(
        best_state
    )

    final_val_mse = evaluate_mse(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    # ---------------------------------------------------------
    # Persistence baseline.
    #
    # This remains identical to EXP001 because node 0 is the
    # same telemetry target.
    # ---------------------------------------------------------

    persistence_val_mse = float(
        np.mean(
            (
                data.persistence_val
                - data.y_val
            )
            ** 2
        )
    )

    if (
        persistence_val_mse
        <= RELATIVE_EPS
    ):
        relative_reduction = (
            np.nan
        )

    else:
        relative_reduction = (
            100.0
            * (
                persistence_val_mse
                - final_val_mse
            )
            / persistence_val_mse
        )

    result = {
        "experiment_id": (
            config[
                "experiment_id"
            ]
        ),

        "channel": channel,
        "mission": data.mission,
        "seed": seed,

        "original_num_features": (
            data.num_features
        ),

        "graph_num_nodes": (
            graph.num_nodes
        ),

        "graph_num_edges": (
            graph.num_edges
        ),

        "graph_mode": (
            graph.graph_mode
        ),

        "graph_feature_indices": (
            json.dumps(
                graph
                .feature_indices
                .tolist()
            )
        ),

        "graph_source_original_columns": (
            json.dumps(
                graph
                .source_original_columns
                .tolist()
            )
        ),

        "num_values": (
            data.num_values
        ),

        "fit_samples": (
            len(data.y_fit)
        ),

        "validation_samples": (
            len(data.y_val)
        ),

        "calibration_samples": (
            len(data.y_cal)
        ),

        "hidden_size": (
            hidden_size
        ),

        "lookback": (
            lookback
        ),

        "parameters": (
            count_parameters(
                model
            )
        ),

        "best_epoch": (
            best_epoch
        ),

        "epochs_ran": (
            epochs_ran
        ),

        "validation_mse": (
            final_val_mse
        ),

        "persistence_validation_mse": (
            persistence_val_mse
        ),

        "relative_mse_reduction_pct": (
            relative_reduction
        ),

        "elapsed_seconds": (
            elapsed_seconds
        ),
    }

    return (
        model,
        result,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--channel",
        required=True,
        help=(
            "Channel ID, for example "
            "E-1 or C-1."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=17,
    )

    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    with open(
        CONFIG_PATH,
        "r",
    ) as handle:
        config = json.load(
            handle
        )

    allowed_seeds = set(
        config[
            "training"
        ][
            "seeds"
        ]
    )

    if (
        args.seed
        not in allowed_seeds
    ):
        raise ValueError(
            f"Seed {args.seed} "
            "is not in the locked "
            "experiment config."
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model, result = train_one(
        channel=args.channel,
        seed=args.seed,
        config=config,
        smoke_epochs=(
            args.smoke_epochs
        ),
    )

    suffix = (
        "_smoke"
        if args.smoke_epochs
        is not None
        else ""
    )

    stem = (
        f"{args.channel}"
        f"_seed{args.seed}"
        f"{suffix}"
    )

    result_path = (
        RESULTS_DIR
        / f"{stem}.csv"
    )

    pd.DataFrame(
        [result]
    ).to_csv(
        result_path,
        index=False,
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{stem}.pt"
    )

    torch.save(
        {
            "state_dict": (
                model.state_dict()
            ),
            "result": result,
            "config": config,
        },
        checkpoint_path,
    )

    print(
        "\n"
        + "=" * 68
    )

    print(
        "V2 EXP002 TRAINING COMPLETE"
    )

    print(
        "=" * 68
    )

    for key, value in (
        result.items()
    ):
        print(
            f"{key}: {value}"
        )

    print(
        f"\nResult:     "
        f"{result_path}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )


if __name__ == "__main__":
    main()