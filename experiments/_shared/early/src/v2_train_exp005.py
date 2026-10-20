import json
import random
import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from v2_data import load_channel
from v2_graph import build_channel_graph
from v2_models_exp005 import CommandAwareTCN, count_parameters


RELATIVE_EPS = 1e-8


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def make_loader(x, y, batch_size, shuffle, seed):
    x_tensor = torch.from_numpy(x.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32))

    dataset = TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
        num_workers=0,
    )


@torch.no_grad()
def evaluate_mse(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        prediction = model(x_batch)
        loss = criterion(prediction, y_batch)
        n = len(y_batch)
        total_loss += loss.item() * n
        total_samples += n

    if total_samples == 0:
        raise RuntimeError("Evaluation loader contains no samples")

    return total_loss / total_samples


def selected_command_columns(channel, fit_fraction):
    """
    Return target column 0 plus the exact original command columns that send
    messages into telemetry node 0 in EXP002.

    This deliberately uses EXP002's fit-only graph-support selection, but the
    EXP005 model does not use the graph or adjacency itself.
    """
    graph = build_channel_graph(
        channel=channel,
        fit_fraction=float(fit_fraction),
    )

    source_cols = [int(x) for x in graph.source_original_columns.tolist()]
    feature_cols = [0] + source_cols

    if len(set(feature_cols)) != len(feature_cols):
        raise RuntimeError(f"{channel}: duplicate selected feature columns")

    return graph, np.asarray(feature_cols, dtype=np.int64)


def subset_features(x, feature_columns):
    feature_columns = np.asarray(feature_columns, dtype=np.int64)
    if feature_columns.ndim != 1 or len(feature_columns) < 1:
        raise ValueError("feature_columns must be a non-empty 1D array")
    if feature_columns[0] != 0:
        raise ValueError("First selected feature must be telemetry column 0")
    if int(feature_columns.max()) >= x.shape[1]:
        raise ValueError("Selected feature index exceeds input width")
    return x[:, feature_columns, :]


def train_one(channel, seed, base_config, candidate, smoke_epochs=None):
    set_seed(seed)
    device = torch.device("cpu")

    dataset_cfg = base_config["dataset"]
    common = base_config["training_common"]

    lookback = int(candidate["lookback"])
    kernel_size = int(candidate["kernel_size"])
    dilations = tuple(int(d) for d in candidate["dilations"])

    data = load_channel(
        channel=channel,
        lookback=lookback,
        fit_fraction=float(dataset_cfg["fit_fraction"]),
        validation_end_fraction=float(dataset_cfg["validation_end_fraction"]),
    )

    graph, feature_columns = selected_command_columns(
        channel=channel,
        fit_fraction=float(dataset_cfg["fit_fraction"]),
    )

    if graph.mission != data.mission:
        raise RuntimeError(f"{channel}: mission mismatch")

    x_fit = subset_features(data.x_fit, feature_columns)
    x_val = subset_features(data.x_val, feature_columns)
    x_cal = subset_features(data.x_cal, feature_columns)

    hidden_size = int(candidate["hidden_by_mission"][data.mission])

    model = CommandAwareTCN(
        num_features=x_fit.shape[1],
        hidden_size=hidden_size,
        kernel_size=kernel_size,
        dilations=dilations,
    ).to(device)

    if model.receptive_field > lookback:
        raise ValueError(
            f"{candidate['candidate_id']}: receptive field "
            f"{model.receptive_field} exceeds lookback {lookback}"
        )

    # Verify the untouched model is exactly persistence.
    with torch.no_grad():
        probe = torch.from_numpy(x_fit[: min(8, len(x_fit))]).to(device)
        initial = model(probe)
        persistence = probe[:, 0, -1]
        if not torch.allclose(initial, persistence, atol=1e-7, rtol=0.0):
            raise RuntimeError("Initial EXP005 output is not persistence")

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["learning_rate"]),
        weight_decay=float(candidate.get(
            "weight_decay", common["weight_decay"]
        )),
    )

    batch_size = int(common["batch_size"])
    fit_loader = make_loader(
        x_fit, data.y_fit, batch_size, True, seed
    )
    val_loader = make_loader(
        x_val, data.y_val, batch_size, False, seed
    )

    max_epochs = int(common["max_epochs"])
    if smoke_epochs is not None:
        max_epochs = min(max_epochs, int(smoke_epochs))

    patience = int(common["patience"])
    min_delta = float(common["min_delta"])
    clip_norm = float(common["gradient_clip_norm"])

    best_val_mse = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    epochs_ran = 0

    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epochs_ran = epoch
        model.train()

        for x_batch, y_batch in fit_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_norm
            )
            optimizer.step()

        val_mse = evaluate_mse(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if val_mse < best_val_mse - min_delta:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            smoke_epochs is None
            and epochs_without_improvement >= patience
        ):
            break

    elapsed_seconds = time.perf_counter() - start_time

    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint")

    model.load_state_dict(best_state)
    final_val_mse = evaluate_mse(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
    )

    persistence_val_mse = float(np.mean(
        (data.persistence_val - data.y_val) ** 2
    ))

    if persistence_val_mse <= RELATIVE_EPS:
        relative_reduction = np.nan
    else:
        relative_reduction = (
            100.0
            * (persistence_val_mse - final_val_mse)
            / persistence_val_mse
        )

    result = {
        "experiment_id": base_config["experiment_id"],
        "candidate_id": candidate["candidate_id"],
        "channel": channel,
        "mission": data.mission,
        "seed": int(seed),
        "input_contract": "telemetry_plus_exp002_incoming_command_sources",
        "graph_support_mode_reference": graph.graph_mode,
        "selected_feature_columns": json.dumps(feature_columns.tolist()),
        "selected_command_columns": json.dumps(feature_columns[1:].tolist()),
        "num_selected_features": int(len(feature_columns)),
        "num_selected_commands": int(len(feature_columns) - 1),
        "original_num_features": int(data.num_features),
        "num_values": int(data.num_values),
        "fit_samples": int(len(data.y_fit)),
        "validation_samples": int(len(data.y_val)),
        "calibration_samples": int(len(data.y_cal)),
        "lookback": lookback,
        "kernel_size": kernel_size,
        "dilations": json.dumps(list(dilations)),
        "receptive_field": int(model.receptive_field),
        "hidden_size": hidden_size,
        "learning_rate": float(candidate["learning_rate"]),
        "weight_decay": float(candidate.get(
            "weight_decay", common["weight_decay"]
        )),
        "parameters": int(count_parameters(model)),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epochs_ran),
        "validation_mse": float(final_val_mse),
        "persistence_validation_mse": float(persistence_val_mse),
        "relative_mse_reduction_pct": float(relative_reduction),
        "elapsed_seconds": float(elapsed_seconds),
    }

    return model, result
