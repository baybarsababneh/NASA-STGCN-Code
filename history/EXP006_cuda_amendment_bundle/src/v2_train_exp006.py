import json
import random
import time
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from v2_data import TRAIN_DIR, infer_mission, load_channel
from v2_graph import build_channel_graph
from v2_models_exp002 import TCNGCNFixed as HistoricalEXP002GCN
from v2_models_exp005 import CommandAwareTCN
from v2_models_exp006 import TCNGCNFixedTuned, count_parameters
from v2_exp006_device import configure_device, optimizer_execution_kwargs


RELATIVE_EPS = 1e-8


def set_seed(seed, device_name="cpu"):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if str(device_name).lower() == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def make_loader(x, y, batch_size, shuffle, seed):
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
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
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        n = len(y_batch)
        total_loss += loss.item() * n
        total_samples += n
    if total_samples == 0:
        raise RuntimeError("Evaluation loader contains no samples")
    return total_loss / total_samples


def subset_features(x, feature_indices):
    idx = np.asarray(feature_indices, dtype=np.int64)
    if idx.ndim != 1 or len(idx) < 1:
        raise ValueError("feature_indices must be a non-empty 1D array")
    if idx[0] != 0:
        raise ValueError("First retained feature must be telemetry column 0")
    if int(idx.max()) >= x.shape[1]:
        raise ValueError("Selected feature index exceeds input width")
    return x[:, idx, :]


def _train_with_early_stopping(
    model,
    x_fit,
    y_fit,
    x_stop,
    y_stop,
    seed,
    learning_rate,
    common,
    smoke_epochs=None,
    device_name="cpu",
):
    # Models are deliberately initialised on CPU by the caller, then moved
    # here before optimizer construction. This preserves the frozen CPU
    # initialisation procedure while permitting CUDA execution.
    device = configure_device(device_name)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(common["weight_decay"]),
        **optimizer_execution_kwargs(),
    )

    fit_loader = make_loader(
        x_fit, y_fit, common["batch_size"], True, seed
    )
    stop_loader = make_loader(
        x_stop, y_stop, common["batch_size"], False, seed
    )

    max_epochs = int(common["max_epochs"])
    if smoke_epochs is not None:
        max_epochs = min(max_epochs, int(smoke_epochs))
    patience = int(common["patience"])
    min_delta = float(common["min_delta"])
    clip_norm = float(common["gradient_clip_norm"])

    best_mse = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    epochs_ran = 0
    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epochs_ran = epoch
        model.train()
        for x_batch, y_batch in fit_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_norm
            )
            optimizer.step()

        stop_mse = evaluate_mse(
            model, stop_loader, criterion, device
        )
        if stop_mse < best_mse - min_delta:
            best_mse = stop_mse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if smoke_epochs is None and stale >= patience:
            break

    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint")
    model.load_state_dict(best_state)
    final_stop_mse = evaluate_mse(
        model, stop_loader, criterion, device
    )
    return model, final_stop_mse, best_epoch, epochs_ran, elapsed


def _verify_initial_persistence(model, x_probe):
    device = torch.device("cpu")
    model = model.to(device)
    with torch.no_grad():
        probe = torch.from_numpy(x_probe.astype(np.float32)).to(device)
        pred = model(probe)
        persistence = probe[:, 0, -1]
        if not torch.allclose(pred, persistence, atol=1e-7, rtol=0.0):
            raise RuntimeError("Initial output is not exact persistence")


def train_standard(channel, seed, base_config, candidate, smoke_epochs=None, device_name="cpu", execution_environment_id=None):
    """60/20/20 TRAIN-only screening or confirmation fit."""
    set_seed(seed, device_name=device_name)
    dataset_cfg = base_config["dataset"]
    common = base_config["training_common"]

    lookback = int(candidate["lookback"])
    dilations = tuple(int(d) for d in candidate["dilations"])
    kernel_size = int(candidate["kernel_size"])

    data = load_channel(
        channel=channel,
        lookback=lookback,
        fit_fraction=float(dataset_cfg["fit_fraction"]),
        validation_end_fraction=float(dataset_cfg["validation_end_fraction"]),
    )
    graph = build_channel_graph(
        channel=channel,
        fit_fraction=float(dataset_cfg["fit_fraction"]),
    )
    if graph.mission != data.mission:
        raise RuntimeError(f"{channel}: mission mismatch")

    x_fit = subset_features(data.x_fit, graph.feature_indices)
    x_val = subset_features(data.x_val, graph.feature_indices)
    hidden = int(candidate["hidden_by_mission"][data.mission])

    model = TCNGCNFixedTuned(
        num_nodes=graph.num_nodes,
        hidden_size=hidden,
        adjacency=graph.adjacency,
        kernel_size=kernel_size,
        dilations=dilations,
    )
    if model.receptive_field > lookback:
        raise ValueError("Receptive field exceeds lookback")
    _verify_initial_persistence(model, x_fit[: min(8, len(x_fit))])

    model, val_mse, best_epoch, epochs_ran, elapsed = (
        _train_with_early_stopping(
            model=model,
            x_fit=x_fit,
            y_fit=data.y_fit,
            x_stop=x_val,
            y_stop=data.y_val,
            seed=seed,
            learning_rate=float(candidate["learning_rate"]),
            common=common,
            smoke_epochs=smoke_epochs,
            device_name=device_name,
        )
    )

    persistence_mse = float(
        np.mean((data.persistence_val - data.y_val) ** 2)
    )
    reduction = (
        np.nan
        if persistence_mse <= RELATIVE_EPS
        else 100.0 * (persistence_mse - val_mse) / persistence_mse
    )

    result = {
        "experiment_id": base_config["experiment_id"],
        "candidate_id": candidate["candidate_id"],
        "channel": channel,
        "mission": data.mission,
        "seed": int(seed),
        "graph_mode": graph.graph_mode,
        "graph_num_nodes": int(graph.num_nodes),
        "graph_num_edges": int(graph.num_edges),
        "graph_feature_indices": json.dumps(graph.feature_indices.tolist()),
        "graph_source_original_columns": json.dumps(
            graph.source_original_columns.tolist()
        ),
        "num_values": int(data.num_values),
        "fit_samples": int(len(data.y_fit)),
        "validation_samples": int(len(data.y_val)),
        "calibration_samples": int(len(data.y_cal)),
        "lookback": lookback,
        "kernel_size": kernel_size,
        "dilations": json.dumps(list(dilations)),
        "receptive_field": int(model.receptive_field),
        "hidden_size": hidden,
        "learning_rate": float(candidate["learning_rate"]),
        "weight_decay": float(common["weight_decay"]),
        "parameters": int(count_parameters(model)),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epochs_ran),
        "hit_epoch_ceiling": bool(epochs_ran >= int(common["max_epochs"])),
        "best_at_ceiling": bool(best_epoch >= int(common["max_epochs"])),
        "validation_mse": float(val_mse),
        "persistence_validation_mse": float(persistence_mse),
        "relative_mse_reduction_pct": float(reduction),
        "elapsed_seconds": float(elapsed),
        "device": str(device_name),
        "dtype": "float32",
        "optimizer_foreach": False,
        "optimizer_fused": False,
        "execution_environment_id": execution_environment_id,
    }
    return model, result


def _make_windows(standardized, target_times, lookback):
    windows = np.lib.stride_tricks.sliding_window_view(
        standardized, window_shape=int(lookback), axis=0
    )
    indices = target_times - int(lookback)
    x = windows[indices].copy().astype(np.float32)
    y = standardized[target_times, 0].copy().astype(np.float32)
    persistence = standardized[target_times - 1, 0].copy().astype(np.float32)
    return x, y, persistence


def _load_robust_fold(channel, lookback, fold):
    raw = np.load(TRAIN_DIR / f"{channel}.npy").astype(np.float32)
    mission = infer_mission(raw)
    n = len(raw)
    fit_end = int(np.floor(n * float(fold["fit_end_fraction"])))
    stop_end = int(np.floor(n * float(fold["early_stop_end_fraction"])))
    assess_end = int(np.floor(n * float(fold["assessment_end_fraction"])))
    if not (lookback < fit_end < stop_end < assess_end <= n):
        raise ValueError(
            f"{channel}: invalid robustness boundaries "
            f"{lookback}, {fit_end}, {stop_end}, {assess_end}, {n}"
        )

    fit_target = raw[:fit_end, 0]
    mean = float(fit_target.mean())
    scale = float(fit_target.std())
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    standardized = raw.copy()
    standardized[:, 0] = (standardized[:, 0] - mean) / scale

    fit_times = np.arange(lookback, fit_end, dtype=np.int64)
    stop_times = np.arange(fit_end, stop_end, dtype=np.int64)
    assess_times = np.arange(stop_end, assess_end, dtype=np.int64)
    x_fit, y_fit, _ = _make_windows(standardized, fit_times, lookback)
    x_stop, y_stop, _ = _make_windows(standardized, stop_times, lookback)
    x_assess, y_assess, p_assess = _make_windows(
        standardized, assess_times, lookback
    )
    return {
        "mission": mission,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "x_stop": x_stop,
        "y_stop": y_stop,
        "x_assess": x_assess,
        "y_assess": y_assess,
        "p_assess": p_assess,
        "fit_end": fit_end,
        "stop_end": stop_end,
        "assess_end": assess_end,
        "num_values": n,
    }


def train_robustness(
    channel,
    seed,
    base_config,
    role,
    fold_name,
    candidate=None,
    smoke_epochs=None,
    device_name="cpu",
    execution_environment_id=None,
):
    """
    Earlier-origin TRAIN-only robustness fit.

    role is one of: gcn_candidate, exp002_reference, exp005_reference.
    The graph/command support is recomputed from the fold fit block.
    """
    set_seed(seed, device_name=device_name)
    common = base_config["training_common"]
    fold = base_config["robustness"]["folds"][fold_name]

    if role == "gcn_candidate":
        if candidate is None:
            raise ValueError("candidate is required for gcn_candidate")
        spec = candidate
    elif role == "exp002_reference":
        spec = base_config["references"]["exp002"]
    elif role == "exp005_reference":
        # Mission-specific direct-TCN spec is resolved after reading the fold.
        spec = None
    else:
        raise ValueError(f"Unsupported robustness role: {role}")

    lookback = 64 if spec is None else int(spec["lookback"])
    arrays = _load_robust_fold(channel, lookback, fold)
    mission = arrays["mission"]

    # Fit-only graph support for this chronological origin.
    graph = build_channel_graph(
        channel=channel,
        fit_fraction=float(fold["fit_end_fraction"]),
    )
    if graph.mission != mission:
        raise RuntimeError(f"{channel}: graph/data mission mismatch")

    if role == "exp005_reference":
        spec = base_config["references"]["exp005_selected_by_mission"][mission]
        lookback = int(spec["lookback"])
        if lookback != 64:
            raise RuntimeError("EXP006 robustness assumes lookback 64")
        feature_cols = np.asarray(
            [0] + [int(x) for x in graph.source_original_columns.tolist()],
            dtype=np.int64,
        )
        x_fit = subset_features(arrays["x_fit"], feature_cols)
        x_stop = subset_features(arrays["x_stop"], feature_cols)
        x_assess = subset_features(arrays["x_assess"], feature_cols)
        hidden = int(spec["hidden_by_mission"][mission])
        model = CommandAwareTCN(
            num_features=x_fit.shape[1],
            hidden_size=hidden,
            kernel_size=int(spec["kernel_size"]),
            dilations=tuple(int(d) for d in spec["dilations"]),
        )
        receptive_field = int(model.receptive_field)
    else:
        x_fit = subset_features(arrays["x_fit"], graph.feature_indices)
        x_stop = subset_features(arrays["x_stop"], graph.feature_indices)
        x_assess = subset_features(arrays["x_assess"], graph.feature_indices)
        hidden = int(spec["hidden_by_mission"][mission])
        dilations = tuple(int(d) for d in spec["dilations"])
        if role == "exp002_reference":
            model = HistoricalEXP002GCN(
                num_nodes=graph.num_nodes,
                hidden_size=hidden,
                adjacency=graph.adjacency,
                kernel_size=int(spec["kernel_size"]),
                dilations=dilations,
            )
            receptive_field = 1 + (int(spec["kernel_size"]) - 1) * sum(dilations)
        else:
            model = TCNGCNFixedTuned(
                num_nodes=graph.num_nodes,
                hidden_size=hidden,
                adjacency=graph.adjacency,
                kernel_size=int(spec["kernel_size"]),
                dilations=dilations,
            )
            receptive_field = int(model.receptive_field)

    if receptive_field > lookback:
        raise ValueError("Receptive field exceeds lookback")
    _verify_initial_persistence(model, x_fit[: min(8, len(x_fit))])

    model, stop_mse, best_epoch, epochs_ran, elapsed = (
        _train_with_early_stopping(
            model=model,
            x_fit=x_fit,
            y_fit=arrays["y_fit"],
            x_stop=x_stop,
            y_stop=arrays["y_stop"],
            seed=seed,
            learning_rate=float(spec["learning_rate"]),
            common=common,
            smoke_epochs=smoke_epochs,
            device_name=device_name,
        )
    )

    criterion = nn.MSELoss()
    assess_loader = make_loader(
        x_assess, arrays["y_assess"], common["batch_size"], False, seed
    )
    assessment_mse = evaluate_mse(
        model, assess_loader, criterion, configure_device(device_name)
    )
    persistence_mse = float(
        np.mean((arrays["p_assess"] - arrays["y_assess"]) ** 2)
    )

    candidate_id = (
        candidate["candidate_id"] if role == "gcn_candidate" else role
    )
    return {
        "experiment_id": base_config["experiment_id"],
        "role": role,
        "candidate_id": candidate_id,
        "fold": fold_name,
        "channel": channel,
        "mission": mission,
        "seed": int(seed),
        "fit_end_fraction": float(fold["fit_end_fraction"]),
        "early_stop_end_fraction": float(fold["early_stop_end_fraction"]),
        "assessment_end_fraction": float(fold["assessment_end_fraction"]),
        "graph_mode": graph.graph_mode,
        "graph_num_nodes": int(graph.num_nodes),
        "graph_num_edges": int(graph.num_edges),
        "graph_source_original_columns": json.dumps(
            graph.source_original_columns.tolist()
        ),
        "lookback": int(lookback),
        "receptive_field": int(receptive_field),
        "hidden_size": int(hidden),
        "learning_rate": float(spec["learning_rate"]),
        "parameters": int(count_parameters(model)),
        "best_epoch": int(best_epoch),
        "epochs_ran": int(epochs_ran),
        "hit_epoch_ceiling": bool(epochs_ran >= int(common["max_epochs"])),
        "early_stop_mse": float(stop_mse),
        "assessment_mse": float(assessment_mse),
        "persistence_assessment_mse": float(persistence_mse),
        "elapsed_seconds": float(elapsed),
        "device": str(device_name),
        "dtype": "float32",
        "optimizer_foreach": False,
        "optimizer_fused": False,
        "execution_environment_id": execution_environment_id,
    }
