"""Functions extracted unchanged from completed EXP006 GPU trainer."""
import random
import time
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from v2_data import TRAIN_DIR, infer_mission

def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
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
):
    device = torch.device("cuda")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(common["weight_decay"]),
        foreach=False,
        fused=False,
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
    history = []
    torch.cuda.synchronize()
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
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_norm
            )
            optimizer.step()

        stop_mse = evaluate_mse(
            model, stop_loader, criterion, device
        )
        if not np.isfinite(stop_mse):
            raise RuntimeError("Non-finite early-stopping MSE")
        history.append({"epoch": epoch, "early_stop_mse": stop_mse})
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch={epoch} stop_mse={stop_mse:.9g}", flush=True)
        if stop_mse < best_mse - min_delta:
            best_mse = stop_mse
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if smoke_epochs is None and stale >= patience:
            break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint")
    model.load_state_dict(best_state)
    final_stop_mse = evaluate_mse(
        model, stop_loader, criterion, device
    )
    model.exp006_history = history
    return model, final_stop_mse, best_epoch, epochs_ran, elapsed

def _verify_initial_persistence(model, x_probe):
    device = torch.device("cuda")
    model = model.to(device)
    with torch.no_grad():
        probe = torch.from_numpy(x_probe.astype(np.float32)).to(device)
        pred = model(probe)
        persistence = probe[:, 0, -1]
        if not torch.allclose(pred, persistence, atol=1e-7, rtol=0.0):
            raise RuntimeError("Initial output is not exact persistence")

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
        "target_mean": mean,
        "target_scale": scale,
    }
