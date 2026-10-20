from __future__ import annotations

import json
import sys
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from io_utils import load_protocol, project_root, source_root

SRC = source_root()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from v2_data import load_channel, infer_mission, DATA_ROOT, TRAIN_DIR  # noqa: E402
from v2_graph import build_channel_graph  # noqa: E402
from v2_models import TCNBaseline  # noqa: E402
from v2_models_exp002 import TCNGCNFixed  # noqa: E402
from v2_models_exp003 import TCNNoGraphMatched  # noqa: E402
from v2_models_exp004 import TCNGCNLearnedEdges  # noqa: E402
from v2_models_exp005 import CommandAwareTCN  # noqa: E402


MODEL_KEYS = ("exp001", "exp002", "exp003", "exp004", "exp005")
SEEDS = (17, 29, 43)


@dataclass
class LoadedModel:
    model_key: str
    channel: str
    mission: str
    seed: int
    model: torch.nn.Module
    checkpoint_path: Path
    checkpoint: dict
    feature_indices: np.ndarray
    lookback: int


@lru_cache(maxsize=128)
def mission_for_channel(channel: str) -> str:
    path = Path(TRAIN_DIR) / f"{channel}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = np.load(path, mmap_mode="r")
    return infer_mission(raw)


@lru_cache(maxsize=128)
def cached_graph(channel: str, fit_fraction: float = 0.60):
    return build_channel_graph(channel, fit_fraction=float(fit_fraction))


def checkpoint_path(model_key: str, channel: str, mission: str, seed: int) -> Path:
    root = project_root() / "checkpoints"
    protocol = load_protocol()
    if model_key == "exp005":
        candidate = protocol["models"]["exp005"]["selected_candidate_by_mission"][mission]
        return root / "v2_exp005_command_tcn_tuned" / "confirm" / candidate / f"{channel}_seed{seed}.pt"
    exp_id = protocol["models"][model_key]["experiment_id"]
    return root / exp_id / f"{channel}_seed{seed}.pt"


def _as_int_list(value):
    if isinstance(value, str):
        value = json.loads(value)
    return [int(x) for x in value]


def load_checkpoint_model(model_key: str, channel: str, seed: int) -> LoadedModel:
    # Mission inference uses only TRAIN array shape; no TEST/labels are accessed.
    mission = mission_for_channel(channel)
    path = checkpoint_path(model_key, channel, mission, seed)
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu")
    result = ckpt["result"]
    if str(result.get("channel")) != channel or int(result.get("seed")) != int(seed):
        raise RuntimeError(f"{path}: checkpoint metadata mismatch")
    if str(result.get("mission")) != mission:
        raise RuntimeError(f"{path}: mission metadata mismatch")

    if model_key == "exp001":
        cfg = ckpt["config"]
        mcfg = cfg["model"]
        lookback = int(mcfg["lookback"])
        model = TCNBaseline(
            num_features=int(result["num_features"]),
            hidden_size=int(result["hidden_size"]),
            kernel_size=int(mcfg["kernel_size"]),
            dilations=tuple(int(x) for x in mcfg["dilations"]),
        )
        feature_indices = np.arange(int(result["num_features"]), dtype=np.int64)

    elif model_key in {"exp002", "exp003", "exp004"}:
        cfg = ckpt["config"]
        mcfg = cfg["model"]
        lookback = int(mcfg["lookback"])
        graph = cached_graph(channel, float(cfg["dataset"]["fit_fraction"]))
        recorded = _as_int_list(result["graph_feature_indices"])
        if recorded != graph.feature_indices.tolist():
            raise RuntimeError(f"{path}: graph feature selection no longer reproduces checkpoint metadata")
        if model_key == "exp003":
            recorded_sources = _as_int_list(result.get("reference_graph_source_original_columns", "[]"))
        else:
            recorded_sources = _as_int_list(result["graph_source_original_columns"])
        if recorded_sources != graph.source_original_columns.tolist():
            raise RuntimeError(f"{path}: graph source columns no longer reproduce checkpoint metadata")
        kwargs = dict(
            num_nodes=graph.num_nodes,
            hidden_size=int(result["hidden_size"]),
            kernel_size=int(mcfg["kernel_size"]),
            dilations=tuple(int(x) for x in mcfg["dilations"]),
        )
        if model_key == "exp002":
            model = TCNGCNFixed(adjacency=graph.adjacency, **kwargs)
        elif model_key == "exp003":
            model = TCNNoGraphMatched(**kwargs)
        else:
            model = TCNGCNLearnedEdges(adjacency=graph.adjacency, **kwargs)
        feature_indices = graph.feature_indices.astype(np.int64)

    elif model_key == "exp005":
        candidate = ckpt["candidate"]
        lookback = int(candidate["lookback"])
        feature_indices = np.asarray(_as_int_list(result["selected_feature_columns"]), dtype=np.int64)
        graph = cached_graph(channel, float(ckpt["experiment_config"]["dataset"]["fit_fraction"]))
        expected = [0] + graph.source_original_columns.astype(int).tolist()
        if feature_indices.tolist() != expected:
            raise RuntimeError(f"{path}: EXP005 selected feature columns do not reproduce EXP002 incoming sources")
        protocol = load_protocol()
        selected = protocol["models"]["exp005"]["selected_candidate_by_mission"][mission]
        if candidate["candidate_id"] != selected:
            raise RuntimeError(f"{path}: runner-up EXP005 checkpoint supplied where selected checkpoint was required")
        model = CommandAwareTCN(
            num_features=len(feature_indices),
            hidden_size=int(result["hidden_size"]),
            kernel_size=int(candidate["kernel_size"]),
            dilations=tuple(int(x) for x in candidate["dilations"]),
        )
    else:
        raise KeyError(model_key)

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return LoadedModel(model_key, channel, mission, int(seed), model, path, ckpt, feature_indices, lookback)


def train_split_data(channel: str, lookback: int = 64):
    protocol = load_protocol()
    return load_channel(
        channel=channel,
        lookback=lookback,
        fit_fraction=float(protocol["development_split"]["fit_fraction"]),
        validation_end_fraction=float(protocol["development_split"]["validation_end_fraction"]),
    )


def select_windows(x: np.ndarray, feature_indices: np.ndarray) -> np.ndarray:
    return x[:, feature_indices, :].astype(np.float32, copy=False)


@torch.no_grad()
def predict_numpy(model: torch.nn.Module, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    preds = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start:start + batch_size].astype(np.float32, copy=False))
        yb = model(xb).detach().cpu().numpy().astype(np.float64)
        preds.append(yb)
    if not preds:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(preds)


def load_standardized_test(channel: str):
    # Intentionally label-blind. Uses TEST array only after freeze and fit-only scaler from TRAIN.
    train = train_split_data(channel, lookback=64)
    path = Path(DATA_ROOT) / "test" / f"{channel}.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = np.load(path).astype(np.float32)
    if raw.ndim != 2:
        raise ValueError(f"{channel}: expected 2D TEST array")
    mission = infer_mission(raw)
    if mission != train.mission:
        raise RuntimeError(f"{channel}: TRAIN/TEST mission mismatch")
    standardized = raw.copy()
    standardized[:, 0] = (standardized[:, 0] - train.target_mean) / train.target_scale
    return standardized, train


def make_test_windows(standardized: np.ndarray, lookback: int = 64):
    n = len(standardized)
    target_idx = np.arange(lookback, n, dtype=np.int64)
    all_windows = np.lib.stride_tricks.sliding_window_view(
        standardized, window_shape=lookback, axis=0
    )
    x = all_windows[target_idx - lookback].copy().astype(np.float32)
    y = standardized[target_idx, 0].copy().astype(np.float32)
    persistence = standardized[target_idx - 1, 0].copy().astype(np.float32)
    return target_idx, x, y, persistence
