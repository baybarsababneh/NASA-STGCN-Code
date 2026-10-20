from pathlib import Path
import json
import random

import numpy as np
import torch

from v2_graph import build_channel_graph
from v2_models_exp002 import TCNGCNFixed
from v2_models_exp003 import TCNNoGraphMatched


ROOT = Path(__file__).resolve().parents[1]

CONFIG_002 = (
    ROOT
    / "configs"
    / "v2_exp002_tcn_gcn_fixed.json"
)

CONFIG_003 = (
    ROOT
    / "configs"
    / "v2_exp003_tcn_no_graph.json"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def verify_configs():
    with open(CONFIG_002, "r") as handle:
        c2 = json.load(handle)

    with open(CONFIG_003, "r") as handle:
        c3 = json.load(handle)

    dataset_keys = [
        "missions",
        "unique_labelled_channels",
        "exclude_unlabelled",
        "duplicate_label_policy",
        "fit_fraction",
        "validation_end_fraction",
        "calibration_fraction",
        "test_access_during_development",
    ]

    for key in dataset_keys:
        if c2["dataset"][key] != c3["dataset"][key]:
            raise RuntimeError(
                f"Dataset mismatch: {key}"
            )

    model_keys = [
        "lookback",
        "kernel_size",
        "dilations",
        "hidden_by_mission",
        "residual_prediction",
    ]

    for key in model_keys:
        if c2["model"][key] != c3["model"][key]:
            raise RuntimeError(
                f"Model mismatch: {key}"
            )

    training_keys = [
        "seeds",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "patience",
        "min_delta",
        "gradient_clip_norm",
    ]

    for key in training_keys:
        if c2["training"][key] != c3["training"][key]:
            raise RuntimeError(
                f"Training mismatch: {key}"
            )

    print("Config match: PASSED")


def verify_channel(
    channel,
    hidden_size,
    seed=17,
):
    graph = build_channel_graph(
        channel
    )

    set_seed(seed)

    exp002 = TCNGCNFixed(
        num_nodes=graph.num_nodes,
        hidden_size=hidden_size,
        adjacency=graph.adjacency,
        kernel_size=3,
        dilations=(1, 2),
    )

    set_seed(seed)

    exp003 = TCNNoGraphMatched(
        num_nodes=graph.num_nodes,
        hidden_size=hidden_size,
        kernel_size=3,
        dilations=(1, 2),
    )

    state2 = exp002.state_dict()
    state3 = exp003.state_dict()

    for name, _ in exp002.named_parameters():
        if name not in state3:
            raise RuntimeError(
                f"{channel}: missing parameter {name}"
            )

        if not torch.equal(
            state2[name],
            state3[name],
        ):
            raise RuntimeError(
                f"{channel}: initialization differs "
                f"for {name}"
            )

    if (
        torch.count_nonzero(
            exp003.adjacency
        ).item()
        != 0
    ):
        raise RuntimeError(
            f"{channel}: EXP003 adjacency not zero."
        )

    if (
        exp002.adjacency.shape
        != exp003.adjacency.shape
    ):
        raise RuntimeError(
            f"{channel}: adjacency shape changed."
        )

    n2 = sum(
        p.numel()
        for p in exp002.parameters()
        if p.requires_grad
    )

    n3 = sum(
        p.numel()
        for p in exp003.parameters()
        if p.requires_grad
    )

    if n2 != n3:
        raise RuntimeError(
            f"{channel}: parameter count changed: "
            f"{n2} vs {n3}"
        )

    print(
        f"{channel}: PASSED | "
        f"mission={graph.mission} | "
        f"nodes={graph.num_nodes} | "
        f"EXP002_edges={graph.num_edges} | "
        f"EXP003_edges=0 | "
        f"parameters={n2:,}"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("V2 EXP003 MATCHED-ABLATION VERIFICATION")
    print("=" * 70)

    verify_configs()

    verify_channel(
        channel="E-1",
        hidden_size=64,
    )

    verify_channel(
        channel="C-1",
        hidden_size=32,
    )

    print()
    print(
        "VERIFICATION PASSED: parameter initialization, "
        "parameter count, retained input-node support, "
        "data split and training configuration are matched. "
        "The intended difference is zero adjacency in EXP003."
    )
