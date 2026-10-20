import random

import numpy as np
import torch

from v2_graph import build_channel_graph
from v2_models_exp002 import TCNGCNFixed
from v2_models_exp004 import TCNGCNLearnedEdges


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def verify_channel(
    channel,
    hidden_size,
    seed=17,
):
    graph = build_channel_graph(
        channel
    )

    set_seed(seed)

    fixed = TCNGCNFixed(
        num_nodes=graph.num_nodes,
        hidden_size=hidden_size,
        adjacency=graph.adjacency,
        kernel_size=3,
        dilations=(1, 2),
    )

    set_seed(seed)

    learned = TCNGCNLearnedEdges(
        num_nodes=graph.num_nodes,
        hidden_size=hidden_size,
        adjacency=graph.adjacency,
        kernel_size=3,
        dilations=(1, 2),
    )

    fixed_state = (
        fixed.state_dict()
    )

    learned_state = (
        learned.state_dict()
    )

    for name, _ in fixed.named_parameters():
        if name not in learned_state:
            raise RuntimeError(
                f"{channel}: missing shared "
                f"parameter {name}"
            )

        if not torch.equal(
            fixed_state[name],
            learned_state[name],
        ):
            raise RuntimeError(
                f"{channel}: initialization "
                f"differs for {name}"
            )

    weights = (
        learned.current_edge_weights()
    )

    if graph.num_edges > 0:
        expected = torch.full(
            (graph.num_edges,),
            1.0 / graph.num_edges,
            dtype=torch.float32,
        )

        if not torch.allclose(
            weights.cpu(),
            expected,
            atol=1e-7,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"{channel}: initial learned "
                "weights do not equal EXP002."
            )

    dummy = torch.randn(
        16,
        graph.num_nodes,
        64,
    )

    fixed.eval()
    learned.eval()

    with torch.no_grad():
        y_fixed = fixed(
            dummy
        )

        y_learned = learned(
            dummy
        )

    if not torch.allclose(
        y_fixed,
        y_learned,
        atol=1e-7,
        rtol=0.0,
    ):
        max_diff = float(
            torch.max(
                torch.abs(
                    y_fixed
                    - y_learned
                )
            )
        )

        raise RuntimeError(
            f"{channel}: EXP004 initial "
            f"prediction differs from EXP002 "
            f"by {max_diff}"
        )

    print(
        f"{channel}: PASSED | "
        f"mission={graph.mission} | "
        f"nodes={graph.num_nodes} | "
        f"support_edges={graph.num_edges} | "
        f"learned_edge_params="
        f"{learned.edge_logits.numel()}"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("V2 EXP004 MATCHED INITIALIZATION VERIFICATION")
    print("=" * 70)

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
        "VERIFICATION PASSED: EXP004 starts "
        "from the exact EXP002 equal-weight graph "
        "and differs only by making existing edge "
        "weights trainable."
    )
