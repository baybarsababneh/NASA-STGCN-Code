import numpy as np
import torch

from v2_graph import build_channel_graph
from v2_models_exp002 import (
    TCNGCNFixed,
    count_parameters,
)


class TCNNoGraphMatched(TCNGCNFixed):
    """
    EXP003 matched no-edge control for EXP002.

    The complete learnable architecture is inherited directly
    from the frozen EXP002 model.

    The only intended change is:

        EXP002 adjacency -> fixed command-to-target support
        EXP003 adjacency -> all zeros

    Therefore graph messages cannot contribute, while parameter
    count and parameter initialization order remain matched.
    """

    def __init__(
        self,
        num_nodes,
        hidden_size,
        kernel_size=3,
        dilations=(1, 2),
    ):
        zero_adjacency = np.zeros(
            (num_nodes, num_nodes),
            dtype=np.float32,
        )

        super().__init__(
            num_nodes=num_nodes,
            hidden_size=hidden_size,
            adjacency=zero_adjacency,
            kernel_size=kernel_size,
            dilations=dilations,
        )

        if (
            torch.count_nonzero(
                self.adjacency
            ).item()
            != 0
        ):
            raise RuntimeError(
                "EXP003 adjacency must contain only zeros."
            )


def smoke_test_channel(
    channel,
    hidden_size,
):
    reference_graph = build_channel_graph(
        channel
    )

    model = TCNNoGraphMatched(
        num_nodes=reference_graph.num_nodes,
        hidden_size=hidden_size,
        kernel_size=3,
        dilations=(1, 2),
    )

    dummy = torch.randn(
        8,
        reference_graph.num_nodes,
        64,
    )

    output = model(
        dummy
    )

    if output.shape != (8,):
        raise RuntimeError(
            f"{channel}: incorrect output shape "
            f"{output.shape}."
        )

    if not torch.isfinite(
        output
    ).all():
        raise RuntimeError(
            f"{channel}: non-finite output."
        )

    if (
        torch.count_nonzero(
            model.adjacency
        ).item()
        != 0
    ):
        raise RuntimeError(
            f"{channel}: adjacency is not zero."
        )

    persistence = dummy[
        :,
        0,
        -1,
    ]

    if not torch.allclose(
        output,
        persistence,
        atol=1e-7,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"{channel}: initial model does not "
            "equal persistence."
        )

    print(
        f"{channel} | "
        f"{reference_graph.mission} | "
        f"reference_mode="
        f"{reference_graph.graph_mode} | "
        f"nodes={reference_graph.num_nodes} | "
        f"reference_edges="
        f"{reference_graph.num_edges} | "
        f"active_edges=0 | "
        f"parameters="
        f"{count_parameters(model):,}"
    )


if __name__ == "__main__":
    smoke_test_channel(
        channel="E-1",
        hidden_size=64,
    )

    smoke_test_channel(
        channel="C-1",
        hidden_size=32,
    )

    print(
        "V2 EXP003 matched no-edge model "
        "smoke test PASSED"
    )
