import numpy as np
import torch

from v2_models_exp002 import (
    TCNGCNFixed,
    count_parameters,
)
from v2_graph import build_channel_graph


class TCNGCNLearnedEdges(TCNGCNFixed):
    """
    EXP004 learned-edge extension of EXP002.

    The graph support is fixed exactly as in EXP002.

    Only the relative weights of the existing incoming
    command -> telemetry edges are learned.

    Edge logits are initialized to zero, so softmax gives
    equal weights at initialization. Therefore EXP004 starts
    from the EXP002 graph weighting.
    """

    def __init__(
        self,
        num_nodes,
        hidden_size,
        adjacency,
        kernel_size=3,
        dilations=(1, 2),
    ):
        adjacency = np.asarray(
            adjacency,
            dtype=np.float32,
        )

        super().__init__(
            num_nodes=num_nodes,
            hidden_size=hidden_size,
            adjacency=adjacency,
            kernel_size=kernel_size,
            dilations=dilations,
        )

        source_nodes = np.flatnonzero(
            adjacency[0] > 0
        ).astype(
            np.int64
        )

        self.register_buffer(
            "source_nodes",
            torch.tensor(
                source_nodes,
                dtype=torch.long,
            ),
        )

        self.edge_logits = torch.nn.Parameter(
            torch.zeros(
                len(source_nodes),
                dtype=torch.float32,
            )
        )

    def current_edge_weights(self):
        if self.edge_logits.numel() == 0:
            return self.edge_logits

        return torch.softmax(
            self.edge_logits,
            dim=0,
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "[batch, nodes, lookback]."
            )

        if x.shape[1] != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, "
                f"received {x.shape[1]}."
            )

        z = self.temporal(
            x
        )

        z = torch.relu(
            z * self.node_gain
            + self.node_bias
        )

        if self.source_nodes.numel() == 0:
            aggregate = torch.zeros_like(
                z
            )

        else:
            weights = (
                self.current_edge_weights()
            )

            target_aggregate = torch.sum(
                z[
                    :,
                    self.source_nodes,
                    :,
                ]
                *
                weights[
                    None,
                    :,
                    None,
                ],
                dim=1,
            )

            if self.num_nodes == 1:
                aggregate = (
                    target_aggregate[
                        :,
                        None,
                        :,
                    ]
                )

            else:
                remaining = torch.zeros_like(
                    z[
                        :,
                        1:,
                        :,
                    ]
                )

                aggregate = torch.cat(
                    [
                        target_aggregate[
                            :,
                            None,
                            :,
                        ],
                        remaining,
                    ],
                    dim=1,
                )

        z = torch.relu(
            self.self_map(z)
            +
            self.message(
                aggregate
            )
        )

        persistence = x[
            :,
            0,
            -1,
        ]

        correction = (
            self.head(
                z[:, 0]
            )
            .squeeze(-1)
        )

        return (
            persistence
            + correction
        )


def smoke_test_channel(
    channel,
    hidden_size,
):
    graph = build_channel_graph(
        channel
    )

    model = TCNGCNLearnedEdges(
        num_nodes=graph.num_nodes,
        hidden_size=hidden_size,
        adjacency=graph.adjacency,
        kernel_size=3,
        dilations=(1, 2),
    )

    dummy = torch.randn(
        8,
        graph.num_nodes,
        64,
    )

    output = model(
        dummy
    )

    if output.shape != (8,):
        raise RuntimeError(
            f"{channel}: incorrect output shape."
        )

    weights = (
        model.current_edge_weights()
        .detach()
        .cpu()
        .numpy()
    )

    if len(weights) > 0:
        if not np.isclose(
            weights.sum(),
            1.0,
            atol=1e-7,
        ):
            raise RuntimeError(
                f"{channel}: learned weights "
                "do not sum to one."
            )

        expected = np.full(
            len(weights),
            1.0 / len(weights),
        )

        if not np.allclose(
            weights,
            expected,
            atol=1e-7,
        ):
            raise RuntimeError(
                f"{channel}: initial weights "
                "are not equal."
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
            f"{channel}: initial output "
            "is not persistence."
        )

    print(
        f"{channel} | "
        f"{graph.mission} | "
        f"support={graph.graph_mode} | "
        f"nodes={graph.num_nodes} | "
        f"edges={graph.num_edges} | "
        f"edge_params={len(weights)} | "
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
        "V2 EXP004 learned-edge model smoke test PASSED"
    )
