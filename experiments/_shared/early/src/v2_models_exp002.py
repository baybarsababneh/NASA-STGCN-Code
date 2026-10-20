import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

from v2_graph import build_channel_graph


class CausalNodeTCN(nn.Module):
    """
    Shared causal temporal encoder for graph nodes.

    Input:
        x: [batch, nodes, lookback]

    Every graph node is treated as its own 1D time series.
    The same temporal filters are shared across all nodes.

    For the locked V2 architecture:
        kernel size = 3
        dilation 1 = 1
        dilation 2 = 2

    Effective receptive field = 7 observations.
    """

    def __init__(
        self,
        hidden_size,
        kernel_size=3,
        dilations=(1, 2),
    ):
        super().__init__()

        if len(dilations) != 2:
            raise ValueError(
                "EXP002 expects exactly two TCN layers."
            )

        self.hidden_size = int(
            hidden_size
        )

        self.kernel_size = int(
            kernel_size
        )

        self.dilations = tuple(
            int(d)
            for d in dilations
        )

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            dilation=self.dilations[0],
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            dilation=self.dilations[1],
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "[batch, nodes, lookback]."
            )

        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        lookback = x.shape[2]

        # -----------------------------------------------------
        # Treat every graph node as a separate time series,
        # while sharing the same TCN weights.
        # -----------------------------------------------------

        z = x.reshape(
            batch_size * num_nodes,
            1,
            lookback,
        )

        # -----------------------------------------------------
        # First causal convolution.
        # Left padding only -> no future leakage.
        # -----------------------------------------------------

        pad1 = (
            self.dilations[0]
            * (self.kernel_size - 1)
        )

        z = F.pad(
            z,
            (pad1, 0),
        )

        z = torch.relu(
            self.conv1(z)
        )

        # -----------------------------------------------------
        # Second causal convolution.
        # -----------------------------------------------------

        pad2 = (
            self.dilations[1]
            * (self.kernel_size - 1)
        )

        z = F.pad(
            z,
            (pad2, 0),
        )

        z = torch.relu(
            self.conv2(z)
        )

        # Only the latest causal representation is needed for
        # one-step-ahead prediction.
        z = z[:, :, -1]

        z = z.reshape(
            batch_size,
            num_nodes,
            self.hidden_size,
        )

        return z


class TCNGCNFixed(nn.Module):
    """
    EXP002 fixed-graph TCN-GCN.

    Nodes:
        node 0     -> telemetry target
        nodes 1... -> command-context inputs

    Pipeline:
        shared causal TCN per node
        -> node-specific gain/bias
        -> one fixed graph message-passing layer
        -> telemetry-node readout
        -> persistence + learned residual

    The adjacency is fixed and contains no learnable edges.
    """

    def __init__(
        self,
        num_nodes,
        hidden_size,
        adjacency,
        kernel_size=3,
        dilations=(1, 2),
    ):
        super().__init__()

        adjacency = np.asarray(
            adjacency,
            dtype=np.float32,
        )

        if adjacency.shape != (
            num_nodes,
            num_nodes,
        ):
            raise ValueError(
                "Adjacency shape does not match "
                f"num_nodes={num_nodes}: "
                f"{adjacency.shape}"
            )

        self.num_nodes = int(
            num_nodes
        )

        self.hidden_size = int(
            hidden_size
        )

        self.temporal = CausalNodeTCN(
            hidden_size=self.hidden_size,
            kernel_size=kernel_size,
            dilations=dilations,
        )

        # Node-specific affine adjustment after the shared
        # temporal encoder.
        self.node_gain = nn.Parameter(
            torch.ones(
                self.num_nodes,
                self.hidden_size,
            )
        )

        self.node_bias = nn.Parameter(
            torch.zeros(
                self.num_nodes,
                self.hidden_size,
            )
        )

        # Self transformation and graph-message transformation.
        self.self_map = nn.Linear(
            self.hidden_size,
            self.hidden_size,
        )

        self.message = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
        )

        self.head = nn.Linear(
            self.hidden_size,
            1,
        )

        self.register_buffer(
            "adjacency",
            torch.tensor(
                adjacency,
                dtype=torch.float32,
            ),
        )

        # Start as a pure persistence forecaster.
        # Training learns only the residual correction.
        nn.init.zeros_(
            self.head.weight
        )

        nn.init.zeros_(
            self.head.bias
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

        # -----------------------------------------------------
        # Temporal encoding.
        # -----------------------------------------------------

        z = self.temporal(x)

        # -----------------------------------------------------
        # Node-specific affine transformation.
        # -----------------------------------------------------

        z = torch.relu(
            z * self.node_gain
            + self.node_bias
        )

        # -----------------------------------------------------
        # Fixed graph message passing.
        #
        # adjacency[target, source]
        #
        # For telemetry node 0 this forms the weighted aggregate
        # of command-node embeddings selected by v2_graph.py.
        # -----------------------------------------------------

        aggregate = (
            self.adjacency
            @ z
        )

        z = torch.relu(
            self.self_map(z)
            +
            self.message(aggregate)
        )

        # -----------------------------------------------------
        # Residual prediction from telemetry node 0.
        # -----------------------------------------------------

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


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def smoke_test_channel(
    channel,
    hidden_size,
):
    graph = build_channel_graph(
        channel
    )

    model = TCNGCNFixed(
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
            f"{channel}: incorrect output shape "
            f"{output.shape}."
        )

    if not torch.isfinite(
        output
    ).all():
        raise RuntimeError(
            f"{channel}: non-finite model output."
        )

    # Because the readout head is initially zero,
    # the untouched model must initially equal persistence.
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
            f"{channel}: initial output is not "
            "equal to persistence."
        )

    print(
        f"{channel} | "
        f"{graph.mission} | "
        f"mode={graph.graph_mode} | "
        f"nodes={graph.num_nodes} | "
        f"edges={graph.num_edges} | "
        f"input={tuple(dummy.shape)} | "
        f"output={tuple(output.shape)} | "
        f"parameters={count_parameters(model):,}"
    )


if __name__ == "__main__":
    # One SMAP and one MSL real graph.
    smoke_test_channel(
        channel="E-1",
        hidden_size=64,
    )

    smoke_test_channel(
        channel="C-1",
        hidden_size=32,
    )

    print(
        "V2 EXP002 TCN-GCN model smoke test PASSED"
    )