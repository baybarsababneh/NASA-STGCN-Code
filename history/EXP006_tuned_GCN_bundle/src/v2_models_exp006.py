import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CausalNodeTCNVariable(nn.Module):
    """Shared causal temporal encoder for graph nodes with a variable dilation stack."""

    def __init__(self, hidden_size, kernel_size=3, dilations=(1, 2)):
        super().__init__()
        if int(hidden_size) < 1:
            raise ValueError("hidden_size must be >= 1")
        if int(kernel_size) < 1:
            raise ValueError("kernel_size must be >= 1")
        if len(dilations) < 1:
            raise ValueError("At least one dilation is required")

        self.hidden_size = int(hidden_size)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)
        if any(d < 1 for d in self.dilations):
            raise ValueError("All dilations must be positive")

        convs = []
        in_channels = 1
        for dilation in self.dilations:
            convs.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=self.hidden_size,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    padding=0,
                )
            )
            in_channels = self.hidden_size
        self.convs = nn.ModuleList(convs)

    @property
    def receptive_field(self):
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected [batch, nodes, lookback]")

        batch_size, num_nodes, lookback = x.shape
        z = x.reshape(batch_size * num_nodes, 1, lookback)

        for conv, dilation in zip(self.convs, self.dilations):
            left_padding = dilation * (self.kernel_size - 1)
            z = F.pad(z, (left_padding, 0))
            z = torch.relu(conv(z))

        z = z[:, :, -1]
        return z.reshape(batch_size, num_nodes, self.hidden_size)


class TCNGCNFixedTuned(nn.Module):
    """
    EXP006 fixed-edge TCN-GCN.

    The graph support and aggregation are intentionally the same as EXP002.
    EXP006 changes only the bounded temporal/capacity/learning-rate search.
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

        adjacency = np.asarray(adjacency, dtype=np.float32)
        if adjacency.shape != (int(num_nodes), int(num_nodes)):
            raise ValueError(
                f"Adjacency shape {adjacency.shape} does not match num_nodes={num_nodes}"
            )

        self.num_nodes = int(num_nodes)
        self.hidden_size = int(hidden_size)
        self.temporal = CausalNodeTCNVariable(
            hidden_size=self.hidden_size,
            kernel_size=kernel_size,
            dilations=dilations,
        )

        self.node_gain = nn.Parameter(
            torch.ones(self.num_nodes, self.hidden_size)
        )
        self.node_bias = nn.Parameter(
            torch.zeros(self.num_nodes, self.hidden_size)
        )
        self.self_map = nn.Linear(self.hidden_size, self.hidden_size)
        self.message = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.head = nn.Linear(self.hidden_size, 1)

        self.register_buffer(
            "adjacency", torch.tensor(adjacency, dtype=torch.float32)
        )

        # Preserve EXP002's persistence-residual initialisation.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def receptive_field(self):
        return self.temporal.receptive_field

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected [batch, nodes, lookback]")
        if x.shape[1] != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, received {x.shape[1]}"
            )

        z = self.temporal(x)
        z = torch.relu(z * self.node_gain + self.node_bias)
        aggregate = self.adjacency @ z
        z = torch.relu(self.self_map(z) + self.message(aggregate))

        persistence = x[:, 0, -1]
        correction = self.head(z[:, 0]).squeeze(-1)
        return persistence + correction


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
