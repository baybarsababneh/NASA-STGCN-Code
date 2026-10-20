import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    """1D convolution with left-only padding."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=0,
        )

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class CommandAwareTCN(nn.Module):
    """
    Non-graph causal TCN control.

    Input:
        [batch, selected_features, lookback]

    Feature 0 is target telemetry. Remaining features are exactly the
    command histories that feed telemetry node 0 in EXP002's fixed graph:
      - SMAP: all fit-varying commands (star support)
      - MSL: fit-only selected top-4 commands

    Features are fused directly by the first temporal convolution. There is
    no graph construction or graph message passing inside this model.

    Prediction:
        persistence + learned residual correction
    """

    def __init__(
        self,
        num_features,
        hidden_size,
        kernel_size=3,
        dilations=(1, 2),
    ):
        super().__init__()

        if int(num_features) < 1:
            raise ValueError("num_features must be >= 1")
        if int(hidden_size) < 1:
            raise ValueError("hidden_size must be >= 1")
        if len(dilations) < 1:
            raise ValueError("At least one dilation is required")

        self.num_features = int(num_features)
        self.hidden_size = int(hidden_size)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)

        layers = []
        in_channels = self.num_features
        for dilation in self.dilations:
            layers.append(
                CausalConv1d(
                    in_channels=in_channels,
                    out_channels=self.hidden_size,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                )
            )
            layers.append(nn.ReLU())
            in_channels = self.hidden_size

        self.temporal = nn.Sequential(*layers)
        self.head = nn.Linear(self.hidden_size, 1)

        # Start exactly at persistence, matching EXP001/EXP002 philosophy.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def receptive_field(self):
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected [batch, features, lookback]")
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, got {x.shape[1]}"
            )

        z = self.temporal(x)
        latest = z[:, :, -1]
        correction = self.head(latest).squeeze(-1)
        persistence = x[:, 0, -1]
        return persistence + correction


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
