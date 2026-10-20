import torch
from torch import nn


class CausalConv1d(nn.Module):
    """
    1D convolution with left-only padding.

    This ensures the representation at time t can only depend
    on time t and earlier values.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
    ):
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x):
        x = nn.functional.pad(
            x,
            (self.left_padding, 0),
        )

        return self.conv(x)


class TCNBaseline(nn.Module):
    """
    Causal multivariate TCN forecasting baseline.

    Input shape:
        [batch, features, lookback]

    The network receives the same telemetry + command-context
    information that later graph models will receive, but performs
    no graph message passing.

    Prediction:
        persistence forecast + learned residual correction
    """

    def __init__(
        self,
        num_features,
        hidden_size,
        kernel_size=3,
        dilations=(1, 2),
    ):
        super().__init__()

        if len(dilations) != 2:
            raise ValueError(
                "V2 exp001 expects exactly two TCN dilation layers."
            )

        self.temporal = nn.Sequential(
            CausalConv1d(
                in_channels=num_features,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                dilation=dilations[0],
            ),
            nn.ReLU(),

            CausalConv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                dilation=dilations[1],
            ),
            nn.ReLU(),
        )

        self.residual_head = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, x):
        """
        x:
            [batch, features, lookback]

        Column / feature 0 is telemetry.
        The final telemetry value in the input window is therefore
        the one-step persistence prediction.
        """

        z = self.temporal(x)

        latest_embedding = z[:, :, -1]

        residual = (
            self.residual_head(latest_embedding)
            .squeeze(-1)
        )

        persistence = x[:, 0, -1]

        prediction = persistence + residual

        return prediction


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    # ---------------------------------------------------------
    # Simple architecture smoke test.
    # ---------------------------------------------------------

    for mission, num_features, hidden_size in [
        ("SMAP", 25, 64),
        ("MSL", 55, 32),
    ]:
        model = TCNBaseline(
            num_features=num_features,
            hidden_size=hidden_size,
            kernel_size=3,
            dilations=(1, 2),
        )

        dummy = torch.randn(
            8,
            num_features,
            64,
        )

        output = model(dummy)

        assert output.shape == (8,)
        assert torch.isfinite(output).all()

        print(
            f"{mission}: "
            f"input={tuple(dummy.shape)}, "
            f"output={tuple(output.shape)}, "
            f"parameters={count_parameters(model):,}"
        )

    print("V2 TCN model smoke test PASSED")