"""One bounded, identity-initialised scalar gate on the existing total message."""
import torch
from torch import nn
from v2_models_exp006 import TCNGCNFixedTuned


class MessageGatedGCN(TCNGCNFixedTuned):
    def __init__(self, num_nodes, hidden_size, adjacency, kernel_size=3,
                 dilations=(1, 2), gate_mode="fixed", remove_messages=False):
        super().__init__(num_nodes, hidden_size, adjacency, kernel_size, dilations)
        if gate_mode not in {"fixed", "static", "dynamic"}:
            raise ValueError(gate_mode)
        self.gate_mode = gate_mode
        self.remove_messages = bool(remove_messages)
        # Direct zero parameters consume no random numbers. Backbone initialisation
        # is identical for every role at a given seed, including the final ablation.
        self.register_parameter("gate_weight", nn.Parameter(torch.zeros(2*hidden_size))
                                if gate_mode == "dynamic" else None)
        self.register_parameter("gate_bias", nn.Parameter(torch.zeros(()))
                                if gate_mode in {"static", "dynamic"} else None)

    def _forward_gate(self, x):
        if x.ndim != 3 or x.shape[1] != self.num_nodes:
            raise ValueError("Expected [batch, graph nodes, lookback]")
        z = self.temporal(x)
        z = torch.relu(z * self.node_gain + self.node_bias)
        aggregate = self.adjacency @ z
        if self.remove_messages:
            # Remove the command aggregate BEFORE both message and gate paths.
            # The gate cannot retain a command-dependent bypass in the ablation.
            aggregate = torch.zeros_like(aggregate)
        if self.gate_mode == "fixed":
            gate = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        elif self.gate_mode == "static":
            gate = (2*torch.sigmoid(self.gate_bias)).expand(x.shape[0])
        else:
            context = torch.cat((z[:, 0], aggregate[:, 0]), dim=-1)
            gate = 2*torch.sigmoid((context*self.gate_weight).sum(dim=-1)+self.gate_bias)
        message = self.message(aggregate)
        if self.gate_mode != "fixed":
            message = message * gate[:, None, None]
        # Only telemetry is read out. Other adjacency rows are exactly zero.
        hidden = torch.relu(self.self_map(z) + message)
        prediction = x[:, 0, -1] + self.head(hidden[:, 0]).squeeze(-1)
        return prediction, gate

    def forward(self, x):
        return self._forward_gate(x)[0]

    def forward_with_gate(self, x):
        return self._forward_gate(x)
