from __future__ import annotations

import math

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """PPO actor-critic MLP with orthogonal initialization."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.activation_name = activation.lower()
        self.dropout = max(0.0, float(dropout))

        layers: list[nn.Module] = []
        input_dim = state_dim
        for _ in range(max(1, int(hidden_layers))):
            layers.extend(
                [
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    self._make_activation(),
                ]
            )
            if self.dropout > 0.0:
                layers.append(nn.Dropout(p=self.dropout))
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _make_activation(self) -> nn.Module:
        if self.activation_name == "gelu":
            return nn.GELU()
        if self.activation_name == "relu":
            return nn.ReLU()
        if self.activation_name == "tanh":
            return nn.Tanh()
        return nn.SiLU()

    def _reset_parameters(self) -> None:
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(state)
        logits = torch.clamp(self.policy_head(features), -10.0, 10.0)
        value = self.value_head(features).squeeze(-1)
        return logits, value
