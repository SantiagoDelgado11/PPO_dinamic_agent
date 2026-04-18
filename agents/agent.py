from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical

from agents.policy_network import PolicyNetwork


@dataclass
class AgentStep:
    action: int
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class PPOAgent(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        logit_temperature: float = 1.0,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.network = PolicyNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)

        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.logit_temperature = max(float(logit_temperature), 1e-3)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.network(state)

    def _distribution(self, state: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        logits, value = self.network(state)
        logits = logits / self.logit_temperature
        return Categorical(logits=logits), value

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> AgentStep:
        dist, value = self._distribution(state)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        return AgentStep(
            action=int(action.item()),
            log_prob=dist.log_prob(action),
            entropy=dist.entropy(),
            value=value,
        )

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, values = self._distribution(states)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values


ReinforceAgent = PPOAgent
