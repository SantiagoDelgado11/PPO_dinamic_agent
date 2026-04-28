from __future__ import annotations

from dataclasses import dataclass

import torch

from agents.agent import PPOAgent
from environment.diffusion_env import DiffusionSolverEnv, EpisodeSample


@dataclass
class Trajectory:
    """Storage for one rollout trajectory."""

    states: list[torch.Tensor]
    actions: list[int]
    log_probs: list[torch.Tensor]
    values: list[torch.Tensor]
    rewards: list[float]
    dones: list[bool]


def rollout_episode(
    env: DiffusionSolverEnv,
    agent: PPOAgent,
    sample: EpisodeSample,
    device: str | torch.device,
) -> tuple[Trajectory, dict]:
    """Collect one episode trajectory."""
    states: list[torch.Tensor] = []
    actions: list[int] = []
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[float] = []
    dones: list[bool] = []

    state = env.reset(sample).to(device)
    done = False
    info = {}

    while not done:
        with torch.no_grad():
            policy_step = agent.select_action(state.unsqueeze(0))

        next_state, reward, done, info = env.step(policy_step.action)

        states.append(state.detach())
        actions.append(policy_step.action)

        log_probs.append(policy_step.log_prob.squeeze())
        values.append(policy_step.value.squeeze())

        rewards.append(float(reward))
        dones.append(bool(done))

        state = next_state.to(device)

    trajectory = Trajectory(
        states=states,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
    )
    return trajectory, info
