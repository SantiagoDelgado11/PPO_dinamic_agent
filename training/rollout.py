from __future__ import annotations
from dataclasses import dataclass
import torch
from agents.agent import ReinforceAgent
from environment.diffusion_env import DiffusionSolverEnv, EpisodeSample


@dataclass
class Trajectory:
    """Storage for one rollout trajectory."""

    states: list[torch.Tensor]
    actions: list[int]
    log_probs: list[torch.Tensor]
    entropies: list[torch.Tensor]
    values: list[torch.Tensor]
    rewards: list[float]
    dones: list[bool]


def discounted_returns(rewards: list[float], gamma: float, device: str | torch.device) -> torch.Tensor:
    """Compute discounted returns R_t."""
    returns = []
    running = 0.0

    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(
        returns, 
        dtype=torch.float32, 
        device=device,
    )


def rollout_episode(
    env: DiffusionSolverEnv,
    agent: ReinforceAgent,  
    sample: EpisodeSample,
    gamma: float,
    device: str | torch.device,
) -> tuple[Trajectory, torch.Tensor, dict]:
    """Collect one episode and compute discounted returns."""
    states: list[torch.Tensor] = []
    actions: list[int] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[float] = []
    dones: list[bool] = []

    state = env.reset(sample).to(device)
    done = False
    info = {}

    while not done:
        policy_step = agent.select_action(state.unsqueeze(0))

        next_state, reward, done, info = env.step(policy_step.action)

        states.append(state.detach())
        actions.append(policy_step.action)

        log_probs.append(policy_step.log_prob.squeeze())

        entropies.append(policy_step.entropy.squeeze())

        values.append(policy_step.value.squeeze())

        rewards.append(float(reward))
        dones.append(bool(done))

        state = next_state.to(device)

    trajectory = Trajectory(
        states=states,
        actions=actions,
        log_probs=log_probs,
        entropies=entropies,
        values=values,
        rewards=rewards,
        dones=dones,
    )
    returns = discounted_returns(
        rewards, 
        gamma=gamma,
        device=device,
    )
    return trajectory, returns, info
