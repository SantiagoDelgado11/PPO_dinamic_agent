from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import optim

from agents.agent import PPOAgent
from environment.diffusion_env import DiffusionSolverEnv, EpisodeSample
from training.rollout import Trajectory, rollout_episode


@dataclass
class ReinforceTrainerConfig:
    """Configuration for PPO actor-critic training."""

    num_episodes: int = 1000
    gamma: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    optimizer: str = "adamw"
    grad_clip_norm: float = 1.0
    grad_explosion_threshold: float = 10.0
    checkpoint_dir: str = "weights/ppo_solver_selector"
    checkpoint_every: int = 25
    normalize_returns: bool = True
    normalize_advantages: bool = True
    max_abs_advantage: float = 5.0
    returns_norm_momentum: float = 0.99
    psnr_norm_aux_weight: float = 0.0
    critic_loss_type: str = "smooth_l1"
    huber_beta: float = 0.5
    ppo_clip_eps: float = 0.2
    ppo_update_epochs: int = 4
    gae_lambda: float = 0.98
    ppo_value_clip_eps: float = 0.2
    target_kl: float = 0.03
    episodes_per_update: int = 4
    eps: float = 1e-8


class ReinforceTrainer:
    """Trainer that optimizes policy and value heads jointly with PPO + GAE."""

    def __init__(
        self,
        agent: PPOAgent,
        env: DiffusionSolverEnv,
        config: ReinforceTrainerConfig,
        device: str | torch.device,
    ) -> None:
        self.agent = agent
        self.env = env
        self.config = config
        self.device = torch.device(device)

        optimizer_name = config.optimizer.lower()
        if optimizer_name == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.agent.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                alpha=0.99,
                eps=config.eps,
            )
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.agent.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optim.AdamW(
                self.agent.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        self._running_return_mean = 0.0
        self._running_return_var = 1.0
        self._best_reward = float("-inf")
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _checkpoint_payload(self, episode: int, episode_log: dict[str, float]) -> dict:
        return {
            "episode": episode,
            "agent_state": self.agent.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "metrics": episode_log,
            "best_reward": self._best_reward,
        }

    def _save_checkpoint(self, episode: int, episode_log: dict[str, float], is_best: bool) -> None:
        checkpoint_dir = Path(self.config.checkpoint_dir)
        payload = self._checkpoint_payload(episode, episode_log)

        episode_path = checkpoint_dir / f"episode_{episode:05d}.pt"
        latest_path = checkpoint_dir / "latest.pt"
        latest_agent_path = checkpoint_dir / "latest_agent.pt"

        torch.save(payload, episode_path)
        torch.save(payload, latest_path)
        torch.save(self.agent.state_dict(), latest_agent_path)

        if is_best:
            torch.save(payload, checkpoint_dir / "best.pt")
            torch.save(self.agent.state_dict(), checkpoint_dir / "best_agent.pt")

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        std = tensor.std(unbiased=False)
        if torch.isnan(std) or std.item() < self.config.eps:
            return tensor - tensor.mean()
        return (tensor - tensor.mean()) / (std + self.config.eps)

    def _normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize(advantages)
        return torch.clamp(normalized, -self.config.max_abs_advantage, self.config.max_abs_advantage)

    def _normalize_returns_online(self, returns: torch.Tensor) -> torch.Tensor:
        mean = returns.mean().item()
        var = returns.var(unbiased=False).item() if returns.numel() > 1 else 0.0

        m = self.config.returns_norm_momentum
        self._running_return_mean = m * self._running_return_mean + (1.0 - m) * mean
        self._running_return_var = m * self._running_return_var + (1.0 - m) * max(var, self.config.eps)

        denom = (self._running_return_var**0.5) + self.config.eps
        normalized = (returns - self._running_return_mean) / denom
        return torch.clamp(normalized, -self.config.max_abs_advantage, self.config.max_abs_advantage)

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, dtype=torch.float32, device=rewards.device)
        next_value = torch.tensor(0.0, dtype=torch.float32, device=rewards.device)

        for t in reversed(range(rewards.shape[0])):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_value * not_done - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * not_done * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def _trajectory_to_tensors(
        self,
        trajectory: Trajectory,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states = torch.stack(trajectory.states).to(self.device)
        actions = torch.tensor(trajectory.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.stack(trajectory.log_probs).to(self.device).view(-1).detach()
        old_values = torch.stack(trajectory.values).to(self.device).view(-1).detach()
        rewards = torch.tensor(trajectory.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(trajectory.dones, dtype=torch.float32, device=self.device)
        return states, actions, old_log_probs, old_values, rewards, dones

    def _build_update_batch(
        self,
        collected_episodes: list[tuple[int, Trajectory, dict[str, Any]]],
    ) -> dict[str, Any]:
        batch_states = []
        batch_actions = []
        batch_old_log_probs = []
        batch_old_values = []
        batch_advantages = []
        batch_returns = []
        episode_summaries: list[dict[str, float]] = []
        solver_to_index = {"DDNM": 0, "DPS": 1, "DiffPIR": 2}

        for episode, trajectory, info in collected_episodes:
            states, actions, old_log_probs, old_values, rewards, dones = self._trajectory_to_tensors(trajectory)
            advantages, returns_gae = self._compute_gae(
                rewards=rewards,
                values=old_values,
                dones=dones,
            )

            psnr_norm = float(info.get("psnr_component", 0.0))
            if self.config.psnr_norm_aux_weight > 0.0:
                advantages = advantages + (self.config.psnr_norm_aux_weight * psnr_norm)

            batch_states.append(states)
            batch_actions.append(actions)
            batch_old_log_probs.append(old_log_probs)
            batch_old_values.append(old_values)
            batch_advantages.append(advantages)
            batch_returns.append(returns_gae)

            episode_summaries.append(
                {
                    "episode": float(episode),
                    "reward": float(sum(trajectory.rewards)),
                    "selected_action": float(trajectory.actions[-1]),
                    "selected_solver": float(solver_to_index[str(info["solver"])]),
                    "psnr_norm": psnr_norm,
                    "ssim": float(info.get("ssim", 0.0)),
                    "consistency": float(info.get("consistency", 0.0)),
                    "consistency_delta": float(info.get("consistency_delta", 0.0)),
                    "action_switch_penalty": float(info.get("action_switch_penalty", 0.0)),
                }
            )

        states = torch.cat(batch_states, dim=0)
        actions = torch.cat(batch_actions, dim=0)
        old_log_probs = torch.cat(batch_old_log_probs, dim=0)
        old_values = torch.cat(batch_old_values, dim=0)
        raw_returns = torch.cat(batch_returns, dim=0)
        advantages = torch.cat(batch_advantages, dim=0).detach()
        targets = raw_returns

        if self.config.normalize_returns:
            targets = self._normalize_returns_online(targets)
        if self.config.normalize_advantages:
            advantages = self._normalize_advantages(advantages)

        return {
            "states": states,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "old_values": old_values,
            "advantages": advantages,
            "targets": targets,
            "raw_returns": raw_returns,
            "episode_summaries": episode_summaries,
        }

    def _optimize_batch(self, batch: dict[str, Any]) -> dict[str, float] | None:
        states = batch["states"]
        actions = batch["actions"]
        old_log_probs = batch["old_log_probs"]
        old_values = batch["old_values"]
        advantages = batch["advantages"]
        targets = batch["targets"]
        raw_returns = batch["raw_returns"]

        ratio_mean = 1.0
        approx_kl = 0.0
        grad_norm_value = 0.0
        grad_exploded = False
        loss = torch.tensor(0.0, device=self.device)
        policy_loss = torch.tensor(0.0, device=self.device)
        value_loss = torch.tensor(0.0, device=self.device)
        entropy_bonus = torch.tensor(0.0, device=self.device)

        targets_flat = targets.view(-1)
        raw_returns_flat = raw_returns.view(-1)
        old_values_flat = old_values.view(-1)

        for _ in range(max(1, int(self.config.ppo_update_epochs))):
            log_probs, entropies, values = self.agent.evaluate_actions(states=states, actions=actions)
            values = values.view(-1)

            ratios = torch.exp(log_probs - old_log_probs)
            approx_kl = float((old_log_probs - log_probs).mean().detach().item())
            clipped_ratios = torch.clamp(
                ratios,
                1.0 - self.config.ppo_clip_eps,
                1.0 + self.config.ppo_clip_eps,
            )
            surrogate_1 = ratios * advantages
            surrogate_2 = clipped_ratios * advantages
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()

            values_clipped = old_values_flat + torch.clamp(
                values - old_values_flat,
                -self.config.ppo_value_clip_eps,
                self.config.ppo_value_clip_eps,
            )
            if self.config.critic_loss_type == "smooth_l1":
                value_loss_unclipped = F.smooth_l1_loss(values, targets_flat, beta=self.config.huber_beta)
                value_loss_clipped = F.smooth_l1_loss(values_clipped, targets_flat, beta=self.config.huber_beta)
            else:
                value_loss_unclipped = torch.mean((values - targets_flat) ** 2)
                value_loss_clipped = torch.mean((values_clipped - targets_flat) ** 2)
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

            entropy_bonus = entropies.mean()
            loss = policy_loss + self.agent.value_coef * value_loss - self.agent.entropy_coef * entropy_bonus

            if not torch.isfinite(loss) or approx_kl > self.config.target_kl:
                break

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.config.grad_clip_norm,
                error_if_nonfinite=False,
            )
            grad_norm_value = float(
                grad_norm_preclip.item() if isinstance(grad_norm_preclip, torch.Tensor) else grad_norm_preclip
            )
            grad_exploded = grad_norm_value > self.config.grad_explosion_threshold
            self.optimizer.step()
            ratio_mean = float(ratios.mean().item())

        if not torch.isfinite(loss):
            return None

        return {
            "loss_total": float(loss.item()),
            "loss_policy": float(policy_loss.item()),
            "loss_value": float(value_loss.item()),
            "entropy": float(entropy_bonus.item()),
            "advantage_mean": float(advantages.mean().item()),
            "returns_mean": float(targets.mean().item()),
            "returns_gae_mean": float(raw_returns_flat.mean().item()),
            "ratio_mean": float(ratio_mean),
            "approx_kl": float(approx_kl),
            "grad_norm": grad_norm_value,
            "grad_clipped_to": float(self.config.grad_clip_norm),
            "grad_exploded": float(grad_exploded),
            "running_return_mean": float(self._running_return_mean),
            "running_return_std": float(self._running_return_var**0.5),
        }

    def train(self, episode_sampler) -> list[dict[str, float]]:
        self.agent.train()
        logs: list[dict[str, float]] = []
        pending_episodes: list[tuple[int, Trajectory, dict[str, Any]]] = []
        episodes_per_update = max(1, int(self.config.episodes_per_update))
        solver_names = ("DDNM", "DPS", "DiffPIR")

        for episode in range(1, self.config.num_episodes + 1):
            sample: EpisodeSample = episode_sampler()

            trajectory, info = rollout_episode(
                env=self.env,
                agent=self.agent,
                sample=sample,
                device=self.device,
            )

            pending_episodes.append((episode, trajectory, info))

            if len(pending_episodes) < episodes_per_update and episode != self.config.num_episodes:
                continue

            batch = self._build_update_batch(pending_episodes)
            update_metrics = self._optimize_batch(batch)
            if update_metrics is None:
                pending_episodes.clear()
                continue

            episode_summaries = batch["episode_summaries"]
            for episode_summary in episode_summaries:
                episode_log = {**episode_summary, **update_metrics}
                logs.append(episode_log)

                episode_number = int(episode_log["episode"])
                episode_reward = float(episode_log["reward"])
                is_best = episode_reward >= self._best_reward
                if is_best:
                    self._best_reward = episode_reward

                if episode_number % 5 == 0 or episode_number == 1:
                    solver_name = solver_names[int(episode_log["selected_solver"])]
                    print(
                        f"[Episode {episode_number:04d}] "
                        f"reward={episode_log['reward']:.3f} "
                        f"solver={solver_name} "
                        f"psnr_norm={episode_log['psnr_norm']:.3f} "
                        f"ssim={episode_log['ssim']:.3f} "
                        f"consistency={episode_log['consistency']:.6f} "
                        f"cons_delta={episode_log['consistency_delta']:.3f} "
                        f"loss={episode_log['loss_total']:.4f}"
                    )

                if (
                    episode_number % self.config.checkpoint_every == 0
                    or is_best
                    or episode_number == self.config.num_episodes
                ):
                    self._save_checkpoint(episode_number, episode_log, is_best=is_best)


            pending_episodes.clear()

        return logs


PPOTrainerConfig = ReinforceTrainerConfig
PPOTrainer = ReinforceTrainer
