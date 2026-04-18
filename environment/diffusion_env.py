from __future__ import annotations

import argparse
from typing import Any

import torch

from environment import reward as reward_utils
from environment.state_builder import StateBuilder
from solvers.solver_library import SolverLibrary


def psnr_normalized(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    return float(reward_utils.psnr_normalized(x_hat, x_true))


def ssim_reward(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    return float(reward_utils.ssim_reward(x_hat, x_true))


def get_args():
    parser = argparse.ArgumentParser(description="Diffusion RL Environment")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


class EpisodeSample:
    """Single inverse-problem sample for one PPO episode."""

    def __init__(
        self,
        x_true: torch.Tensor,
        H: Any,
        noise_std: float = 0.0,
    ) -> None:
        self.x_true = x_true
        self.H = H
        self.noise_std = noise_std


class ModelDomainOperator:
    """Pass-through wrapper used to keep operator interface consistent."""

    def __init__(self, operator: Any) -> None:
        self._operator = operator

    def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        return self._operator.forward_pass(x)

    def transpose_pass(self, y: torch.Tensor) -> torch.Tensor:
        return self._operator.transpose_pass(y)

    def pseudo_inverse(self, y: torch.Tensor) -> torch.Tensor:
        if hasattr(self._operator, "pseudo_inverse"):
            return self._operator.pseudo_inverse(y)
        return self._operator.transpose_pass(y)


class DiffusionSolverEnv:
    """Sequential environment that selects one solver at each reverse-diffusion step."""

    def __init__(
        self,
        solver_library: SolverLibrary,
        state_builder: StateBuilder,
        max_steps: int = 1000,
        device: str | torch.device = "cuda",
        psnr_reward_weight: float = 1.0,
        ssim_reward_weight: float = 0.0,
        use_ssim_in_reward: bool = False,
        verbose: bool = False,
        args=None,
    ) -> None:
        self.solver_library = solver_library
        self.state_builder = state_builder

        if args is not None:
            self.max_steps = max(1, int(args.max_steps))
            self.device = torch.device(args.device)
            self.verbose = bool(getattr(args, "verbose", verbose))
        else:
            self.max_steps = max(1, int(max_steps))
            self.device = torch.device(device)
            self.verbose = bool(verbose)

        self.psnr_reward_weight = float(psnr_reward_weight)
        self.ssim_reward_weight = float(ssim_reward_weight)
        self.use_ssim_in_reward = bool(use_ssim_in_reward)

        self.iteration = 0
        self.current_timestep = self.max_steps - 1
        self.previous_action = 0
        self.current_sample: EpisodeSample | None = None
        self.y: torch.Tensor | None = None
        self.x_current: torch.Tensor | None = None
        self.x_estimate: torch.Tensor | None = None
        self.x_previous_estimate: torch.Tensor | None = None
        self.operator_model_domain: ModelDomainOperator | None = None
        self.previous_consistency = 0.0

    def _build_state(self) -> torch.Tensor:
        if self.y is None or self.operator_model_domain is None or self.x_current is None or self.x_estimate is None:
            raise RuntimeError("Environment state is not initialized.")

        return self.state_builder.build(
            y=self.y.to(self.device),
            H=self.operator_model_domain,
            x_latent=self.x_current,
            x_estimate=self.x_estimate,
            previous_estimate=self.x_previous_estimate,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            diffusion_timestep=max(self.current_timestep, 0),
            max_diffusion_timestep=max(self.max_steps - 1, 1),
            previous_action=self.previous_action,
            action_count=self.solver_library.action_dim,
            previous_consistency=self.previous_consistency,
        )

    def _build_measurement(self, x_true: torch.Tensor, H: Any, noise_std: float) -> torch.Tensor:
        measurement = H.forward_pass(x_true)
        if noise_std > 0.0:
            measurement = measurement + noise_std * torch.randn_like(measurement)
        return measurement

    def _compute_consistency_mse(self, x_estimate: torch.Tensor) -> float:
        if self.current_sample is None or self.y is None:
            return 0.0
        residual = self.current_sample.H.forward_pass(x_estimate) - self.y
        return float(torch.mean(residual.square()).item())

    def reset(self, sample: EpisodeSample) -> torch.Tensor:
        self.current_sample = sample
        self.iteration = 0
        self.current_timestep = self.max_steps - 1
        self.previous_action = 0

        self.operator_model_domain = ModelDomainOperator(sample.H)
        self.y = self._build_measurement(sample.x_true.to(self.device), sample.H, sample.noise_std).to(self.device)

        self.x_current = torch.randn_like(sample.x_true, device=self.device)
        self.x_estimate = torch.clamp(sample.H.transpose_pass(self.y).detach().to(self.device), -1.0, 1.0)
        self.x_previous_estimate = None
        self.previous_consistency = self._compute_consistency_mse(self.x_estimate)

        return self._build_state()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        if (
            self.current_sample is None
            or self.y is None
            or self.x_current is None
            or self.x_estimate is None
            or self.operator_model_domain is None
        ):
            raise RuntimeError("Call reset() before step().")

        x_true = self.current_sample.x_true.to(self.device)
        previous_estimate = self.x_estimate.detach()

        prev_psnr_norm = psnr_normalized(previous_estimate, x_true)
        prev_ssim = ssim_reward(previous_estimate, x_true)

        step_result = self.solver_library.apply_solver_step(
            action=action,
            x_t=self.x_current,
            timestep=self.current_timestep,
            y=self.y.to(self.device),
            Phi=self.operator_model_domain,
        )
        next_latent = step_result.x_prev.detach()
        next_estimate = torch.clamp(step_result.x0_estimate.detach(), -1.0, 1.0)

        next_psnr_norm = psnr_normalized(next_estimate, x_true)
        next_ssim = ssim_reward(next_estimate, x_true)

        delta_psnr_norm = next_psnr_norm - prev_psnr_norm
        delta_ssim = next_ssim - prev_ssim

        if self.use_ssim_in_reward:
            w_psnr = max(0.0, self.psnr_reward_weight)
            w_ssim = max(0.0, self.ssim_reward_weight)
            denom = w_psnr + w_ssim
            if denom <= 0.0:
                w_psnr, w_ssim, denom = 1.0, 0.0, 1.0
            reward = float((w_psnr * delta_psnr_norm + w_ssim * delta_ssim) / denom)
        else:
            reward = float(delta_psnr_norm)
        reward = float(max(-1.0, min(1.0, reward)))

        self.x_previous_estimate = previous_estimate
        self.x_current = next_latent
        self.x_estimate = next_estimate
        self.previous_action = int(action)
        consistency = self._compute_consistency_mse(self.x_estimate)
        self.previous_consistency = consistency

        self.iteration += 1
        self.current_timestep -= 1
        done = self.iteration >= self.max_steps

        if self.verbose:
            print(
                f"Step {self.iteration:04d}/{self.max_steps:04d}, "
                f"t={max(self.current_timestep + 1, 0):04d}, "
                f"action={action}, reward={reward:.4f}, "
                f"PSNR_norm {prev_psnr_norm:.3f}->{next_psnr_norm:.3f}, "
                f"SSIM {prev_ssim:.3f}->{next_ssim:.3f}, "
                f"Residual {consistency:.6f}"
            )

        next_state = (
            torch.zeros(self.state_builder.state_dim, dtype=torch.float32, device=self.device)
            if done
            else self._build_state()
        )

        info = {
            "solver": self.solver_library.get_solver(action).name,
            "reward": reward,
            "psnr_norm": next_psnr_norm,
            "psnr_norm_delta": delta_psnr_norm,
            "ssim": next_ssim,
            "ssim_delta": delta_ssim,
            "psnr_component": next_psnr_norm,
            "consistency": consistency,
            "timestep": max(self.current_timestep, -1),
            **step_result.info,
        }
        return next_state, reward, done, info
