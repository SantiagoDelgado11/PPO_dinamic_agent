from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from utils.ddpm import get_named_beta_schedule


@dataclass
class DiffusionStepResult:
    x_prev: torch.Tensor
    x0_estimate: torch.Tensor
    info: dict[str, Any] = field(default_factory=dict)


class BaseDiffusionStepSolver:
    """Shared DDPM schedule utilities for step-wise inverse-problem solvers."""

    name = "BaseSolver"

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device,
        steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        img_size: int = 32,
        schedule_name: str = "cosine",
        channels: int = 3,
        clip_denoised: bool = False,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.steps = int(steps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.img_size = int(img_size)
        self.schedule_name = schedule_name
        self.channels = int(channels)
        self.clip_denoised = bool(clip_denoised)

        self.beta = self._prepare_noise_schedule().to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = torch.cat([torch.ones(1, device=self.device), self.alpha_hat[:-1]])

        self.sqrt_alphas_cumprod = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alpha_hat)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alpha_hat)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alpha_hat - 1.0)

        self.posterior_mean_coef1 = self.beta * torch.sqrt(self.alpha_hat_prev) / (1.0 - self.alpha_hat)
        self.posterior_mean_coef2 = (1.0 - self.alpha_hat_prev) * torch.sqrt(self.alpha) / (1.0 - self.alpha_hat)
        self.posterior_variance = self.beta * (1.0 - self.alpha_hat_prev) / (1.0 - self.alpha_hat)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat((self.posterior_variance[1:2], self.posterior_variance[1:]))
        )

    def _prepare_noise_schedule(self) -> torch.Tensor:
        if self.schedule_name == "cosine":
            return torch.tensor(
                get_named_beta_schedule("cosine", self.steps, self.beta_end).copy(),
                dtype=torch.float32,
            )
        return torch.linspace(self.beta_start, self.beta_end, self.steps)

    def _make_timestep_tensor(self, x: torch.Tensor, timestep: int) -> torch.Tensor:
        return torch.full((x.shape[0],), int(timestep), dtype=torch.long, device=x.device)

    def _extract(self, array: torch.Tensor, time: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        values = array.to(target.device)[time].float()
        while values.ndim < target.ndim:
            values = values.unsqueeze(-1)
        return values.expand_as(target)

    def _process_xstart(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_denoised:
            return x.clamp(-1.0, 1.0)
        return x

    def _predict_xstart(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        coef1 = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, eps)
        return coef1 * x_t - coef2 * eps

    def _q_posterior_mean(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        coef1 = self._extract(self.posterior_mean_coef1, t, x_start)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t)
        return coef1 * x_start + coef2 * x_t

    def _get_variance(self, model_output: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        min_log = self._extract(self.posterior_log_variance_clipped, t, model_output)
        max_log = self._extract(torch.log(self.beta), t, model_output)
        frac = (model_output + 1.0) / 2.0
        model_log_variance = frac * max_log + (1.0 - frac) * min_log
        model_variance = torch.exp(model_log_variance)
        return model_variance, model_log_variance

    def _p_mean_variance(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        model_output = model(x_t, t)
        pred_xstart = self._process_xstart(self._predict_xstart(x_t, t, model_output))
        model_mean = self._q_posterior_mean(pred_xstart, x_t, t)
        model_variance, model_log_variance = self._get_variance(model_output, t)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "predicted_noise": model_output,
        }

    def _p_sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        timestep: int,
    ) -> dict[str, torch.Tensor]:
        t = self._make_timestep_tensor(x_t, timestep)
        out = self._p_mean_variance(model, x_t, t)
        sample = out["mean"]
        if int(timestep) > 0:
            sample = sample + torch.exp(0.5 * out["log_variance"]) * torch.randn_like(x_t)
        out["sample"] = sample
        out["timestep_tensor"] = t
        return out

    @staticmethod
    def _clamp_estimate(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, -1.0, 1.0)
