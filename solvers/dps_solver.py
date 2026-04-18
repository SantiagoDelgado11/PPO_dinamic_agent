from __future__ import annotations

import torch

from solvers.base_diffusion_solver import BaseDiffusionStepSolver, DiffusionStepResult


class DPSSolver(BaseDiffusionStepSolver):
    """Single reverse-diffusion DPS update."""

    name = "DPS"

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
        scale: float = 0.0125,
    ) -> None:
        super().__init__(
            model=model,
            device=device,
            steps=steps,
            beta_start=beta_start,
            beta_end=beta_end,
            img_size=img_size,
            schedule_name=schedule_name,
            channels=channels,
            clip_denoised=clip_denoised,
        )
        self.scale = float(scale)

    def _conditioning(
        self,
        x_prev: torch.Tensor,
        x_t: torch.Tensor,
        x_0_hat: torch.Tensor,
        measurement: torch.Tensor,
        forward_pass,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        difference = measurement - forward_pass(x_0_hat)
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, allow_unused=False)[0]
        return x_t - norm_grad * self.scale, norm

    def step(
        self,
        x_t: torch.Tensor,
        timestep: int,
        y: torch.Tensor,
        H,
    ) -> DiffusionStepResult:
        x_t = x_t.detach().clone().requires_grad_(True)
        out = self._p_sample(self.model, x_t, timestep)
        conditioned_sample, distance = self._conditioning(
            x_prev=x_t,
            x_t=out["sample"],
            x_0_hat=out["pred_xstart"],
            measurement=y,
            forward_pass=H.forward_pass,
        )
        return DiffusionStepResult(
            x_prev=conditioned_sample.detach(),
            x0_estimate=self._clamp_estimate(out["pred_xstart"].detach()),
            info={"measurement_distance": float(distance.detach().item())},
        )
