from __future__ import annotations

import torch

from solvers.base_diffusion_solver import BaseDiffusionStepSolver, DiffusionStepResult


class DDNMSolver(BaseDiffusionStepSolver):
    """Single reverse-diffusion DDNM update."""

    name = "DDNM"

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
        eta: float = 1.0,
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
            clip_denoised=False,
        )
        self.eta = float(eta)

    def step(
        self,
        x_t: torch.Tensor,
        timestep: int,
        y: torch.Tensor,
        H,
    ) -> DiffusionStepResult:
        t = self._make_timestep_tensor(x_t, timestep)
        pseudo_inverse = getattr(H, "pseudo_inverse", H.transpose_pass)

        with torch.no_grad():
            predicted_noise = self.model(x_t, t)
            alpha_hat_t = self._extract(self.alpha_hat, t, x_t)
            sqrt_alpha_hat_t = torch.sqrt(alpha_hat_t)
            sqrt_one_minus_alpha_hat_t = torch.sqrt(1.0 - alpha_hat_t)
            alpha_hat_prev_t = self._extract(self.alpha_hat_prev, t, x_t)

            x0_t = (x_t - sqrt_one_minus_alpha_hat_t * predicted_noise) / torch.clamp(sqrt_alpha_hat_t, min=1e-8)
            A_pseudo_inverse_y = pseudo_inverse(y)
            x0_projected = x0_t - pseudo_inverse(H.forward_pass(x0_t)) + A_pseudo_inverse_y

            if int(timestep) == 0:
                x_prev = x0_projected
            else:
                c1 = torch.sqrt(1.0 - alpha_hat_prev_t) * self.eta
                c2 = torch.sqrt(1.0 - alpha_hat_prev_t) * max(0.0, 1.0 - self.eta**2) ** 0.5
                x_prev = torch.sqrt(alpha_hat_prev_t) * x0_projected
                x_prev = x_prev + c1 * torch.randn_like(x_t) + c2 * predicted_noise

        return DiffusionStepResult(
            x_prev=x_prev.detach(),
            x0_estimate=self._clamp_estimate(x0_projected.detach()),
            info={},
        )
