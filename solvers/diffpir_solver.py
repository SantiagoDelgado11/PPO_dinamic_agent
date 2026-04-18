from __future__ import annotations

import torch

from algos.diffpir import conjugate_gradient
from solvers.base_diffusion_solver import BaseDiffusionStepSolver, DiffusionStepResult


class DiffPIRSolver(BaseDiffusionStepSolver):
    """Single reverse-diffusion DiffPIR update."""

    name = "DiffPIR"

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device,
        steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_name: str = "cosine",
        cg_iters: int = 5,
        img_size: int = 32,
        channels: int = 3,
        lambda_: float = 1.0,
        noise_level_img: float = 0.0,
        clip_denoised: bool = False,
        skip_type: str = "uniform",
        iter_num: int = 1000,
        eta: float = 0.0,
        zeta: float = 1.0,
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
        self.cg_iters = int(cg_iters)
        self.lambda_ = float(lambda_)
        self.noise_level_img = float(noise_level_img)
        self.skip_type = skip_type
        self.iter_num = int(iter_num)
        self.eta = float(eta)
        self.zeta = float(zeta)

    def step(
        self,
        x_t: torch.Tensor,
        timestep: int,
        y: torch.Tensor,
        H,
    ) -> DiffusionStepResult:
        t = self._make_timestep_tensor(x_t, timestep)

        with torch.no_grad():
            out = self._p_sample(self.model, x_t, timestep)
            z = out["pred_xstart"].detach()

            sigma_k = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t) / torch.clamp(
                self._extract(self.sqrt_alphas_cumprod, t, x_t),
                min=1e-8,
            )
            sigma = max(1e-3, self.noise_level_img)
            rho_t = self.lambda_ * (sigma**2) / torch.clamp(sigma_k.square(), min=1e-8)

            b = H.transpose_pass(y) + rho_t * z

            def A_fn(v: torch.Tensor) -> torch.Tensor:
                return H.transpose_pass(H.forward_pass(v)) + rho_t * v

            x0 = conjugate_gradient(A_fn, b, x0=z, n_iter=self.cg_iters).detach()

            if int(timestep) == 0:
                x_prev = x0
            else:
                t_prev = self._make_timestep_tensor(x_t, timestep - 1)
                sqrt_alpha_hat_t = self._extract(self.sqrt_alphas_cumprod, t, x_t)
                sqrt_one_minus_alpha_hat_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t)
                sqrt_alpha_hat_prev = self._extract(self.sqrt_alphas_cumprod, t_prev, x_t)
                sqrt_one_minus_alpha_hat_prev = self._extract(self.sqrt_one_minus_alphas_cumprod, t_prev, x_t)
                beta_t = self._extract(self.beta, t, x_t)

                eps = (x_t - sqrt_alpha_hat_t * x0) / torch.clamp(sqrt_one_minus_alpha_hat_t, min=1e-8)
                eta_sigma = (
                    self.eta
                    * sqrt_one_minus_alpha_hat_prev
                    / torch.clamp(sqrt_one_minus_alpha_hat_t, min=1e-8)
                    * torch.sqrt(beta_t)
                )
                residual_sigma = torch.sqrt(
                    torch.clamp(sqrt_one_minus_alpha_hat_prev.square() - eta_sigma.square(), min=0.0)
                )

                x_prev = sqrt_alpha_hat_prev * x0
                x_prev = x_prev + (1.0 - self.zeta) ** 0.5 * (
                    residual_sigma * eps + eta_sigma * torch.randn_like(x_t)
                )
                x_prev = x_prev + (self.zeta**0.5) * sqrt_one_minus_alpha_hat_prev * torch.randn_like(x_t)

        return DiffusionStepResult(
            x_prev=x_prev.detach(),
            x0_estimate=self._clamp_estimate(x0.detach()),
            info={"rho_t": float(rho_t.mean().item())},
        )
