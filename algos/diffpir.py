from __future__ import annotations

from typing import Callable, List, Tuple

import torch
from tqdm import tqdm
from utils.ddpm import get_named_beta_schedule
import numpy as np

# -----------------------------------------------------------------------------
#  Conjugate Gradient                                                          #
# -----------------------------------------------------------------------------


def conjugate_gradient(
    apply_A: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: torch.Tensor | None = None,
    n_iter: int = 5,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Early‑stopping batch CG that keeps autograd enabled for ``apply_A``."""

    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - apply_A(x)
    p = r.clone()
    rs_old = torch.sum(r * r, dim=list(range(1, r.ndim)), keepdim=True)

    for _ in range(n_iter):
        Ap = apply_A(p)
        alpha = rs_old / (torch.sum(p * Ap, dim=list(range(1, r.ndim)), keepdim=True) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r, dim=list(range(1, r.ndim)), keepdim=True)
        if torch.sqrt(rs_new.mean()) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


# -----------------------------------------------------------------------------
#  DiffPIR Sampler                                                             #
# -----------------------------------------------------------------------------
class DiffPIR:
    """Plug‑and‑Play diffusion sampler, now with *official* ρₜ schedule."""

    def __init__(
        self,
        *,
        noise_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_name: str = "linear",
        cg_iters: int = 40,
        img_size: int = 32,
        channels: int = 1,
        lambda_: float = 1.0,
        noise_level_img: float = None,
        clip_denoised: bool = False,
        device: str | torch.device = "cuda",
        skip_type: str = "quad",
        iter_num=20,
        eta: float = 0.0,
        zeta: float = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_name = schedule_name
        self.skip_type = skip_type
        self.iter_num = iter_num
        self.skip = self.noise_steps // self.iter_num
        self.eta = eta
        self.zeta = zeta
        # ---- diffusion schedule ----
        self.beta = self._prepare_noise_schedule().to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = torch.cat([torch.ones(1, device=self.device), self.alpha_hat[:-1]])

        # ---- official ρ_t schedule ----
        self.noise_level_img = noise_level_img
        self.sigma = max(0.001, noise_level_img)
        self.lambda_ = lambda_

        # ---- pre‑compute coeffs ----
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alpha_hat)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alpha_hat - 1.0)
        self.posterior_mean_coef1 = (
            self.beta * torch.sqrt(self.alpha_hat_prev) / (1.0 - self.alpha_hat)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_hat_prev) * torch.sqrt(self.alpha) / (1.0 - self.alpha_hat)
        )
        self.posterior_variance = self.beta * (1.0 - self.alpha_hat_prev) / (1.0 - self.alpha_hat)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alpha_hat)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alpha_hat)
        self.reduced_alpha_cumprod = torch.div(
            self.sqrt_1m_alphas_cumprod, self.sqrt_alphas_cumprod
        )
        self.t_start = self.noise_steps - 1
        # ---- misc ----
        self.T = noise_steps
        self.cg_iters = cg_iters
        self.clip_denoised = clip_denoised
        self.img_size = img_size
        self.channels = channels

    # ----------------------- helpers -----------------------
    def _prepare_noise_schedule(self) -> torch.Tensor:
        if self.schedule_name == "cosine":
            return torch.tensor(
                get_named_beta_schedule("cosine", self.noise_steps, self.beta_end).copy(),
                dtype=torch.float32,
            )
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        out = arr.to(ref.device)[t].float()
        while out.ndim < ref.ndim:
            out = out.unsqueeze(-1)
        return out.expand_as(ref)

    def p_sample(self, x, t, model):
        out = self.p_mean_variance(model, x, t)
        sample = out["mean"]
        noise = torch.randn_like(x)
        if t != 0:  # no noise when t == 0
            sample += torch.exp(0.5 * out["log_variance"]) * noise

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        coef1 = self.extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = self.extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return coef1 * x_start + coef2 * noise

    def p_mean_variance(self, model, x, t):
        model_output = model(x, t)
        model_var_values = model_output

        model_mean, pred_xstart = self.get_mean_and_xstart(x, t, model_output)
        model_variance, model_log_variance = self.get_variance(model_var_values, t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def get_variance(self, x, t):
        model_var_values = x
        min_log = self.posterior_log_variance_clipped
        max_log = torch.log(self.beta)

        min_log = self.extract_and_expand(min_log, t, x)
        max_log = self.extract_and_expand(max_log, t, x)

        # The model_var_values is [-1, 1] for [min_var, max_var]
        frac = (model_var_values + 1.0) / 2.0
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)
        return model_variance, model_log_variance

    def get_mean_and_xstart(self, x, t, model_output):
        pred_xstart = self.process_xstart(self.predict_xstart(x, t, model_output))
        mean = self.q_posterior_mean(pred_xstart, x, t)

        return mean, pred_xstart

    def process_xstart(self, x):
        if self.clip_denoised:
            x = x.clamp(-1, 1)
        return x

    def predict_xstart(self, x_t, t, eps):
        coef1 = self.extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = self.extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, eps)
        return coef1 * x_t - coef2 * eps

    def extract_and_expand(self, array, time, target):
        array = array.to(target.device)[time].float()
        while array.ndim < target.ndim:
            array = array.unsqueeze(-1)
        return array.expand_as(target)

    def q_posterior_mean(self, x_start, x_t, t):
        """
        Compute the mean of the diffusion posteriro:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        coef1 = self.extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = self.extract_and_expand(self.posterior_mean_coef2, t, x_t)
        return coef1 * x_start + coef2 * x_t

    def find_nearest(self, array, value):
        array = np.asarray(array.cpu())
        idx = (np.abs(array - value)).argmin()
        return idx

    def sample(
        self,
        model: torch.nn.Module,
        y: torch.Tensor,
        forward_pass: Callable[[torch.Tensor], torch.Tensor],
        transpose_pass: Callable[[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, List[float] | None]:

        y = y + torch.randn_like(y) * self.noise_level_img * 2

        ############################

        x_shape = transpose_pass(y).shape
        x = torch.randn(x_shape, device=self.device)

        ############################

        sigmas = []
        sigma_ks = []
        rhos = []
        for i in range(self.noise_steps):
            sigmas.append(self.reduced_alpha_cumprod[self.noise_steps - 1 - i])
            sigma_ks.append((self.sqrt_1m_alphas_cumprod[i] / self.sqrt_alphas_cumprod[i]))
            rhos.append(self.lambda_ * (self.sigma**2) / (sigma_ks[i] ** 2))

        rhos, sigmas, sigma_ks = (
            torch.tensor(rhos).to(self.device),
            torch.tensor(sigmas).to(self.device),
            torch.tensor(sigma_ks).to(self.device),
        )

        model.eval()

        if self.skip_type == "uniform":
            seq = [i * self.skip for i in range(self.iter_num)]
            if self.skip > 1:
                seq.append(self.noise_steps - 1)
        elif self.skip_type == "quad":
            seq = np.sqrt(np.linspace(0, self.noise_steps**2, self.iter_num))
            seq = [int(s) for s in list(seq)]
            seq[-1] = seq[-1] - 1
        progress_seq = seq[:: (len(seq) // 10)]
        progress_seq.append(seq[-1])

        pbar = tqdm(range(len(seq)))
        for i in pbar:
            curr_sigma = sigmas[seq[i]].cpu().numpy()
            t_i = self.find_nearest(self.reduced_alpha_cumprod, curr_sigma)
            if t_i > self.t_start:
                continue

            t_step = self.find_nearest(self.reduced_alpha_cumprod, (curr_sigma))
            vec_t = torch.tensor([t_step] * x.shape[0], device=x.device)

            # ---- prior ----
            model_out = self.p_sample(x, vec_t, model)
            z = model_out["pred_xstart"].detach()

            # ---- data fidelity ----
            rho_t = rhos[i]
            b = transpose_pass(y) + rho_t * z

            def A_fn(v):
                return transpose_pass(forward_pass(v)) + rho_t * v

            if not (seq[i] == seq[-1]):
                if i < self.noise_steps:
                    x0 = conjugate_gradient(A_fn, b, x0=z, n_iter=self.cg_iters).detach()
                else:
                    x0 = self.p_sample(x, vec_t, model)["sample"].detach()

            ################################
            if not (seq[i] == seq[-1]):
                t_im1 = self.find_nearest(
                    self.reduced_alpha_cumprod, sigmas[seq[i + 1]].cpu().numpy()
                )
                eps = (x - self.sqrt_alphas_cumprod[t_i] * x0) / self.sqrt_1m_alphas_cumprod[t_i]
                eta_sigma = (
                    self.eta
                    * self.sqrt_1m_alphas_cumprod[t_im1]
                    / self.sqrt_1m_alphas_cumprod[t_i]
                    * torch.sqrt(self.beta[t_i])
                )
                x = (
                    self.sqrt_alphas_cumprod[t_im1] * x0
                    + np.sqrt(1 - self.zeta)
                    * (
                        torch.sqrt(self.sqrt_1m_alphas_cumprod[t_im1] ** 2 - eta_sigma**2) * eps
                        + eta_sigma * torch.randn_like(x)
                    )
                    + np.sqrt(self.zeta) * self.sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                )
            ############ METRICS ############
            difference = y - forward_pass(x)
            norm = torch.linalg.norm(difference)
            pbar.set_description(f"Sampling - Step {i} - Distance: {norm.item():.4f}")
            ############ METRICS ############

        return x
