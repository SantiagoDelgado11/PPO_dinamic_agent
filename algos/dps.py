import torch
from tqdm import tqdm
from utils.ddpm import get_named_beta_schedule
import matplotlib.pyplot as plt
import numpy as np


class DPS:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=32,
        device="cuda",
        schedule_name="linear",
        channels=1,
        # ------- DPS ------
        clip_denoised=False,
        scale=0.5,
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.schedule_name = schedule_name
        self.channels = channels

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # ---- DPS -----
        self.alpha_hat_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alpha_hat[:-1]])

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alpha_hat)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alpha_hat - 1.0)
        self.posterior_mean_coef1 = self.beta * torch.sqrt(self.alpha_hat_prev) / (1 - self.alpha_hat)
        self.posterior_mean_coef2 = (1.0 - self.alpha_hat_prev) * torch.sqrt(self.alpha) / (1 - self.alpha_hat)
        self.clip_denoised = clip_denoised
        self.posterior_variance = self.beta * (1.0 - self.alpha_hat_prev) / (1.0 - self.alpha_hat)
        self.posterior_log_variance_clipped = torch.log(torch.cat((self.posterior_variance[1:2], self.posterior_variance[1:])))
        self.scale = scale
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alpha_hat)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alpha_hat)

    def prepare_noise_schedule(self):
        if self.schedule_name == "cosine":
            return torch.tensor(
                get_named_beta_schedule("cosine", self.noise_steps, self.beta_end).copy(),
                dtype=torch.float32,
            )

        if self.schedule_name == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample(self, model, y, forward_pass):
        x_start = torch.randn((1, self.channels, self.img_size, self.img_size), device=self.device).requires_grad_()

        img = x_start

        pbar = tqdm(list(range(self.noise_steps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=self.device)

            img = img.requires_grad_()
            out = self.p_sample(x=img, t=time, model=model)
            # Give condition.
            noisy_measurement = self.q_sample(y, t=time)

            img, distance = self.conditioning(
                x_t=out["sample"],
                measurement=y,
                x_prev=img,
                x_0_hat=out["pred_xstart"],
                forward_pass=forward_pass,
            )
            img = img.detach_()

            pbar.set_postfix({"distance": distance.item()}, refresh=False)

        return img

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

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, forward_pass):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, forward_pass=forward_pass)
        x_t -= norm_grad * self.scale
        return x_t, norm

    def grad_and_value(self, x_prev, x_0_hat, measurement, forward_pass):
        difference = measurement - forward_pass(x_0_hat)
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        return norm_grad, norm
