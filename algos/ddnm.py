import torch
from tqdm import tqdm
from utils.ddpm import get_named_beta_schedule
from utils.torchvision_compat import patch_torchvision_fake_registration

patch_torchvision_fake_registration()

from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)


class DDNM:
    """Diffusion implicit consistent equilibrium (DICE) sampler."""

    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        img_size=32,
        device="cuda",
        schedule_name="cosine",
        channels=1,
        eta=0.85,
    ):
        """Initialize the sampler.

        Parameters
        ----------
        noise_steps : int
            Number of diffusion steps.
        beta_start, beta_end : float
            Noise schedule range.
        img_size : int
            Spatial size of the image.
        device : str
            Device on which to run.
        schedule_name : str
            Noise schedule type.
        channels : int
            Number of image channels.
        rho : float
            Rho
        mu : float
            Relaxation parameter for the outer update.
        skip_type : str
            Strategy to subsample diffusion steps.
        iter_num : int
            Number of diffusion iterations.
        """
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

        self.eta = eta
        self.alpha_hat_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alpha_hat[:-1]])

    def prepare_noise_schedule(self):
        if self.schedule_name == "cosine":
            return torch.tensor(
                get_named_beta_schedule("cosine", self.noise_steps, self.beta_end).copy(),
                dtype=torch.float32,
            )

        if self.schedule_name == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample(self, model, y, pseudo_inverse, forward_pass, ground_truth, track_metrics):

        x = torch.randn((1, self.channels, self.img_size, self.img_size)).to(self.device)

        pbar = tqdm(list(range(self.noise_steps))[::-1])

        A_psudo_inverse_y = pseudo_inverse(y)

        metrics = {
            "psnr": [],
            "ssim": [],
            "consistency": [],
            "error": [],
        }

        if track_metrics and ground_truth is not None:
            SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            PSNR = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        for i in pbar:
            t = (torch.ones(1) * i).long().to(self.device)
            alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1)
            sqrt_alpha_hat = torch.sqrt(alpha_hat)
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)
            alpha_hat_prev = self.alpha_hat_prev[t].view(-1, 1, 1, 1)

            with torch.no_grad():
                predicted_noise = model(x, t)

            x_0_t = (x - sqrt_one_minus_alpha_hat * predicted_noise) / sqrt_alpha_hat
            X_hat_0_t = x_0_t - pseudo_inverse(forward_pass(x_0_t)) + A_psudo_inverse_y

            ############################################################

            c1 = (1 - alpha_hat_prev).sqrt() * self.eta
            c2 = (1 - alpha_hat_prev).sqrt() * ((1 - self.eta**2) ** 0.5)

            x = torch.sqrt(alpha_hat_prev) * X_hat_0_t + c1 * torch.randn_like(x) + c2 * predicted_noise

            ############################################################

            difference = y - forward_pass(x)
            norm = torch.linalg.norm(difference)

            error = x - ground_truth
            error_norm = torch.linalg.norm(error)

            if track_metrics and ground_truth is not None:
                with torch.inference_mode():
                    x_eval = (x + 1) / 2.0
                    gt_eval = (ground_truth + 1) / 2.0
                    x_eval = x_eval.clamp(0.0, 1.0)
                    gt_eval = gt_eval.clamp(0.0, 1.0)
                    psnr_val = PSNR(x_eval, gt_eval).item()
                    ssim_val = SSIM(x_eval, gt_eval).item()
                    metrics["psnr"].append(psnr_val)
                    metrics["ssim"].append(ssim_val)
                    metrics["consistency"].append(norm.item())
                    metrics["error"].append(error_norm.item())

            pbar.set_description(f"Sampling - Step {i} - Consistency: {norm.item():.4f} - Error: {error_norm.item():.4f}")

        return x
