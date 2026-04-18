import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import torch
import wandb

from utils.torchvision_compat import patch_torchvision_fake_registration

patch_torchvision_fake_registration()

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision import transforms
from torchvision.datasets import CIFAR10

from algos.ddnm import DDNM
from algos.diffpir import DiffPIR
from algos.dps import DPS
from algos.ppo import PPO
from guided_diffusion.script_util import create_model
from utils.SPC_model import SPCModel
from utils.ppo_utils import (
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_DIFFUSION_WEIGHTS,
    DEFAULT_POLICY_HIDDEN_DIM,
    DEFAULT_PPO_AGENT_WEIGHTS,
    DEFAULT_PPO_ALGO,
    freeze_module,
)
from utils.utils import set_seed


def build_backbone(weights_path: str, image_size: int, device: torch.device) -> torch.nn.Module:
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model = create_model(
        image_size=image_size,
        num_channels=64,
        num_res_blocks=3,
        input_channels=3,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return freeze_module(model)


def main(opt):
    print("Options:")
    for key, value in vars(opt).items():
        print(f"{key}: {value}")

    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() and opt.device.startswith("cuda") else "cpu")
    set_seed(opt.seed)

    ckpt = build_backbone(opt.weights, opt.image_size, device)

    dataset = CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    GT = dataset[opt.idx][0].unsqueeze(0).to(device)
    GT = GT * 2 - 1

    inverse_model = SPCModel(
        im_size=opt.image_size,
        compression_ratio=opt.sampling_ratio,
    ).to(device)

    y = inverse_model.forward_pass(GT)
    x_estimate = inverse_model.transpose_pass(y)

    action_summary = None

    if opt.algo == "DPS":
        diff = DPS(
            device=device,
            img_size=opt.image_size,
            noise_steps=opt.diffusion_steps,
            schedule_name="cosine",
            channels=3,
            scale=opt.dps_scale,
            clip_denoised=False,
        )
        reconstruction = diff.sample(
            model=ckpt,
            y=y,
            forward_pass=inverse_model.forward_pass,
        )
    elif opt.algo == "DDNM":
        diff = DDNM(
            device=device,
            img_size=opt.image_size,
            noise_steps=opt.diffusion_steps,
            schedule_name="cosine",
            channels=3,
            eta=opt.ddnm_eta,
        )
        reconstruction = diff.sample(
            model=ckpt,
            y=y,
            forward_pass=inverse_model.forward_pass,
            pseudo_inverse=inverse_model.pseudo_inverse,
            ground_truth=GT,
            track_metrics=opt.plot_metrics == "True",
        )
    elif opt.algo == "DiffPIR":
        diff = DiffPIR(
            device=device,
            img_size=opt.image_size,
            noise_steps=opt.diffusion_steps,
            schedule_name="cosine",
            channels=3,
            cg_iters=opt.CG_iters_diffpir,
            noise_level_img=opt.noise_level_img,
            iter_num=opt.iter_num,
            eta=opt.diffpir_eta,
            zeta=opt.diffpir_zeta,
            lambda_=opt.diffpir_lambda,
            skip_type=opt.skip_type,
        )
        reconstruction = diff.sample(
            model=ckpt,
            y=y,
            forward_pass=inverse_model.forward_pass,
            transpose_pass=inverse_model.transpose_pass,
        )
    elif opt.algo == DEFAULT_PPO_ALGO:
        diff = PPO(
            model=ckpt,
            agent_weights=opt.agent_weights,
            device=device,
            img_size=opt.image_size,
            noise_steps=opt.diffusion_steps,
            channels=3,
            hidden_dim=opt.policy_hidden_dim,
            ddnm_eta=opt.ddnm_eta,
            dps_scale=opt.dps_scale,
            cg_iters=opt.CG_iters_diffpir,
            diffpir_lambda=opt.diffpir_lambda,
            noise_level_img=opt.noise_level_img,
            diffpir_eta=opt.diffpir_eta,
            diffpir_zeta=opt.diffpir_zeta,
        )
        reconstruction, action_summary = diff.sample(
            y=y,
            operator=inverse_model,
        )
    else:
        raise ValueError("Invalid algorithm specified. Choose from 'DPS', 'DDNM', 'DiffPIR', or 'PPO'.")

    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)

    reconstruction = (reconstruction + 1) / 2
    GT = (GT + 1) / 2
    x_estimate_vis = ((x_estimate + 1) / 2).clamp(0, 1)

    reconstruction = reconstruction.clamp(0, 1)
    GT = GT.clamp(0, 1)

    error_map = (reconstruction - GT).abs().mean(dim=1, keepdim=True)

    ssim_pred = SSIM(reconstruction, GT)
    psnr_pred = PSNR(reconstruction, GT)
    consistency = torch.linalg.norm(inverse_model.forward_pass(reconstruction * 2 - 1) - y).item()
    error = torch.linalg.norm(reconstruction - GT).item()

    fig, ax = plt.subplots(1, 4, figsize=(25, 5))
    ax[0].imshow(GT[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[0].axis("off")
    ax[0].set_title("Ground Truth")

    ax[1].imshow(x_estimate_vis[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[1].axis("off")
    ax[1].set_title("A^T y")

    ax[2].imshow(reconstruction[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[2].axis("off")
    title = f"{opt.algo} Predicted\nSSIM: {ssim_pred:.4f}, PSNR: {psnr_pred:.2f}"
    if action_summary:
        counts_text = ", ".join(f"{name}:{count}" for name, count in sorted(action_summary["counts"].items()))
        title = f"{title}\n{counts_text}"
    ax[2].set_title(title)

    ax[3].imshow(error_map[0, 0].cpu().detach().numpy(), cmap="hot")
    ax[3].axis("off")
    ax[3].set_title(f"Error Map ({opt.algo} - GT)")
    plt.colorbar(ax[3].images[0], ax=ax[3])
    plt.suptitle(f"SPC Reconstruction with {opt.algo} Algorithm", fontsize=16)
    plt.tight_layout()

    if opt.save_image == "True":
        save_dir = f"results/SPC/{opt.algo}/{opt.sampling_method}/{opt.sampling_ratio:.2f}/{opt.idx}"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/reconstruction_idx_{opt.idx}_{psnr_pred:.2f}_{ssim_pred:.3f}.pdf",
            bbox_inches="tight",
        )

    if opt.plot == "True":
        plt.show()
    else:
        plt.close()

    if opt.use_wandb == "True":
        wandb.login(key=opt.wandb_id)
        wandb.init(
            project=opt.project_name,
            name=opt.name,
            config=vars(opt),
        )
        payload = {
            "SSIM": ssim_pred.item(),
            "PSNR": psnr_pred.item(),
            "Consistency": consistency,
            "Error": error,
        }
        if action_summary:
            for solver_name, count in action_summary["counts"].items():
                payload[f"solver_count/{solver_name}"] = count
        wandb.log(payload)

    print(
        f"SSIM ({opt.algo}): {ssim_pred:.3f}, "
        f"PSNR ({opt.algo}): {psnr_pred:.2f}, "
        f"Consistency: {consistency:.2f}, "
        f"Error: {error:.2f}"
    )
    if action_summary:
        print(f"PPO solver counts: {action_summary['counts']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--idx", type=int, default=10)
    p.add_argument("--weights", type=str, default=DEFAULT_DIFFUSION_WEIGHTS)
    p.add_argument("--agent_weights", type=str, default=DEFAULT_PPO_AGENT_WEIGHTS)
    p.add_argument("--batch_size", type=int, default=1, help="Unused for one-image inference")
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--save_image", type=str, default="False", choices=["True", "False"])
    p.add_argument("--plot", type=str, default="True", choices=["True", "False"])
    p.add_argument(
        "--plot_metrics",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Plot PSNR/SSIM diagnostics during DDNM sampling",
    )

    p.add_argument("--sampling_ratio", type=float, default=0.5)
    p.add_argument("--sampling_method", type=str, default="hadamard")
    p.add_argument(
        "--algo",
        type=str,
        default="DiffPIR",
        choices=["DPS", "DDNM", "DiffPIR", DEFAULT_PPO_ALGO],
    )
    p.add_argument("--diffusion_steps", type=int, default=DEFAULT_DIFFUSION_STEPS)

    p.add_argument("--ddnm_eta", type=float, default=1.0)
    p.add_argument("--dps_scale", type=float, default=0.0125)
    p.add_argument("--skip_type", type=str, default="uniform", choices=["uniform", "quad"])
    p.add_argument("--iter_num", type=int, default=DEFAULT_DIFFUSION_STEPS)
    p.add_argument("--CG_iters_diffpir", type=int, default=5)
    p.add_argument("--noise_level_img", type=float, default=0.0)
    p.add_argument("--diffpir_lambda", type=float, default=1.0)
    p.add_argument("--diffpir_eta", type=float, default=0.0)
    p.add_argument("--diffpir_zeta", type=float, default=1.0)
    p.add_argument("--policy_hidden_dim", type=int, default=DEFAULT_POLICY_HIDDEN_DIM)

    p.add_argument(
        "--use_wandb",
        type=str,
        default="False",
        choices=["True", "False"],
        help="Use Weights & Biases for logging",
    )
    p.add_argument("--wandb_id", type=str, default="b879bf20f3c31bfcf13289e363f4d3394f7d7671")
    p.add_argument("--project_name", type=str, default="STSIVA_2026")
    p.add_argument("--name", type=str, default="run_one_img")

    main(p.parse_args())
