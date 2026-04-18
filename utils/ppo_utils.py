from __future__ import annotations

from pathlib import Path

import torch

from solvers.ddnm_solver import DDNMSolver
from solvers.diffpir_solver import DiffPIRSolver
from solvers.dps_solver import DPSSolver
from solvers.solver_library import SolverLibrary


DEFAULT_DIFFUSION_WEIGHTS = (
    "weights/e_1000_bs_64_lr_0.0003_seed_2_img_32_schedule_cosine_gpu_0_c_3_si_100/"
    "checkpoints/latest.pth.tar"
)
DEFAULT_PPO_CHECKPOINT_DIR = "weights/ppo_solver_selector"
DEFAULT_PPO_AGENT_WEIGHTS = f"{DEFAULT_PPO_CHECKPOINT_DIR}/best_agent.pt"
DEFAULT_PPO_ALGO = "PPO"
DEFAULT_DIFFUSION_STEPS = 1000
DEFAULT_POLICY_HIDDEN_DIM = 256

DEFAULT_DDNM_ETA = 1.0
DEFAULT_DPS_SCALE = 0.0125
DEFAULT_DIFFPIR_CG_ITERS = 5
DEFAULT_DIFFPIR_LAMBDA = 1.0
DEFAULT_DIFFPIR_NOISE_LEVEL = 0.0
DEFAULT_DIFFPIR_ETA = 0.0
DEFAULT_DIFFPIR_ZETA = 1.0

SOLVER_NAMES = ["DDNM", "DPS", "DiffPIR"]


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    return module


def resolve_agent_checkpoint(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    search_dir = candidate.parent if str(candidate.parent) not in {"", "."} else Path(DEFAULT_PPO_CHECKPOINT_DIR)
    fallback_names = [
        candidate.name,
        "best_agent.pt",
        "latest_agent.pt",
        "best.pt",
        "latest.pt",
    ]
    for name in fallback_names:
        fallback = search_dir / name
        if fallback.exists():
            return fallback

    raise FileNotFoundError(
        f"PPO agent checkpoint not found at '{candidate}'. "
        f"Expected one of: {', '.join(str(search_dir / name) for name in fallback_names)}"
    )


def load_agent_checkpoint(
    agent: torch.nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device,
) -> dict:
    resolved = resolve_agent_checkpoint(checkpoint_path)
    checkpoint = torch.load(resolved, map_location=device, weights_only=True)
    state_dict = checkpoint["agent_state"] if isinstance(checkpoint, dict) and "agent_state" in checkpoint else checkpoint
    agent.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {"agent_state": state_dict, "path": str(resolved)}


def build_solver_library(
    model: torch.nn.Module,
    device: str | torch.device,
    image_size: int,
    channels: int,
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
    ddnm_eta: float = DEFAULT_DDNM_ETA,
    dps_scale: float = DEFAULT_DPS_SCALE,
    diffpir_cg_iters: int = DEFAULT_DIFFPIR_CG_ITERS,
    diffpir_lambda: float = DEFAULT_DIFFPIR_LAMBDA,
    diffpir_noise_level_img: float = DEFAULT_DIFFPIR_NOISE_LEVEL,
    diffpir_eta: float = DEFAULT_DIFFPIR_ETA,
    diffpir_zeta: float = DEFAULT_DIFFPIR_ZETA,
) -> SolverLibrary:
    return SolverLibrary(
        diffpir_solver=DiffPIRSolver(
            model=model,
            device=device,
            img_size=image_size,
            channels=channels,
            steps=diffusion_steps,
            cg_iters=diffpir_cg_iters,
            lambda_=diffpir_lambda,
            noise_level_img=diffpir_noise_level_img,
            eta=diffpir_eta,
            zeta=diffpir_zeta,
            skip_type="uniform",
            iter_num=diffusion_steps,
        ),
        dps_solver=DPSSolver(
            model=model,
            device=device,
            img_size=image_size,
            channels=channels,
            steps=diffusion_steps,
            scale=dps_scale,
            clip_denoised=False,
        ),
        ddnm_solver=DDNMSolver(
            model=model,
            device=device,
            img_size=image_size,
            channels=channels,
            steps=diffusion_steps,
            eta=ddnm_eta,
        ),
    )
