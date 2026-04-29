from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.torchvision_compat import patch_torchvision_fake_registration

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

patch_torchvision_fake_registration()

from torchvision import transforms
from torchvision.datasets import CIFAR10

try:
    import wandb
except ImportError:
    wandb = None

from agents.agent import PPOAgent
from environment.diffusion_env import DiffusionSolverEnv, EpisodeSample
from environment.state_builder import StateBuilder
from guided_diffusion.script_util import create_model
from training.reinforce_trainer import PPOTrainer, PPOTrainerConfig
from utils.SPC_model import SPCModel
from utils.ppo_utils import (
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_DIFFUSION_WEIGHTS,
    DEFAULT_POLICY_HIDDEN_DIM,
    DEFAULT_PPO_CHECKPOINT_DIR,
    build_solver_library,
    freeze_module,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_backbone(args, device: torch.device) -> torch.nn.Module:
    model = create_model(
        image_size=args.image_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        input_channels=args.input_channels,
    ).to(device)
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {args.weights}")

    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return freeze_module(model)


def train(args) -> None:
    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb no esta instalado. Instala con: pip install wandb")

        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
            wandb.login(key=args.wandb_api_key, relogin=True)
        else:
            wandb.login()

        wandb_config = {k: v for k, v in vars(args).items() if k != "wandb_api_key"}
        init_kwargs = {
            "project": args.project_name,
            "name": args.run_name,
            "config": wandb_config,
        }
        if args.wandb_entity:
            init_kwargs["entity"] = args.wandb_entity

        wandb_run = wandb.init(**init_kwargs)

    set_seed(args.seed)

    use_cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    device = torch.device(f"cuda:{args.gpu_id}" if use_cuda else "cpu")

    model = build_backbone(args=args, device=device)

    train_dataset = CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=use_cuda,
    )

    solver_library = build_solver_library(
        model=model,
        device=device,
        image_size=args.image_size,
        channels=args.input_channels,
        diffusion_steps=args.diffusion_steps,
        ddnm_eta=args.ddnm_eta,
        dps_scale=args.dps_scale,
        diffpir_cg_iters=args.CG_iters_diffpir,
        diffpir_lambda=args.diffpir_lambda,
        diffpir_noise_level_img=args.noise_level_img,
        diffpir_eta=args.diffpir_eta,
        diffpir_zeta=args.diffpir_zeta,
    )

    state_builder = StateBuilder()
    env = DiffusionSolverEnv(
        solver_library=solver_library,
        state_builder=state_builder,
        max_steps=args.diffusion_steps,
        device=device,
        psnr_reward_weight=args.reward_psnr_weight,
        ssim_reward_weight=args.reward_ssim_weight,
        use_ssim_in_reward=args.use_ssim_in_reward,
        consistency_reward_weight=args.reward_consistency_weight,
        action_switch_penalty=args.action_switch_penalty,
        final_quality_weight=args.final_quality_weight,
        reward_clip=args.reward_clip,
        verbose=args.env_verbose,
    )

    agent = PPOAgent(
        state_dim=state_builder.state_dim,
        action_dim=solver_library.action_dim,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        logit_temperature=args.logit_temperature,
        hidden_dim=args.policy_hidden_dim,
        hidden_layers=args.policy_hidden_layers,
        activation=args.policy_activation,
        dropout=args.policy_dropout,
    ).to(device)

    trainer = PPOTrainer(
        agent=agent,
        env=env,
        config=PPOTrainerConfig(
            num_episodes=args.num_episodes,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer,
            grad_clip_norm=args.grad_clip_norm,
            grad_explosion_threshold=args.grad_explosion_threshold,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            returns_norm_momentum=args.returns_norm_momentum,
            psnr_norm_aux_weight=args.psnr_norm_aux_weight,
            critic_loss_type=args.critic_loss_type,
            huber_beta=args.huber_beta,
            ppo_clip_eps=args.ppo_clip_eps,
            ppo_update_epochs=args.ppo_update_epochs,
            gae_lambda=args.gae_lambda,
            ppo_value_clip_eps=args.ppo_value_clip_eps,
            target_kl=args.target_kl,
            episodes_per_update=args.episodes_per_update,
        ),
        device=device,
    )

    operator = SPCModel(
        im_size=args.image_size,
        compression_ratio=args.sampling_ratio,
    ).to(device)

    data_iter = iter(dataloader)

    def sample_episode() -> EpisodeSample:
        nonlocal data_iter
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images, _ = batch
        x_true = images[0:1].to(device)
        x_true = x_true * 2.0 - 1.0

        return EpisodeSample(
            x_true=x_true,
            H=operator,
        )

    logs = trainer.train(sample_episode)

    if args.use_wandb and wandb_run is not None:
        for entry in logs:
            wandb.log(entry, step=int(entry.get("episode", 0)))

    rewards = [entry["reward"] for entry in logs]

    print("Training finished.")
    print(f"Episodes: {len(logs)}")
    if rewards:
        print(f"Average Reward: {np.mean(rewards):.4f}")
        print(f"Best Reward: {np.max(rewards):.4f}")
        print(f"Best agent checkpoint: {Path(args.checkpoint_dir) / 'best_agent.pt'}")

        if args.use_wandb and wandb_run is not None:
            wandb_run.summary["average_reward"] = float(np.mean(rewards))
            wandb_run.summary["best_reward"] = float(np.max(rewards))
    else:
        print("Average Reward: n/a (no valid episodes logged)")
        print("Best Reward: n/a (no valid episodes logged)")

    if args.use_wandb and wandb_run is not None:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO-based diffusion solver selector.")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--weights", type=str, default=DEFAULT_DIFFUSION_WEIGHTS)

    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=3)
    parser.add_argument("--diffusion_steps", type=int, default=DEFAULT_DIFFUSION_STEPS)

    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.98)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--logit_temperature", type=float, default=1.0)
    parser.add_argument("--policy_hidden_dim", type=int, default=DEFAULT_POLICY_HIDDEN_DIM)
    parser.add_argument("--policy_hidden_layers", type=int, default=3)
    parser.add_argument("--policy_activation", type=str, default="silu", choices=["silu", "gelu", "relu", "tanh"])
    parser.add_argument("--policy_dropout", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "rmsprop"])
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--grad_explosion_threshold", type=float, default=10.0)
    parser.add_argument("--returns_norm_momentum", type=float, default=0.99)
    parser.add_argument("--psnr_norm_aux_weight", type=float, default=0.0)
    parser.add_argument("--critic_loss_type", type=str, default="smooth_l1", choices=["smooth_l1", "mse"])
    parser.add_argument("--huber_beta", type=float, default=0.5)
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2)
    parser.add_argument("--ppo_value_clip_eps", type=float, default=0.2)
    parser.add_argument("--ppo_update_epochs", type=int, default=4)
    parser.add_argument("--episodes_per_update", type=int, default=4)
    parser.add_argument("--target_kl", type=float, default=0.03)
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_PPO_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint_every", type=int, default=25)

    parser.add_argument("--sampling_ratio", type=float, default=0.5)
    parser.add_argument("--sampling_method", type=str, default="hadamard", choices=["hadamard"])
    parser.add_argument("--reward_psnr_weight", type=float, default=0.9)
    parser.add_argument("--reward_ssim_weight", type=float, default=0.1)
    parser.add_argument("--reward_consistency_weight", type=float, default=0.1)
    parser.add_argument("--action_switch_penalty", type=float, default=0.01)
    parser.add_argument("--final_quality_weight", type=float, default=0.05)
    parser.add_argument("--reward_clip", type=float, default=1.0)
    parser.add_argument(
        "--use_ssim_in_reward",
        type=lambda x: str(x).lower() in ["1", "true", "yes", "y"],
        default=True,
    )
    parser.add_argument(
        "--env_verbose",
        type=lambda x: str(x).lower() in ["1", "true", "yes", "y"],
        default=False,
    )

    parser.add_argument("--ddnm_eta", type=float, default=1.0)
    parser.add_argument("--dps_scale", type=float, default=0.0125)
    parser.add_argument("--CG_iters_diffpir", type=int, default=5)
    parser.add_argument("--diffpir_lambda", type=float, default=1.0)
    parser.add_argument("--noise_level_img", type=float, default=0.0)
    parser.add_argument("--diffpir_eta", type=float, default=0.0)
    parser.add_argument("--diffpir_zeta", type=float, default=1.0)

    parser.add_argument(
        "--use_wandb",
        type=lambda x: str(x).lower() in ["1", "true", "yes", "y"],
        default=True,
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default="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",
    )
    parser.add_argument("--project_name", type=str, default="PPO_dinamic_agent")
    parser.add_argument("--run_name", type=str, default="run_training_ppo")
    parser.add_argument("--wandb_entity", type=str, default="")

    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
