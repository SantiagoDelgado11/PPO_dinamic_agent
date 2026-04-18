from utils.utils import save_images, save_metrics, AverageMeter
import wandb
from torchvision import transforms
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import logging
from utils.ddpm import Diffusion
from utils.utils import (
    save_checkpoint,
    load_checkpoint,
    cleanup_old_checkpoints,
)
from guided_diffusion.script_util import create_model
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def train(args):

    path_name = f"e_{args.epochs}_bs_{args.batch_size}_lr_{args.lr}_seed_{args.seed}_img_{args.image_size}_schedule_{args.schedule_name}_gpu_{args.gpu_id}_c_{args.channels}_si_{args.save_img}"
    args.save_path = os.path.join(args.save_path, path_name)
    images_path, model_path, metrics_path = save_metrics(args.save_path)

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.save_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{metrics_path}/training.log",
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
    )

    logging.info(f"Starting training with parameters: {args}")

    wandb.login(key="b879bf20f3c31bfcf13289e363f4d3394f7d7671")
    wandb.init(project=args.project_name, name=path_name, config=args)

    device = args.device

    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = create_model(image_size=args.image_size, num_channels=64, num_res_blocks=3, input_channels=args.channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(
        device=device,
        img_size=args.image_size,
        schedule_name=args.schedule_name,
        channels=args.channels,
    )

    for epoch in range(args.epochs):
        train_loss = AverageMeter()
        data_loop_train = tqdm(enumerate(train_loader), total=len(train_loader), colour="red")

        for _, train_data in data_loop_train:
            images = train_data[0]
            images = images.to(device)

            images = images * 2 - 1

            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item())
            data_loop_train.set_postfix(loss=train_loss.avg)

        logging.info(f"Epoch {epoch} loss: {train_loss.avg}")

        checkpoint_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "args": vars(args),
        }
        latest_checkpoint = os.path.join(checkpoint_dir, "latest.pth.tar")
        save_checkpoint(checkpoint_state, latest_checkpoint)

        sampled_image_for_log = None
        if (epoch) % args.save_img == 0:
            sampled_images = diffusion.sample(model, n=1)
            sampled_grid = save_images(sampled_images, f"{images_path}/epoch_{epoch}_sampled.png")
            sampled_image_for_log = wandb.Image(sampled_grid)

        wandb.log(
            {
                "epoch": epoch,
                "sampled_images": sampled_image_for_log,
                "train_loss": train_loss.avg,
            }
        )


def launch():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_path", type=str, default="weights/")
    parser.add_argument("--save_img", type=int, default=100, help="Save images every N epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--project_name", type=str, default="STSIVA AGENTS")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training (e.g., 'latest.pth.tar')",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 5)",
    )
    parser.add_argument(
        "--keep_checkpoints",
        type=int,
        default=3,
        help="Number of recent checkpoints to keep (default: 3)",
    )
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument(
        "--schedule_name",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="Noise schedule name (default: 'linear')",
    )

    parser.add_argument("--gpu_id", type=str, default="0")

    args = parser.parse_args()
    args.device = f"cuda:{args.gpu_id}"
    train(args)


if __name__ == "__main__":
    launch()
