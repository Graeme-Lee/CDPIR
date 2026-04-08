from __future__ import annotations

import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Tuple

import torch
from torch.utils.data import DataLoader, RandomSampler

from dataset import MedicalSliceDataset
from models import SiT_models
from train_utils import parse_transport_args
from transport import create_transport
import wandb_utils


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float = 0.9999) -> None:
    """Step the EMA model toward the current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """Enable or disable gradients for all parameters."""
    for param in model.parameters():
        param.requires_grad = flag


def create_logger(log_path: Path) -> logging.Logger:
    """Create a logger that writes to stdout and to a log file."""
    logger = logging.getLogger("sit_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def create_experiment_dir(args: argparse.Namespace) -> Tuple[Path, Path]:
    """Create experiment and checkpoint directories."""
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    experiment_index = len([path for path in results_dir.iterdir() if path.is_dir()])
    model_name = args.model.replace("/", "-")
    experiment_name = (
        f"{experiment_index:03d}-{model_name}-"
        f"{args.path_type}-{args.prediction}-{args.loss_weight}"
    )
    experiment_dir = results_dir / experiment_name
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir, checkpoint_dir


def build_dataloader(args: argparse.Namespace) -> Tuple[MedicalSliceDataset, DataLoader]:
    dataset = MedicalSliceDataset(
        root=args.data_path,
        mat_key=args.mat_key,
        normalize=not args.disable_normalize,
    )
    sampler = RandomSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    return dataset, loader


def build_model_and_optimizer(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.optim.Optimizer]:
    model = SiT_models[args.model](
        input_size=args.image_size,
        num_classes=args.num_classes,
    ).to(device)
    ema = deepcopy(model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    requires_grad(ema, False)
    update_ema(ema, model, decay=0.0)
    return model, ema, optimizer


def maybe_resume(
    args: argparse.Namespace,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[int, int]:
    """Resume from a checkpoint if ``args.ckpt`` is provided."""
    if not args.ckpt:
        return 0, 0

    ckpt_path = Path(args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["opt"])

    start_epoch = int(checkpoint.get("epoch", 0))
    train_steps = int(checkpoint.get("train_steps", 0))
    logger.info(f"Resumed from checkpoint: {ckpt_path}")
    return start_epoch, train_steps


def save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_steps: int,
) -> Path:
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": optimizer.state_dict(),
        "epoch": epoch,
        "train_steps": train_steps,
    }
    checkpoint_path = checkpoint_dir / f"{train_steps:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def train_one_epoch(
    *,
    model: torch.nn.Module,
    ema: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    transport,
    device: torch.device,
    logger: logging.Logger,
    args: argparse.Namespace,
    epoch: int,
    train_steps: int,
) -> int:
    model.train()
    running_loss = 0.0
    log_steps = 0
    start_time = time()

    logger.info(f"Beginning epoch {epoch}...")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        loss_dict = transport.training_losses(model, x, model_kwargs={"y": y})
        loss = loss_dict["loss"].mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        update_ema(ema, model, decay=args.ema_decay)

        train_steps += 1
        log_steps += 1
        running_loss += float(loss.item())

        if train_steps % args.log_every == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            elapsed = max(time() - start_time, 1e-6)
            steps_per_sec = log_steps / elapsed
            avg_loss = running_loss / log_steps
            logger.info(
                f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                f"Train Steps/Sec: {steps_per_sec:.2f}"
            )
            if args.wandb:
                wandb_utils.log(
                    {"train loss": avg_loss, "train steps/sec": steps_per_sec},
                    step=train_steps,
                )
            running_loss = 0.0
            log_steps = 0
            start_time = time()

        if train_steps % args.ckpt_every == 0:
            checkpoint_path = save_checkpoint(
                checkpoint_dir=Path(args.checkpoint_dir),
                model=model,
                ema=ema,
                optimizer=optimizer,
                epoch=epoch,
                train_steps=train_steps,
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    return train_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SiT on .mat/.npy medical slices.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-S/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a training checkpoint.")
    parser.add_argument("--mat-key", type=str, default="sub_label1")
    parser.add_argument("--disable-normalize", action="store_true")
    parse_transport_args(parser)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one CUDA GPU.")

    device = torch.device("cuda:0")
    torch.manual_seed(args.global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.global_seed)

    experiment_dir, checkpoint_dir = create_experiment_dir(args)
    args.checkpoint_dir = str(checkpoint_dir)
    logger = create_logger(experiment_dir / "log.txt")
    logger.info(f"Experiment directory created at {experiment_dir}")

    if args.wandb:
        wandb_utils.initialize(
            args=args,
            entity="haodong",
            experiment=experiment_dir.name,
            project="SVCT-SiT",
        )

    dataset, loader = build_dataloader(args)
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(f"Detected classes: {dataset.class_to_idx}")

    model, ema, optimizer = build_model_and_optimizer(args, device)
    start_epoch, train_steps = maybe_resume(args, model, ema, optimizer, device, logger)

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
    )

    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs):
        train_steps = train_one_epoch(
            model=model,
            ema=ema,
            optimizer=optimizer,
            loader=loader,
            transport=transport,
            device=device,
            logger=logger,
            args=args,
            epoch=epoch,
            train_steps=train_steps,
        )

    model.eval()
    ema.eval()
    logger.info("Done!")


if __name__ == "__main__":
    main(parse_args())
