#!/usr/bin/env python3
"""Train Inf-Dehaze on paired hazy/clear datasets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from inf_dehaze.data import DehazeDataset
from inf_dehaze.model import InfDehaze


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def reduce_moe_loss(moe_loss):
    if isinstance(moe_loss, (list, tuple)):
        return torch.stack(moe_loss).mean()
    if isinstance(moe_loss, torch.Tensor) and moe_loss.dim() > 0:
        return moe_loss.mean()
    return moe_loss


def compute_loss(pred, target, criterion, moe_loss, moe_weight):
    loss = criterion(pred, target)
    if moe_loss is not None:
        loss = loss + moe_weight * reduce_moe_loss(moe_loss)
    return loss


def load_checkpoint(model, checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    elif isinstance(model, nn.parallel.DistributedDataParallel):
        prefixed = {}
        for key, value in state_dict.items():
            prefixed[key if key.startswith("module.") else f"module.{key}"] = value
        model.load_state_dict(prefixed)
    else:
        model.load_state_dict(state_dict)


def save_checkpoint(model, save_path: str):
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, save_path)


def train(args):
    cfg = load_config(Path(args.config))
    train_cfg = cfg.get("training", {})
    dataset_cfg = cfg.get("dataset", {})

    data_dir = args.data_dir or train_cfg.get("data_dir", "./datasets/8KDehaze")
    save_dir = args.save_dir or train_cfg.get("save_dir", "./checkpoints/train")
    os.makedirs(save_dir, exist_ok=True)

    dataset = DehazeDataset(
        root_dir=data_dir,
        crop_size=args.crop_size or dataset_cfg.get("crop_size", 1024),
        normalize=args.normalize or train_cfg.get("normalize", False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = InfDehaze.from_config(cfg).to(device)

    if torch.cuda.device_count() > 1 and not args.no_cuda:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = nn.L1Loss().to(device)
    lr = args.lr or train_cfg.get("lr", 1e-4)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    resume = args.resume or train_cfg.get("resume")
    if resume:
        print(f"Resuming from {resume}")
        load_checkpoint(model, resume)
    else:
        print("Training from scratch")

    epochs = args.epochs or train_cfg.get("epochs", 100)
    save_cycle = args.save_cycle or train_cfg.get("save_cycle", 1)
    moe_weight = train_cfg.get("moe_loss_weight", 0.01)
    log_path = os.path.join(save_dir, "train.log")

    global_step = 0
    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            global_step += 1
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            pred, moe_loss = model(hazy)
            loss = compute_loss(pred, clear, criterion, moe_loss, moe_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            if save_cycle > 0 and global_step % int(save_cycle * len(loader)) == 0:
                save_checkpoint(model, os.path.join(save_dir, f"step_{global_step}.pth"))

        mean_loss = float(np.mean(epoch_losses))
        line = f"Epoch {epoch + 1}/{epochs} | loss={mean_loss:.4f}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    save_checkpoint(model, os.path.join(save_dir, "last.pth"))
    print(f"Training complete. Checkpoints saved to {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Inf-Dehaze")
    parser.add_argument("--config", type=str, default="./configs/dehaze_default.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset root directory")
    parser.add_argument("--save_dir", type=str, default=None, help="Checkpoint output directory")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--crop_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--save_cycle", type=int, default=None, help="Save every N epochs")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
