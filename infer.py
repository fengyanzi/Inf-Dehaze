#!/usr/bin/env python3
"""Run memory-efficient dehazing inference on a single image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

from inf_dehaze.model import InfDehaze

DEFAULT_MEAN = (0.45837133, 0.47633536, 0.44432645)
DEFAULT_STD = (0.16936361, 0.15927625, 0.15468806)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pad_to_crop_multiple(image: Image.Image, crop_size: int) -> tuple[Image.Image, int, int]:
    width, height = image.size
    pad_w = (crop_size - width % crop_size) % crop_size
    pad_h = (crop_size - height % crop_size) % crop_size
    if pad_w == 0 and pad_h == 0:
        return image, width, height

    padded = Image.new("RGB", (width + pad_w, height + pad_h))
    padded.paste(image, (0, 0))
    return padded, width, height


def build_input_tensor(image: Image.Image, normalize: bool, fp16: bool, device: torch.device):
    transform = [transforms.ToTensor()]
    if normalize:
        transform.append(transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD))
    tensor = transforms.Compose(transform)(image).unsqueeze(0).to(device)
    if fp16:
        tensor = tensor.half()
    return tensor


def denormalize_tensor(tensor: torch.Tensor, normalize: bool) -> torch.Tensor:
    if not normalize:
        return tensor.clamp(0, 1)
    mean = torch.tensor(DEFAULT_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(DEFAULT_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


@torch.inference_mode()
def infer(args):
    cfg = load_config(Path(args.config))
    infer_cfg = cfg.get("inference", {})
    crop_size = cfg.get("model", {}).get("crop_size", 256)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = InfDehaze.from_config(cfg)
    model.inference_batch_size = args.inference_batch_size or infer_cfg.get("batch_size", 4)
    model.use_residual_cache = not args.no_cache
    model.async_io = not args.sync_io

    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    if args.fp16 and device.type == "cuda":
        model = model.half()
    model.eval()

    image = Image.open(args.input).convert("RGB")
    if args.resize:
        image = image.resize((args.resize[0], args.resize[1]), Image.BICUBIC)

    padded, orig_w, orig_h = pad_to_crop_multiple(image, crop_size)
    tensor = build_input_tensor(padded, args.normalize, args.fp16 and device.type == "cuda", device)

    output = model(tensor)
    output = denormalize_tensor(output.float(), args.normalize)
    output = output[:, :, :orig_h, :orig_w]

    os.makedirs(args.output_dir, exist_ok=True)
    out_name = Path(args.input).stem + "_dehazed.png"
    out_path = os.path.join(args.output_dir, out_name)
    transforms.ToPILImage()(output.squeeze(0).cpu()).save(out_path)
    print(f"Saved result to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inf-Dehaze inference")
    parser.add_argument("--config", type=str, default="./configs/dehaze_default.yaml")
    parser.add_argument("--input", type=str, required=True, help="Path to hazy input image")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--inference_batch_size", type=int, default=None)
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=None)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-cache", action="store_true", help="Disable CPU residual caching")
    parser.add_argument("--sync-io", action="store_true", help="Disable async CPU transfer")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    infer(parse_args())
