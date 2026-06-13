"""Training and evaluation datasets for Inf-Dehaze."""

from __future__ import annotations

import os
import random
from typing import Callable, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_MEAN = (0.45837133, 0.47633536, 0.44432645)
DEFAULT_STD = (0.16936361, 0.15927625, 0.15468806)


def build_transform(normalize: bool, mean=DEFAULT_MEAN, std=DEFAULT_STD) -> Callable:
    steps = [transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


class DehazeDataset(Dataset):
    """Synthetic paired dataset with multiple haze levels (8KDehaze layout).

    Expected directory structure::

        root/
        ├── clear/
        ├── cloud_0/ ... cloud_4/   # or cloud_L1/ ... cloud_L4/ for big variant
    """

    def __init__(
        self,
        root_dir: str,
        crop_size: int = 1024,
        normalize: bool = False,
        haze_dirs: Optional[Sequence[str]] = None,
        augment: bool = True,
    ):
        self.root_dir = root_dir
        self.clear_dir = os.path.join(root_dir, "clear")
        self.haze_dirs = list(haze_dirs or [f"cloud_{i}" for i in range(5)])
        self.crop_size = crop_size
        self.augment = augment
        self.transform = build_transform(normalize)

        if not os.path.isdir(self.clear_dir):
            raise FileNotFoundError(f"Missing clear image folder: {self.clear_dir}")

        self.image_filenames = sorted(
            name
            for name in os.listdir(self.clear_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not self.image_filenames:
            raise FileNotFoundError(f"No images found in {self.clear_dir}")

    def __len__(self) -> int:
        return len(self.image_filenames)

    def _random_crop(self, hazy: Image.Image, clear: Image.Image):
        width, height = clear.size
        if width < self.crop_size or height < self.crop_size:
            raise ValueError(
                f"Image {clear.size} is smaller than crop_size={self.crop_size}"
            )
        start_x = random.randint(0, width - self.crop_size)
        start_y = random.randint(0, height - self.crop_size)
        box = (start_x, start_y, start_x + self.crop_size, start_y + self.crop_size)
        return hazy.crop(box), clear.crop(box)

    def __getitem__(self, idx: int):
        haze_folder = random.choice(self.haze_dirs)
        haze_path = os.path.join(self.root_dir, haze_folder, self.image_filenames[idx])
        clear_path = os.path.join(self.clear_dir, self.image_filenames[idx])

        hazy = Image.open(haze_path).convert("RGB")
        clear = Image.open(clear_path).convert("RGB")
        if hazy.size != clear.size:
            raise ValueError("Hazy/clear image pair sizes do not match")

        hazy, clear = self._random_crop(hazy, clear)
        if self.augment:
            angle = random.choice([0, 90, 180, 270])
            hazy = hazy.rotate(angle)
            clear = clear.rotate(angle)

        return {
            "hazy": self.transform(hazy),
            "clear": self.transform(clear),
        }


class DehazeDatasetExpanded(DehazeDataset):
    """Deterministically iterate all haze levels for each clear image."""

    def __len__(self) -> int:
        return len(self.image_filenames) * len(self.haze_dirs)

    def __getitem__(self, idx: int):
        orig_idx = idx % len(self.image_filenames)
        haze_folder = self.haze_dirs[idx // len(self.image_filenames)]
        haze_path = os.path.join(self.root_dir, haze_folder, self.image_filenames[orig_idx])
        clear_path = os.path.join(self.clear_dir, self.image_filenames[orig_idx])

        hazy = Image.open(haze_path).convert("RGB")
        clear = Image.open(clear_path).convert("RGB")
        if hazy.size != clear.size:
            raise ValueError("Hazy/clear image pair sizes do not match")

        hazy, clear = self._random_crop(hazy, clear)
        if self.augment:
            angle = random.choice([0, 90, 180, 270])
            hazy = hazy.rotate(angle)
            clear = clear.rotate(angle)

        return {
            "hazy": self.transform(hazy),
            "clear": self.transform(clear),
        }


class DehazeDatasetBig(DehazeDataset):
    """8KDehaze big variant using cloud_L1 ... cloud_L4 folders."""

    def __init__(self, root_dir: str, crop_size: int = 2048, normalize: bool = False):
        super().__init__(
            root_dir=root_dir,
            crop_size=crop_size,
            normalize=normalize,
            haze_dirs=[f"cloud_L{i}" for i in range(1, 5)],
        )


class PairedDehazeDataset(Dataset):
    """Generic paired dataset with explicit hazy/clear subfolders."""

    def __init__(
        self,
        root_dir: str,
        hazy_dir: str,
        clear_dir: str,
        crop_size: int = 1024,
        normalize: bool = False,
        resize: Optional[tuple[int, int]] = None,
        filename_map: Optional[Callable[[str], str]] = None,
    ):
        self.hazy_dir = os.path.join(root_dir, hazy_dir)
        self.clear_dir = os.path.join(root_dir, clear_dir)
        self.crop_size = crop_size
        self.resize = resize
        self.filename_map = filename_map
        self.transform = build_transform(normalize)

        self.image_filenames = sorted(
            name
            for name in os.listdir(self.clear_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int):
        clear_name = self.image_filenames[idx]
        hazy_name = self.filename_map(clear_name) if self.filename_map else clear_name

        hazy = Image.open(os.path.join(self.hazy_dir, hazy_name)).convert("RGB")
        clear = Image.open(os.path.join(self.clear_dir, clear_name)).convert("RGB")
        if self.resize is not None:
            hazy = hazy.resize(self.resize)
            clear = clear.resize(self.resize)
        if hazy.size != clear.size:
            raise ValueError("Hazy/clear image pair sizes do not match")

        width, height = clear.size
        start_x = random.randint(0, width - self.crop_size)
        start_y = random.randint(0, height - self.crop_size)
        box = (start_x, start_y, start_x + self.crop_size, start_y + self.crop_size)
        hazy = hazy.crop(box)
        clear = clear.crop(box)

        angle = random.choice([0, 90, 180, 270])
        hazy = hazy.rotate(angle)
        clear = clear.rotate(angle)

        return {
            "hazy": self.transform(hazy),
            "clear": self.transform(clear),
        }
