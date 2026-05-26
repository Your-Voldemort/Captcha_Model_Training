"""Dataset and data loading utilities."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

from charset import CHAR_TO_INDEX, IMG_HEIGHT, IMG_WIDTH


def load_label_rows(labels_path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = row.get("filename", "").strip()
            label = row.get("label", "").strip().upper()
            if filename and label:
                rows.append((filename, label))
    return rows


def encode_label(label: str) -> torch.Tensor:
    return torch.tensor([CHAR_TO_INDEX[char] for char in label], dtype=torch.long)


def decode_indices(indices: list[int]) -> str:
    from charset import INDEX_TO_CHAR

    chars: list[str] = []
    previous = None
    for index in indices:
        if index == 0:
            previous = None
            continue
        if index != previous:
            chars.append(INDEX_TO_CHAR[index])
        previous = index
    return "".join(chars)


class CaptchaDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, str]],
        images_dir: Path,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.images_dir = images_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, filename: str) -> Image.Image:
        path = self.images_dir / filename
        return Image.open(path).convert("L")

    def _augment(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            angle = random.uniform(-4, 4)
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=255)

        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)
            image = ImageEnhance.Brightness(image).enhance(factor)

        if random.random() < 0.3:
            image = ImageOps.invert(image)

        if random.random() < 0.4:
            array = np.array(image, dtype=np.float32)
            noise = np.random.normal(0, 8, array.shape)
            array = np.clip(array + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(array)

        return image

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR)
        array = np.array(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        filename, label = self.samples[index]
        image = self._load_image(filename)
        if self.augment:
            image = self._augment(image)
        tensor = self._preprocess(image)
        encoded = encode_label(label)
        return tensor, encoded, len(label)


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor, int]]):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    targets = torch.cat(labels)
    return images, targets, label_lengths


def split_samples(
    samples: list[tuple[str, str]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    return train_samples, val_samples


def create_dataloaders(
    labels_path: Path,
    images_dir: Path,
    batch_size: int,
    num_workers: int,
    val_ratio: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    samples = load_label_rows(labels_path)
    if not samples:
        raise ValueError(f"No labeled samples found in {labels_path}")

    train_samples, val_samples = split_samples(samples, val_ratio=val_ratio)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        CaptchaDataset(train_samples, images_dir, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        CaptchaDataset(val_samples, images_dir, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader
