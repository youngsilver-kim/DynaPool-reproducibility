from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import FakeData, ImageFolder

from .config import ExperimentConfig


@dataclass
class DatasetBundle:
    train: Dataset
    train_eval: Dataset
    validation: Dataset
    class_names: list[str]


def _seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_datasets(config: ExperimentConfig) -> DatasetBundle:
    mean, std = config.normalization_mean, config.normalization_std
    train_transform = T.Compose(
        [
            T.RandomCrop(config.image_size, padding=config.crop_padding, padding_mode="reflect"),
            T.RandomHorizontalFlip(p=config.horizontal_flip_probability),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    eval_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    if config.fake_data:
        common = {"image_size": (3, config.image_size, config.image_size), "num_classes": config.num_classes}
        train = FakeData(size=config.fake_train_size, transform=train_transform, random_offset=0, **common)
        train_eval = FakeData(size=config.fake_train_size, transform=eval_transform, random_offset=0, **common)
        validation = FakeData(
            size=config.fake_val_size, transform=eval_transform, random_offset=config.fake_train_size, **common
        )
        class_names = [f"class_{idx:03d}" for idx in range(config.num_classes)]
        return DatasetBundle(train, train_eval, validation, class_names)

    root = Path(config.data_root)
    train_dir = root / "train"
    val_dir = root / "val" / "images_by_class"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {train_dir}. Run download_data.py first.")
    if not val_dir.is_dir():
        raise FileNotFoundError(
            f"Formatted validation directory not found: {val_dir}. Run download_data.py first."
        )
    train = ImageFolder(train_dir, transform=train_transform)
    train_eval = ImageFolder(train_dir, transform=eval_transform)
    validation = ImageFolder(val_dir, transform=eval_transform)
    if train.classes != validation.classes:
        raise RuntimeError("Training and validation class-to-index mappings differ")
    return DatasetBundle(train, train_eval, validation, list(train.classes))


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
        worker_init_fn=partial(_seed_worker, base_seed=seed),
        generator=generator,
        drop_last=False,
    )


def train_loader_for_epoch(
    bundle: DatasetBundle, config: ExperimentConfig, run_seed: int, epoch: int, pin_memory: bool
) -> DataLoader:
    # The epoch-derived seed makes data order and augmentation reproducible after
    # a runtime disconnect and identical across methods for paired comparisons.
    loader_seed = run_seed * 100_003 + epoch
    return make_loader(
        bundle.train,
        config.batch_size,
        config.num_workers,
        loader_seed,
        shuffle=True,
        pin_memory=pin_memory,
    )


def evaluation_loader(
    dataset: Dataset,
    config: ExperimentConfig,
    seed: int,
    pin_memory: bool,
    batch_size: int | None = None,
) -> DataLoader:
    return make_loader(
        dataset,
        batch_size or config.eval_batch_size,
        config.num_workers,
        seed,
        shuffle=False,
        pin_memory=pin_memory,
    )
