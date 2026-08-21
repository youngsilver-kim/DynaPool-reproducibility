#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile

from tqdm import tqdm


URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"


class DownloadProgressBar(tqdm):
    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int | None = None) -> None:
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="Tiny-ImageNet") as progress:
        urllib.request.urlretrieve(url, partial, reporthook=progress.update_to)
    os.replace(partial, destination)


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
        zipped.extractall(destination)


def format_validation(dataset_root: Path) -> int:
    val_dir = dataset_root / "val"
    source_dir = val_dir / "images"
    annotation_path = val_dir / "val_annotations.txt"
    target_dir = val_dir / "images_by_class"
    if not source_dir.is_dir() or not annotation_path.is_file():
        raise FileNotFoundError("Tiny-ImageNet validation images or annotations are missing")
    target_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    with annotation_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 2:
                continue
            image_name, class_id = fields[0], fields[1]
            source = source_dir / image_name
            class_dir = target_dir / class_id
            class_dir.mkdir(parents=True, exist_ok=True)
            target = class_dir / image_name
            if target.exists():
                continue
            try:
                os.link(source, target)  # no duplicate storage on Colab's local filesystem
            except OSError:
                shutil.copy2(source, target)
            created += 1
    total = sum(1 for path in target_dir.rglob("*.JPEG") if path.is_file())
    if total != 10_000:
        raise RuntimeError(f"Expected 10,000 formatted validation images, found {total}")
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description="Idempotent Tiny-ImageNet downloader and formatter")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--delete-archive", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    archive = data_dir / "tiny-imagenet-200.zip"
    dataset_root = data_dir / "tiny-imagenet-200"
    required = [
        dataset_root / "wnids.txt",
        dataset_root / "train",
        dataset_root / "val" / "images",
        dataset_root / "val" / "val_annotations.txt",
    ]
    dataset_complete = all(path.exists() for path in required)
    if not archive.exists() and not dataset_complete:
        download(URL, archive)
    if not dataset_complete:
        safe_extract(archive, data_dir)
    created = format_validation(dataset_root)
    print(f"Validation layout ready ({created} new links/copies): {dataset_root / 'val' / 'images_by_class'}")
    if args.delete_archive and archive.exists():
        archive.unlink()
        print(f"Deleted archive after verified extraction: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
