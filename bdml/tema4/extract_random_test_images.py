from __future__ import annotations

import argparse
import random
import warnings
from pathlib import Path

import numpy as np

try:
    from torchvision import datasets
except ImportError as exc:  # pragma: no cover - runtime dependency message
    raise ImportError(
        "torchvision is required to extract CIFAR-10 test images"
    ) from exc

try:
    from .constants import DEFAULT_DATA_DIR
except ImportError:
    from constants import DEFAULT_DATA_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract random CIFAR-10 test images into a folder"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).with_name("test")
    )
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.set_defaults(download=True)
    return parser.parse_args()


def extract_random_test_images(
    data_dir: Path,
    output_dir: Path,
    count: int = 25,
    seed: int = 42,
    download: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
    )

    if count <= 0:
        raise ValueError("count must be greater than zero")

    sample_count = min(count, len(dataset))
    rng = random.Random(seed)
    selected_indices = rng.sample(range(len(dataset)), sample_count)
    saved_paths: list[Path] = []

    for index in selected_indices:
        image, label = dataset[index]
        class_name = dataset.classes[label]
        file_name = f"{index:05d}_{class_name}.png"
        file_path = output_dir / file_name
        image.save(file_path)
        saved_paths.append(file_path)

    return saved_paths


def main() -> None:
    args = parse_args()
    saved_paths = extract_random_test_images(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
        download=args.download,
    )

    print(f"Saved {len(saved_paths)} images to {args.output_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
