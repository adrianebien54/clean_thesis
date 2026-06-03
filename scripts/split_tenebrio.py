#!/usr/bin/env python3

"""Split TenebrioVision_Images into train/val/test folders.

The level is encoded by the first number before the underscore in each image
filename. Each level contains 80 images, and this script splits them into:

* 60 train images
* 12 validation images
* 8 test images

The script copies images into:

datasets/Tenebrio/
├── train/
│   ├── img/
│   └── den/
├── val/
│   ├── img/
│   └── den/
└── test/
    ├── img/
    └── den/

Density-map directories are created empty so they can be filled later.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


TRAIN_PER_LEVEL = 60
VAL_PER_LEVEL = 12
TEST_PER_LEVEL = 8
EXPECTED_PER_LEVEL = TRAIN_PER_LEVEL + VAL_PER_LEVEL + TEST_PER_LEVEL

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
SPLITS = {
    "train": TRAIN_PER_LEVEL,
    "val": VAL_PER_LEVEL,
    "test": TEST_PER_LEVEL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Tenebrio images by level.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("TenebrioVision_Images"),
        help="Directory containing the source Tenebrio images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/Tenebrio"),
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle images within each level.",
    )
    return parser.parse_args()


def collect_images(source_dir: Path) -> dict[str, list[Path]]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    grouped: dict[str, list[Path]] = defaultdict(list)
    for image_path in sorted(source_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        level = image_path.stem.split("_", 1)[0]
        if not level.isdigit():
            raise ValueError(f"Cannot parse level from filename: {image_path.name}")

        grouped[level].append(image_path)

    if not grouped:
        raise ValueError(f"No images found in {source_dir}")

    return dict(grouped)


def prepare_output_dirs(output_dir: Path) -> None:
    for split_name in SPLITS:
        (output_dir / split_name / "img").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "den").mkdir(parents=True, exist_ok=True)


def split_images(grouped_images: dict[str, list[Path]], seed: int) -> dict[str, dict[str, list[Path]]]:
    rng = random.Random(seed)
    split_map: dict[str, dict[str, list[Path]]] = {
        split_name: defaultdict(list) for split_name in SPLITS
    }

    for level in sorted(grouped_images, key=lambda item: int(item)):
        images = list(grouped_images[level])
        if len(images) != EXPECTED_PER_LEVEL:
            raise ValueError(
                f"Level {level} has {len(images)} images, expected {EXPECTED_PER_LEVEL}."
            )

        rng.shuffle(images)
        offset = 0
        for split_name, split_size in SPLITS.items():
            split_images_for_level = images[offset : offset + split_size]
            split_map[split_name][level].extend(split_images_for_level)
            offset += split_size

    return split_map


def copy_split_images(split_map: dict[str, dict[str, list[Path]]], output_dir: Path) -> None:
    for split_name, level_map in split_map.items():
        destination_dir = output_dir / split_name / "img"
        for level_images in level_map.values():
            for source_image in level_images:
                shutil.copy2(source_image, destination_dir / source_image.name)


def validate_split_counts(split_map: dict[str, dict[str, list[Path]]]) -> None:
    for split_name, expected_per_level in SPLITS.items():
        for level, level_images in split_map[split_name].items():
            if len(level_images) != expected_per_level:
                raise ValueError(
                    f"Split {split_name} level {level} has {len(level_images)} images, "
                    f"expected {expected_per_level}."
                )


def main() -> None:
    args = parse_args()
    grouped_images = collect_images(args.source)

    for level, images in sorted(grouped_images.items(), key=lambda item: int(item[0])):
        if len(images) != EXPECTED_PER_LEVEL:
            raise ValueError(f"Level {level} has {len(images)} images, expected {EXPECTED_PER_LEVEL}.")

    prepare_output_dirs(args.output)
    split_map = split_images(grouped_images, args.seed)
    validate_split_counts(split_map)
    copy_split_images(split_map, args.output)

    total_train = sum(len(images) for images in split_map["train"].values())
    total_val = sum(len(images) for images in split_map["val"].values())
    total_test = sum(len(images) for images in split_map["test"].values())

    print(f"Processed {len(grouped_images)} levels from {args.source}")
    print(f"train: {total_train} images")
    print(f"val:   {total_val} images")
    print(f"test:  {total_test} images")
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()