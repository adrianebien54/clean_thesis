#!/usr/bin/env python3
"""Resize base train/val/test images into each resolution sub-folder.

Replaces any img/ symlinks under each resolution folder with a real directory
containing images resized to that resolution using LANCZOS resampling.

Usage (from project root)::

    python scripts/resize_splits_to_resolutions.py

Optional flags::

    --data-dir   datasets/Tenebrio   (default)
    --overwrite  re-write images that already exist at the target size
    --workers N  parallel worker processes (default: number of CPU cores)
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path

from PIL import Image


RESOLUTIONS: list[tuple[int, int]] = [
    (1544, 1038),
    (772,  519),
    (386,  260),
    (193,  130),
    (97,   65),
    (49,   33),
]

SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("datasets/Tenebrio"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    return p.parse_args()


def _resize_one(args_tuple: tuple) -> str:
    src_path, dst_path, target_w, target_h, overwrite = args_tuple
    if dst_path.exists() and not overwrite:
        return "skip"
    with Image.open(src_path) as img:
        if img.size == (target_w, target_h) and not overwrite:
            import shutil
            shutil.copy2(src_path, dst_path)
        else:
            resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            resized.save(dst_path)
    return "write"


def replace_symlink_with_dir(path: Path) -> None:
    """If path is a symlink, remove it and create a real directory."""
    if path.is_symlink():
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()

    # Collect all resize tasks
    tasks: list[tuple] = []
    for target_w, target_h in RESOLUTIONS:
        label = f"{target_w}x{target_h}"
        res_dir = data_dir / label
        for split in SPLITS:
            src_img_dir = data_dir / split / "img"
            dst_img_dir = res_dir / split / "img"

            replace_symlink_with_dir(dst_img_dir)

            for src in sorted(src_img_dir.glob("*.png")):
                dst = dst_img_dir / src.name
                tasks.append((src, dst, target_w, target_h, args.overwrite))

    print(f"Resizing {len(tasks)} images across {len(RESOLUTIONS)} resolutions "
          f"using {args.workers} worker(s) ...")

    written = skipped = 0
    with multiprocessing.Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_resize_one, tasks, chunksize=20)):
            if result == "write":
                written += 1
            else:
                skipped += 1
            if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
                print(f"  {i + 1}/{len(tasks)}  written={written}  skipped={skipped}")

    print(f"\nDone. Wrote {written}, skipped {skipped}.")


if __name__ == "__main__":
    main()
