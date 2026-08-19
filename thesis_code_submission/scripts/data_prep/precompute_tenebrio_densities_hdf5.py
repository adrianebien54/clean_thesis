#!/usr/bin/env python3

"""Precompute Tenebrio ground-truth density maps as HDF5 files, per resolution variant.

For each (resolution, sigma) pair, this script:
  1. Reads bbox annotations from a single full-resolution COCO-style JSON file.
  2. Scales bbox centers from the original (3088x2076) annotation space down to
     the variant's resolution.
  3. Places sub-pixel impulses on the density grid at the variant's resolution.
  4. Applies a Gaussian blur with the variant's sigma and renormalizes so the
     density map's integral equals the object count.
  5. Writes a single HDF5 file per image at <data-dir>/<variant>/<split>/den/<stem>.h5
     containing one float32 dataset named "density".

Per-variant sigma (in pixel units at that variant's resolution):
    1544x1038 -> 24
    772x519   -> 12
    386x260   -> 6
    193x130   -> 3
    97x65     -> 2
    49x33     -> 1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter


SPLITS = ("train", "val", "test")

# (label, target_width, target_height, sigma_in_pixels)
VARIANTS: list[tuple[str, int, int, float]] = [
    ("1544x1038", 1544, 1038, 24.0),
    ("772x519",    772,  519, 12.0),
    ("386x260",    386,  260,  6.0),
    ("193x130",    193,  130,  3.0),
    ("97x65",       97,   65,  2.0),
    ("49x33",       49,   33,  1.0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute Tenebrio density HDF5 maps per variant.")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("exp/data/Tenebrio"),
        help="Tenebrio dataset root that contains <variant>/<split>/img/ subdirs.",
    )
    p.add_argument(
        "--annotation-file",
        type=Path,
        default=Path("exp/data/TenebrioVision_Original/TenebrioVision_Annotations.json"),
        help="COCO-style Tenebrio annotation JSON at the original full resolution.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5 density files.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel worker processes.",
    )
    return p.parse_args()


def load_annotations(annotation_file: Path) -> tuple[dict[str, tuple[float, list[list[float]]]], int, int]:
    """Return mapping {file_name: (orig_w, orig_h, boxes)} from the COCO file.

    Also returns the (orig_w, orig_h) seen across the dataset. Errors if it's mixed.
    """
    with annotation_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    sizes = set((int(im["width"]), int(im["height"])) for im in data["images"])
    if len(sizes) != 1:
        raise RuntimeError(f"Expected one source resolution in annotations, got {sizes}")
    (orig_w, orig_h) = next(iter(sizes))

    image_id_to_filename: dict[int, str] = {
        int(im["id"]): str(im["file_name"]) for im in data["images"]
    }

    grouped_boxes: dict[str, list[list[float]]] = defaultdict(list)
    for ann in data["annotations"]:
        fname = image_id_to_filename.get(int(ann["image_id"]))
        if fname is None:
            continue
        grouped_boxes[fname].append([float(v) for v in ann["bbox"]])

    return grouped_boxes, orig_w, orig_h


def build_density_map(
    width: int,
    height: int,
    boxes_at_target: list[list[float]],
    sigma: float,
) -> np.ndarray:
    """Bilinear impulses + Gaussian blur + count-preserving renormalization."""
    density = np.zeros((height, width), dtype=np.float32)
    target_count = float(len(boxes_at_target))

    for bx, by, bw, bh in boxes_at_target:
        cx = bx + bw / 2.0
        cy = by + bh / 2.0

        x0 = int(np.floor(cx))
        y0 = int(np.floor(cy))
        x1 = x0 + 1
        y1 = y0 + 1

        wx1 = cx - x0
        wy1 = cy - y0
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1

        if 0 <= x0 < width  and 0 <= y0 < height: density[y0, x0] += wx0 * wy0
        if 0 <= x1 < width  and 0 <= y0 < height: density[y0, x1] += wx1 * wy0
        if 0 <= x0 < width  and 0 <= y1 < height: density[y1, x0] += wx0 * wy1
        if 0 <= x1 < width  and 0 <= y1 < height: density[y1, x1] += wx1 * wy1

    if density.sum() > 0:
        density = gaussian_filter(density, sigma=sigma, mode="reflect")
        density *= target_count / density.sum()

    return density.astype(np.float32, copy=False)


def write_density_h5(output_path: Path, density: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # write to a tmp file first then rename to make the write atomic
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.create_dataset(
            "density",
            data=density,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    tmp.replace(output_path)


def _process_one(args_tuple: tuple) -> str:
    (
        image_file_name,
        boxes_orig,
        orig_w, orig_h,
        variant_w, variant_h,
        sigma,
        out_path,
        overwrite,
    ) = args_tuple

    if out_path.exists() and not overwrite:
        return "skip"

    sx = variant_w / orig_w
    sy = variant_h / orig_h
    scaled = [[bx * sx, by * sy, bw * sx, bh * sy] for bx, by, bw, bh in boxes_orig]

    density = build_density_map(variant_w, variant_h, scaled, sigma)
    write_density_h5(out_path, density)
    return "write"


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    grouped_boxes, orig_w, orig_h = load_annotations(args.annotation_file)
    print(f"Loaded annotations: {len(grouped_boxes)} images at {orig_w}x{orig_h}")

    # Build the full task list across (variant, split, image).
    tasks: list[tuple] = []
    for label, var_w, var_h, sigma in VARIANTS:
        for split in SPLITS:
            img_dir = data_dir / label / split / "img"
            if not img_dir.is_dir():
                continue
            for img_path in sorted(img_dir.glob("*.png")):
                fname = img_path.name
                boxes = grouped_boxes.get(fname)
                if boxes is None:
                    continue
                out_path = data_dir / label / split / "den" / (img_path.stem + ".h5")
                tasks.append((
                    fname, boxes,
                    orig_w, orig_h,
                    var_w, var_h,
                    sigma,
                    out_path,
                    args.overwrite,
                ))

    print(f"Generating {len(tasks)} density maps across {len(VARIANTS)} resolutions "
          f"using {args.workers} worker(s) ...")

    written = skipped = 0
    with multiprocessing.Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_one, tasks, chunksize=10)):
            if result == "write":
                written += 1
            else:
                skipped += 1
            if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
                print(f"  {i + 1}/{len(tasks)}  written={written}  skipped={skipped}")

    print(f"\nDone. Wrote {written}, skipped {skipped}.")


if __name__ == "__main__":
    main()
