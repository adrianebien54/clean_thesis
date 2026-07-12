#!/usr/bin/env python3
"""Build the offline CSRNet-paper patch dataset from the 386x260 Tenebrio train split.

CSRNet (Li et al., CVPR 2018) augmentation: "We crop 9 patches from each image at
different locations with 1/4 size of the original image. The first four patches contain
four quarters of the image without overlapping while the other five patches are randomly
cropped from the input image. After that, we mirror the patches so that we double the
training set."

Per 386x260 train image this writes 18 patches of exactly 193x130 (1/4 area):
  _q0.._q3   four non-overlapping quarters, origins (0,0),(193,0),(0,130),(193,130)
  _r0.._r4   five random crops, per-stem seeded RNG (fixed forever, order-independent)
  _*m        horizontal mirror of each of the nine

840 train images -> 15,120 patches under exp/data/Tenebrio/386x260_csrnet9/train/{img,den}.
Density .h5 (key 'density', float32) are cropped/flipped identically, so patch sums are
exact local counts. val/ and test/ are relative symlinks to the original 386x260 splits,
so the loader's DATA_PATH+'/val' convention keeps evaluating on full images.

The random crops are re-derived from rng([3035, crc32(stem)]) so regeneration (--force)
is bit-identical. Refuses to run if the output dir already exists (use --force).

Built-in verification (exit 1 on failure):
  - 15,120 PNGs and 15,120 .h5 files
  - per stem: |sum(q0..q3) - sum(full)| < 1e-3   (quarters tile the image exactly)
  - every mirrored patch sum == unmirrored sum
  - density dtype float32
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zlib
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path("/home/umrobotics/clean_thesis/crowd_counting")

SRC = ROOT / "exp/data/Tenebrio/386x260"
DST = ROOT / "exp/data/Tenebrio/386x260_csrnet9"

W, H = 386, 260          # source image size
PW, PH = 193, 130        # patch size = 1/4 area (half width x half height)
SEED = 3035
QUARTERS = [(0, 0), (PW, 0), (0, PH), (PW, PH)]  # (x, y) origins, tile exactly
N_RANDOM = 5


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


def patch_specs(stem: str) -> list[tuple[str, int, int]]:
    """(suffix, x, y) for the 9 unmirrored patches of one source image."""
    specs = [(f"q{i}", x, y) for i, (x, y) in enumerate(QUARTERS)]
    rng = np.random.default_rng([SEED, zlib.crc32(stem.encode())])
    for i in range(N_RANDOM):
        x = int(rng.integers(0, W - PW + 1))
        y = int(rng.integers(0, H - PH + 1))
        specs.append((f"r{i}", x, y))
    return specs


def process_stem(stem: str, img_dir: Path, den_dir: Path) -> dict:
    with Image.open(SRC / "train/img" / f"{stem}.png") as im:
        img = im.convert("RGB") if im.mode != "RGB" else im.copy()
    if img.size != (W, H):
        raise ValueError(f"{stem}: image is {img.size}, expected {(W, H)}")
    with h5py.File(SRC / "train/den" / f"{stem}.h5", "r") as f:
        den = f["density"][:].astype(np.float32, copy=False)
    if den.shape != (H, W):
        raise ValueError(f"{stem}: density is {den.shape}, expected {(H, W)}")

    quarter_sum = 0.0
    mirror_ok = True
    for sfx, x, y in patch_specs(stem):
        img_p = img.crop((x, y, x + PW, y + PH))
        den_p = den[y:y + PH, x:x + PW]
        if sfx.startswith("q"):
            quarter_sum += float(den_p.sum(dtype=np.float64))

        img_p.save(img_dir / f"{stem}_{sfx}.png")
        write_density_h5(den_dir / f"{stem}_{sfx}.h5", den_p)

        den_m = np.flip(den_p, axis=1).copy()  # np.flip returns a negative-stride view
        img_p.transpose(Image.FLIP_LEFT_RIGHT).save(img_dir / f"{stem}_{sfx}m.png")
        write_density_h5(den_dir / f"{stem}_{sfx}m.h5", den_m)
        # float64 accumulation: flipping reorders the float32 pairwise sum
        mirror_ok &= abs(float(den_m.sum(dtype=np.float64))
                         - float(den_p.sum(dtype=np.float64))) < 1e-6

    return {
        "full_sum": float(den.sum(dtype=np.float64)),
        "quarter_sum": quarter_sum,
        "mirror_ok": mirror_ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true",
                    help="delete and regenerate an existing output dir")
    args = ap.parse_args()

    if DST.exists():
        if not args.force:
            print(f"refusing to overwrite existing {DST} (use --force)", file=sys.stderr)
            return 1
        shutil.rmtree(DST)

    stems = sorted(p.stem for p in (SRC / "train/img").glob("*.png"))
    if len(stems) != 840:
        print(f"expected 840 train images in {SRC}/train/img, found {len(stems)}",
              file=sys.stderr)
        return 1

    img_dir = DST / "train/img"
    den_dir = DST / "train/den"
    img_dir.mkdir(parents=True)
    den_dir.mkdir(parents=True)
    for split in ("val", "test"):
        os.symlink(f"../386x260/{split}", DST / split)

    failures = []
    for i, stem in enumerate(stems, 1):
        r = process_stem(stem, img_dir, den_dir)
        if abs(r["quarter_sum"] - r["full_sum"]) >= 1e-3:
            failures.append(f"{stem}: quarters sum {r['quarter_sum']:.6f} != "
                            f"full {r['full_sum']:.6f}")
        if not r["mirror_ok"]:
            failures.append(f"{stem}: mirrored density sum mismatch")
        if i % 100 == 0 or i == len(stems):
            print(f"  {i}/{len(stems)} images -> {i * 18} patches")

    n_png = len(list(img_dir.glob("*.png")))
    n_h5 = len(list(den_dir.glob("*.h5")))
    if n_png != 15120 or n_h5 != 15120:
        failures.append(f"file counts: {n_png} PNGs / {n_h5} h5 (expected 15120 each)")
    with h5py.File(den_dir / f"{stems[0]}_q0.h5", "r") as f:
        if f["density"].dtype != np.float32:
            failures.append(f"density dtype {f['density'].dtype}, expected float32")
    for split in ("val", "test"):
        if not (DST / split / "img").is_dir():
            failures.append(f"{split} symlink does not resolve")

    if failures:
        print("\nVERIFICATION FAILED:", file=sys.stderr)
        for msg in failures[:20]:
            print(f"  {msg}", file=sys.stderr)
        return 1
    print(f"\nOK: {n_png} patches + densities in {DST}, quarter-tiling count conservation "
          f"and mirror sums verified; val/test symlinked to ../386x260.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
