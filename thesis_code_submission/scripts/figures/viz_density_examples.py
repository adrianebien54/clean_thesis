#!/usr/bin/env python3
"""Qualitative density-map figures for the two example plates shown in the thesis.

The 20-vs-80 example figure (images/tenebriovision_20_vs_80.png) is a pixel-exact
crop of two dataset images:
    20 larvae -> val/20_70.png
    80 larvae -> train/80_1.png
This script re-runs both trained nets on those two exact images at three of the six
sweep resolutions -- the smallest (49x33), the resolution the hyper-parameters were
tuned at (386x260), and the largest (1544x1038) -- and renders input / ground-truth
density / CSRNet prediction / MobileCount prediction as one 4x3 grid per image.

Checkpoints are the same best-val-MAE ones the resolution study reports on test
(exp/combined_resolution/testset_results.json), so the counts printed in the panels
are produced by exactly the networks behind the resolution curve. Inference replicates
trainer.validate_V1: pad to a multiple of 8, ImageNet normalisation, pred_cnt =
sum(pred) / LOG_PARA.

Density values are plotted as *count per unit relative image area* (cell value x number
of cells), which is invariant to the grid the map lives on; that makes the colour scale
comparable across resolutions, and every panel in a figure shares one vmax and one
colour bar. GT sigma scales with resolution (24/12/6/3/2/1 px), so the ground-truth
blobs cover the same fraction of the plate at every size.

Writes exp/combined_resolution/density_examples/density_example_{20_70,80_1}.png/.pdf.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image, ImageOps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import config
config.cfg.GPU_ID = [0]

from datasets.Tenebrio.setting import cfg_data
from models.CC import CrowdCounter

TESTSET_JSON = ROOT / "exp/combined_resolution/testset_results.json"
OUT_DIR = ROOT / "exp/combined_resolution/density_examples"

# (stem, split, nominal count).
# The first two are the panels of images/tenebriovision_20_vs_80.png exactly as that
# figure stands; neither is held out (20_70 is val, which selected the checkpoints, and
# 80_1 is train outright), so they illustrate the maps but carry no accuracy claim.
# 40_6 is a held-out TEST image at the dataset's mid count. It was picked as the medoid
# of the eight 40-larvae test images by normalised cross-correlation (mean ncc 0.602 to
# the other seven, the highest of the eight), i.e. the most typical-looking of them
# rather than a hand-picked best or worst case.
# 10_25 is the held-out TEST image on which the two nets' density maps are shaped most
# differently: it tops the map-distance ranking at 49x33 (TV distance 0.700, split max)
# and is also among the largest count gaps there (CSRNet 16.50, MobileCount 5.12, GT 10).
# 80_18 is the held-out TEST stand-in for 80_1. It is both the medoid of the eight
# 80-larvae test images (mean ncc 0.603 to the other seven, the highest) and the closest
# visual match to 80_1 itself (ncc 0.576), so it replaces that train-split panel without
# changing what the figure depicts. Note 80_17 is a near-duplicate frame of it (ncc
# 0.997) -- do not use both.
# See scripts/rank_net_disagreement.py.
EXAMPLES = [
    ("20_70", "val", 20),
    ("80_1", "train", 80),
    ("40_6", "test", 40),
    ("10_25", "test", 10),
    ("80_18", "test", 80),
]

# (variant, column title) smallest -> tuning -> largest
VARIANTS = [
    ("49x33", "49$\\times$33\n(smallest)"),
    ("386x260", "386$\\times$260\n(tuning resolution)"),
    ("1544x1038", "1544$\\times$1038\n(largest)"),
]

# every resolution in the sweep, used with --all
ALL_VARIANTS = [
    ("49x33", "49$\\times$33\n(smallest)"),
    ("97x65", "97$\\times$65"),
    ("193x130", "193$\\times$130"),
    ("386x260", "386$\\times$260\n(tuning resolution)"),
    ("772x519", "772$\\times$519"),
    ("1544x1038", "1544$\\times$1038\n(largest)"),
]

NETS = ["CSRNet", "MobileCount"]
CMAP = "inferno"

# --present is sized for a printed A4 portrait sheet. One font size is used for every
# label on the figure -- column headers, resolution labels and the colour-bar label --
# so nothing competes with the panels themselves.
A4_PORTRAIT = (8.27, 11.69)
PRESENT_FS = 12
PLATE_ASPECT = 1544 / 1038  # every variant is this shape


def load_specs() -> dict:
    """{(net, variant): {checkpoint, log_para, val_mae, test_mae}} from the test eval."""
    rows = json.load(open(TESTSET_JSON))
    return {(r["net"], r["variant"]): r for r in rows}


def load_image_and_gt(variant: str, split: str, stem: str):
    """Image (RGB, unpadded), padded tensor-ready image, GT density (unpadded)."""
    img = Image.open(ROOT / f"exp/data/Tenebrio/{variant}/{split}/img/{stem}.png")
    if img.mode == "L":
        img = img.convert("RGB")
    img_w, img_h = img.size

    with h5py.File(ROOT / f"exp/data/Tenebrio/{variant}/{split}/den/{stem}.h5", "r") as f:
        den = f["density"][:].astype(np.float32, copy=False)

    # same padding rule as datasets/Tenebrio/Tenebrio.py
    pad_w = (8 - img_w % 8) % 8
    pad_h = (8 - img_h % 8) % 8
    padded = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0) if (pad_w or pad_h) else img
    return img, padded, den, (img_w, img_h)


@torch.no_grad()
def predict(net_name: str, ckpt: Path, padded_img: Image.Image, size: tuple[int, int],
            log_para: float) -> tuple[np.ndarray, float]:
    """Density map cropped back to the unpadded size, in raw (count) units, + count."""
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD),
    ])
    x = img_transform(padded_img).unsqueeze(0).cuda()

    net = CrowdCounter(config.cfg.GPU_ID, net_name)
    net.load_state_dict(torch.load(str(ckpt), map_location="cuda", weights_only=True))
    net.cuda()
    net.eval()
    pred = net.test_forward(x)
    count = float(pred.sum().item()) / log_para
    pred = pred.squeeze().cpu().numpy() / log_para

    del net, x
    gc.collect()
    torch.cuda.empty_cache()

    w, h = size
    return pred[:h, :w], count


def to_relative_density(m: np.ndarray) -> np.ndarray:
    """Count per unit relative image area -- comparable across grid sizes."""
    return m * m.size


def build_figure(stem: str, split: str, nominal: int, specs: dict,
                 variants: list | None = None, suffix: str = "",
                 present: bool = False) -> None:
    variants = variants or VARIANTS
    cols = len(variants)
    rows = 2 + len(NETS)

    panels: dict[tuple[int, int], np.ndarray] = {}
    counts: dict[tuple[int, int], float] = {}
    inputs: dict[int, Image.Image] = {}
    gt_counts: dict[int, float] = {}

    for j, (variant, _) in enumerate(variants):
        img, padded, den, size = load_image_and_gt(variant, split, stem)
        inputs[j] = img
        gt_counts[j] = float(den.sum())
        panels[(1, j)] = to_relative_density(den)

        for i, net_name in enumerate(NETS):
            spec = specs[(net_name, variant)]
            pred, count = predict(net_name, ROOT / spec["checkpoint"], padded, size,
                                  float(spec["log_para"]))
            panels[(2 + i, j)] = to_relative_density(pred)
            counts[(2 + i, j)] = count
            print(f"  {stem:<6} {variant:<10} {net_name:<12} pred={count:7.2f} "
                  f"gt={gt_counts[j]:6.2f}  err={count - gt_counts[j]:+6.2f}")

    # one colour scale for every density panel in this figure
    vmax = float(np.percentile(np.concatenate([p.ravel() for p in panels.values()]), 99.9))

    # Row/column roles swap between the two layouts. Default: resolutions across the
    # columns, panel kinds down the rows. Presentation: transposed, so each resolution
    # is a row read left-to-right through input -> GT -> the two nets.
    kind_labels = ["input", "ground truth"] + NETS
    res_labels = [(v[0].replace("x", "$\\times$") if present else v[1]) for v in variants]

    if present:
        nrow, ncol = len(variants), rows
        figsize = A4_PORTRAIT
        title_fs = rowlab_fs = panel_fs = cb_fs = PRESENT_FS
    else:
        nrow, ncol = rows, len(variants)
        figsize = (4.1 * ncol, 2.95 * nrow)
        title_fs, rowlab_fs, panel_fs, cb_fs = 13, 12, 11, 11

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)

    def cell(kind: int, j: int):
        """kind 0=input, 1=GT, 2..=nets; j=resolution index."""
        return axes[j][kind] if present else axes[kind][j]

    for j, (variant, _) in enumerate(variants):
        for kind in range(rows):
            ax = cell(kind, j)
            if kind == 0:
                ax.imshow(inputs[j], interpolation="nearest")
                label = None
            else:
                im = ax.imshow(panels[(kind, j)], cmap=CMAP, vmin=0, vmax=vmax,
                               interpolation="nearest")
                if kind == 1:
                    label = None if present else f"GT count {gt_counts[j]:.0f}"
                elif present:
                    label = None
                else:
                    c = counts[(kind, j)]
                    label = f"pred {c:.2f}   err {c - gt_counts[j]:+.2f}"
            ax.set_xticks([]); ax.set_yticks([])

            # headers on the first row, names on the first column, whichever axis they
            # fall on in this layout
            if present:
                if j == 0:
                    ax.set_title(kind_labels[kind], fontsize=title_fs,
                                 fontweight="bold", pad=6)
                if kind == 0:
                    ax.set_ylabel(res_labels[j], fontsize=rowlab_fs, fontweight="bold",
                                  rotation=90, va="center", labelpad=6)
            else:
                if kind == 0:
                    ax.set_title(res_labels[j], fontsize=title_fs, pad=8)
                if j == 0:
                    ax.set_ylabel(kind_labels[kind], fontsize=rowlab_fs)

            if label and present:
                ax.set_xlabel(label, fontsize=panel_fs, fontweight="bold", labelpad=6)
            elif label:
                ax.text(0.02, 0.97, label, transform=ax.transAxes, va="top", ha="left",
                        fontsize=panel_fs, color="white",
                        bbox=dict(facecolor="black", alpha=0.55, edgecolor="none",
                                  boxstyle="round,pad=0.25"))

    if present:
        # the gutters have to hold a horizontal "1544x1038" on the left and a rotated
        # colour-bar label on the right, neither of which subplots_adjust accounts for
        # The panels are width-limited: four plates of PLATE_ASPECT side by side is a
        # near-square block, so on a portrait page the widest they can be is whatever
        # the page width allows. Work out the height that block then needs and centre it
        # vertically, rather than letting subplots_adjust stretch the axes boxes and
        # leave the images floating in the middle of them.
        left, right, wspace, hspace = 0.05, 0.875, 0.015, 0.03
        header_frac = 0.022  # room above the top row for the column headers
        panel_w = (right - left) * figsize[0] / (ncol + (ncol - 1) * wspace)
        panel_h = panel_w / PLATE_ASPECT
        grid_h = (nrow + (nrow - 1) * hspace) * panel_h / figsize[1]
        bottom = max(0.01, (1.0 - grid_h - header_frac) / 2)
        top = min(0.985, bottom + grid_h)
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                            wspace=wspace, hspace=hspace)
        cax = fig.add_axes([0.895, bottom, 0.016, top - bottom])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("density", fontsize=cb_fs, fontweight="bold")
        cb.ax.tick_params(labelsize=cb_fs - 2)
    else:
        fig.suptitle(
            f"{stem}.png ({split} split, {nominal} larvae) — density maps across resolution",
            fontsize=15, y=0.985)
        fig.subplots_adjust(left=0.045, right=0.9, top=0.9, bottom=0.03,
                            wspace=0.04, hspace=0.06)
        cax = fig.add_axes([0.915, 0.03, 0.014, 0.87])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("density (larvae per unit relative image area)", fontsize=cb_fs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"density_example_{stem}{suffix}.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None)
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    specs = load_specs()
    args = sys.argv[1:]
    all_res = "--all" in args
    # --present: transposed grid (one row per resolution), no figure title, no
    # parenthetical annotations, large bold labels -- built to be read from a slide.
    present = "--present" in args
    wanted = set(a for a in args if not a.startswith("--"))  # rebuild only these stems
    variants = ALL_VARIANTS if all_res else VARIANTS
    suffix = ("_allres" if all_res else "") + ("_present" if present else "")
    for stem, split, nominal in EXAMPLES:
        if wanted and stem not in wanted:
            continue
        print(f"== {stem}.png ({split}, {nominal} larvae)")
        build_figure(stem, split, nominal, specs, variants, suffix, present)


if __name__ == "__main__":
    main()
