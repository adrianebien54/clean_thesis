#!/usr/bin/env python3
"""MAE-vs-resolution figure: validation and test MAE against input resolution.

Log-log connected scatter, one line per net x split. Validation in full
colour (CSRNet blue, MobileCount red), test in a lighter shade of the same
hue with dashed lines. The asterisked tick marks the tuning resolution.

Reads exp/combined_resolution/testset_results.json (val_mae + test_mae per
net x resolution). Writes exp/combined_resolution/mae_vs_resolution.png/.pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

ROOT = Path("/home/umrobotics/clean_thesis/crowd_counting")
DATA = ROOT / "exp/combined_resolution/testset_results.json"
OUT = ROOT / "exp/combined_resolution/mae_vs_resolution"

TUNING = "386x260"
# (net, metric key, colour, linestyle, label)
SERIES = [
    ("CSRNet", "val_mae", "#1f77b4", "-", "CSRNet (validation)"),
    ("CSRNet", "test_mae", "#8fc1e3", "--", "CSRNet (test)"),
    ("MobileCount", "val_mae", "#d62728", "-", "MobileCount (validation)"),
    ("MobileCount", "test_mae", "#f49b9b", "--", "MobileCount (test)"),
]
MARKERS = {"CSRNet": "o", "MobileCount": "s"}


def main() -> None:
    rows = json.load(DATA.open())
    variants = sorted({(r["variant"], r["pixels"]) for r in rows}, key=lambda p: p[1])
    areas = [px for _, px in variants]
    labels = [v.replace("x", "×") + ("*" if v == TUNING else "") for v, _ in variants]
    by = {(r["net"], r["variant"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for net, key, color, ls, label in SERIES:
        ys = [by[(net, v)][key] for v, _ in variants]
        ax.plot(areas, ys, color=color, ls=ls, lw=2, marker=MARKERS[net], ms=7,
                mec="white", mew=1.2, solid_capstyle="round", label=label, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    yticks = [0.25, 0.5, 1, 2, 5]
    ax.xaxis.set_major_locator(FixedLocator(areas))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in yticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax.grid(True, which="major", color="#e0e0e0", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bbbbbb")
    ax.tick_params(colors="#555555", labelsize=9)

    ax.set_xlabel("Input resolution (log scale)", fontsize=10)
    ax.set_ylabel("MAE (log scale)", fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=200)
    fig.savefig(f"{OUT}.pdf")
    print(f"wrote {OUT}.png and .pdf")


if __name__ == "__main__":
    main()
