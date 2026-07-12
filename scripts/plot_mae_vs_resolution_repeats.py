#!/usr/bin/env python3
"""MAE-vs-resolution figure with error bars from the same-seed repeat sweep.

Same layout as plot_mae_vs_resolution.py (log-log connected scatter, one line
per net x split; validation in full colour, test in a lighter dashed shade),
but each point is the mean across the same-seed repeats and the error bars are
±1 sample std (framework nondeterminism at seed 3035, 800 epochs).

Reads $SWEEP_DIR/testset_repeats_results.json (per-rep rows written by
eval_bestcombo_repeats_testset.py). Writes
$SWEEP_DIR/mae_vs_resolution_repeats.png/.pdf.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

ROOT = Path("/home/umrobotics/clean_thesis/crowd_counting")
SWEEP_DIR = ROOT / os.environ.get(
    "SWEEP_DIR", "./exp/bestcombo_resolutions_repeats_800ep")
DATA = SWEEP_DIR / "testset_repeats_results.json"
OUT = SWEEP_DIR / "mae_vs_resolution_repeats"

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
    labels = [v.replace("x", "×") for v, _ in variants]
    by: dict[tuple, list[dict]] = {}
    for r in rows:
        by.setdefault((r["net"], r["variant"]), []).append(r)
    n_reps = sorted({len(v) for v in by.values()})

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for net, key, color, ls, label in SERIES:
        means = [float(np.mean([r[key] for r in by[(net, v)]])) for v, _ in variants]
        stds = [float(np.std([r[key] for r in by[(net, v)]], ddof=1))
                if len(by[(net, v)]) > 1 else 0.0 for v, _ in variants]
        ax.errorbar(areas, means, yerr=stds, color=color, ls=ls, lw=2,
                    marker=MARKERS[net], ms=7, mec="white", mew=1.2,
                    capsize=3, capthick=1.2, elinewidth=1.2,
                    solid_capstyle="round", label=label, zorder=3)

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
    n_txt = "/".join(str(n) for n in n_reps)
    ax.legend(loc="upper right", frameon=False, fontsize=9,
              title=f"error bars: ±1 std, n={n_txt} same-seed runs", title_fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=200)
    fig.savefig(f"{OUT}.pdf")
    print(f"wrote {OUT}.png and .pdf")


if __name__ == "__main__":
    main()
