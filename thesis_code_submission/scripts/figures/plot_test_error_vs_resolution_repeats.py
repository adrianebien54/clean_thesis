#!/usr/bin/env python3
"""Companion figure for the cross-resolution test-error table (Table: tab:resolution_test).

Two panels over one resolution axis:

  (a) test MAE  -- log-log, mean +- 1 sample std over the same-seed repeats of a cell
  (b) test MSE  -- same layout, so the MAE/MSE disagreement between the nets is visible

Built from the five 800-epoch repeats in exp/bestcombo_resolutions_repeats_800ep alone
(identical config and seed 3035 in all five; spread = framework nondeterminism). Set
INCLUDE_1500EP=1 to pool in the original 1500-epoch run as a sixth sample, matching
scripts/pool_bestcombo_resolutions_6runs.py.

LAYOUT=wide (default) writes a 1x2 figure for a full-width IEEEtran figure*;
LAYOUT=column writes the same two panels stacked for a single \\linewidth column.
Both are written when LAYOUT=both.

Writes $SWEEP_DIR/test_error_vs_resolution_repeats[_column].png/.pdf.
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

ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])
SWEEP_DIR = ROOT / os.environ.get(
    "SWEEP_DIR", "./exp/bestcombo_resolutions_repeats_800ep")
DATA = SWEEP_DIR / "testset_repeats_results.json"
ORIGINAL_JSON = ROOT / "exp/combined_resolution/testset_results.json"
INCLUDE_1500EP = os.environ.get("INCLUDE_1500EP", "0") != "0"
LAYOUT = os.environ.get("LAYOUT", "wide")
OUT = SWEEP_DIR / "test_error_vs_resolution_repeats"

# net -> (colour, marker); same assignment as every other figure in the thesis.
NETS = [("CSRNet", "#1f77b4", "o"), ("MobileCount", "#d62728", "s")]
GRID = "#e3e3e3"
AXIS = "#bbbbbb"
TICK = "#555555"


def load_cells() -> tuple[list[str], list[int], dict[tuple[str, str], dict]]:
    """Group the per-run rows into (net, variant) cells and reduce them to stats."""
    runs = json.load(DATA.open())
    if INCLUDE_1500EP:
        runs = runs + json.load(ORIGINAL_JSON.open())

    by: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        by.setdefault((r["net"], r["variant"]), []).append(r)

    order = sorted({(r["variant"], r["pixels"]) for r in runs}, key=lambda p: p[1])
    variants = [v for v, _ in order]
    pixels = [px for _, px in order]

    cells = {}
    for key, rs in by.items():
        cell = {"n": len(rs)}
        for metric in ("test_mae", "test_mse"):
            vals = [r[metric] for r in rs]
            cell[f"{metric}_mean"] = float(np.mean(vals))
            cell[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        cells[key] = cell
    return variants, pixels, cells


def style(ax, *, xlabel: bool, labels: list[str], pixels: list[int]):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(pixels))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_locator(NullLocator())
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TICK, labelsize=8.5)
    if xlabel:
        ax.set_xlabel("Input resolution", fontsize=9.5)
    else:
        plt.setp(ax.get_xticklabels(), visible=False)


def error_panel(ax, metric: str, variants, pixels, cells, *, yticks, title, ylabel):
    for net, color, marker in NETS:
        means = [cells[(net, v)][f"{metric}_mean"] for v in variants]
        stds = [cells[(net, v)][f"{metric}_std"] for v in variants]
        ax.errorbar(pixels, means, yerr=stds, color=color, lw=2, marker=marker, ms=6,
                    mec="white", mew=1.0, capsize=3, capthick=1.1, elinewidth=1.1,
                    solid_capstyle="round", label=net, zorder=3)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in yticks]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_title(title, fontsize=10, loc="left", color="#333333", pad=6)


def build(layout: str, variants, pixels, cells) -> None:
    labels = [v.replace("x", "×") for v in variants]
    wide = layout == "wide"

    if wide:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(3.6, 5.4), sharex=True)
    ax_mae, ax_mse = axes

    error_panel(ax_mae, "test_mae", variants, pixels, cells,
                yticks=[0.25, 0.5, 1, 2, 5], title="(a) Test MAE", ylabel="Test MAE")
    error_panel(ax_mse, "test_mse", variants, pixels, cells,
                yticks=[0.5, 1, 2, 5], title="(b) Test MSE", ylabel="Test MSE")

    for i, ax in enumerate(axes):
        style(ax, xlabel=wide or i == len(axes) - 1, labels=labels, pixels=pixels)

    ax_mae.legend(loc="upper right", frameon=False, fontsize=9, handlelength=1.6,
                  borderaxespad=0.2)

    fig.tight_layout(w_pad=2.0 if wide else None, h_pad=1.2)
    suffix = "" if wide else "_column"
    fig.savefig(f"{OUT}{suffix}.png", dpi=220)
    fig.savefig(f"{OUT}{suffix}.pdf")
    plt.close(fig)
    print(f"wrote {OUT}{suffix}.png and .pdf")


def main() -> None:
    variants, pixels, cells = load_cells()
    for layout in (["wide", "column"] if LAYOUT == "both" else [LAYOUT]):
        build(layout, variants, pixels, cells)


if __name__ == "__main__":
    main()
