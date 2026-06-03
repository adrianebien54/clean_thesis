#!/usr/bin/env python3
"""Generate TensorBoard-style 4-panel training curve plot from pipeline tfevents file."""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]

_parser = argparse.ArgumentParser(description="Plot training curves from a tfevents experiment dir.")
_parser.add_argument("exp_dir", nargs="?", default="exp/05-20_13-07_Tenebrio_CSRNet_0.0001",
                     help="Path to experiment dir, relative to repo root or absolute.")
_args = _parser.parse_args()

_exp_path = Path(_args.exp_dir)
EXP_DIR = _exp_path if _exp_path.is_absolute() else ROOT / _exp_path


def load_scalars_all(exp_dir, tag):
    """Load a tag from all tfevents files in exp_dir, merge and sort by step."""
    tf_files = sorted(glob.glob(str(exp_dir / "events.out.tfevents.*")))
    steps, values = [], []
    for tf_file in tf_files:
        ea = EventAccumulator(tf_file)
        ea.Reload()
        if tag not in ea.Tags().get("scalars", []):
            continue
        for e in ea.Scalars(tag):
            steps.append(e.step)
            values.append(e.value)
    order = np.argsort(steps)
    return np.array(steps, dtype=float)[order], np.array(values, dtype=float)[order]


def smooth(values, window=15):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    pad    = np.pad(values, (window // 2, window // 2), mode="edge")
    return np.convolve(pad, kernel, mode="valid")[: len(values)]


def plot_panel(ax, steps, values, title, xlabel):
    ax.plot(steps, values, color="#c8d8e8", linewidth=0.8, alpha=0.9)
    ax.plot(steps, smooth(values), color="#3a6ea5", linewidth=1.6)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))
    ax.grid(True, linewidth=0.4, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def main():
    mae_steps, mae_vals = load_scalars_all(EXP_DIR, "mae")
    mse_steps, mse_vals = load_scalars_all(EXP_DIR, "mse")
    tl_steps,  tl_vals  = load_scalars_all(EXP_DIR, "train_loss")
    vl_steps,  vl_vals  = load_scalars_all(EXP_DIR, "val_loss")

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    fig.subplots_adjust(wspace=0.35, left=0.05, right=0.98, top=0.88, bottom=0.15)

    plot_panel(axes[0], mae_steps, mae_vals,  "mae",        "Epoch")
    plot_panel(axes[1], mse_steps, mse_vals,  "mse",        "Epoch")
    plot_panel(axes[2], tl_steps,  tl_vals,   "train_loss", "Iteration")
    plot_panel(axes[3], vl_steps,  vl_vals,   "val_loss",   "Epoch")

    out = EXP_DIR / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
