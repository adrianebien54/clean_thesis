#!/usr/bin/env python3
"""Plot Adam-vs-AdamW val-MAE curves per seed from an optimizer seed sweep.

Reads optimizer_seed_sweep_results.json (written by sweep_optimizer_seeds.py)
and produces one PNG in the sweep dir: top row smoothed val MAE per seed
(raw shown faint, star at best checkpoint), bottom row best-so-far val MAE.
Style matches the single-run comparison plots in optimizer_arch_compare_06-10_18-12.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

COLORS = {"Adam": "tab:blue", "AdamW": "tab:orange"}
SMOOTH_WINDOW = 51  # epochs, rolling median


def rolling_median(values, window):
    if len(values) < window:
        return values.copy()
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(len(values))])


def mae_series(run):
    pts = sorted((m["step"], m["value"]) for m in run["metrics"] if m["tag"] == "mae")
    return np.array([s for s, _ in pts]), np.array([v for _, v in pts])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", help="Sweep dir containing optimizer_seed_sweep_results.json")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.is_absolute():
        sweep_dir = ROOT / sweep_dir
    runs = json.loads((sweep_dir / "optimizer_seed_sweep_results.json").read_text())
    runs = [r for r in runs if "error" not in r]
    if not runs:
        raise SystemExit("No completed runs in results JSON.")

    net = runs[0]["net"]
    seeds = sorted({r["seed"] for r in runs}, key=lambda s: [r["seed"] for r in runs].index(s))
    by_key = {(r["optimizer"], r["seed"]): r for r in runs}
    optimizers = [o for o in ("Adam", "AdamW") if any(r["optimizer"] == o for r in runs)]

    fig, axes = plt.subplots(2, len(seeds), figsize=(5.2 * len(seeds), 7.6),
                             sharex=True, sharey="row", squeeze=False)
    lr = runs[0]["lr"]
    fig.suptitle(f"{net} on Tenebrio {runs[0]['variant']} — Adam vs AdamW per seed "
                 f"(lr {lr:g}, step γ=0.995, bs {runs[0]['batch_size']})", fontsize=13)

    for col, seed in enumerate(seeds):
        ax_top, ax_bot = axes[0][col], axes[1][col]
        for opt in optimizers:
            run = by_key.get((opt, seed))
            if run is None:
                continue
            steps, vals = mae_series(run)
            color = COLORS[opt]
            ax_top.plot(steps, vals, color=color, linewidth=0.6, alpha=0.18)
            ax_top.plot(steps, rolling_median(vals, SMOOTH_WINDOW), color=color,
                        linewidth=1.8, label=f"{opt} ({SMOOTH_WINDOW}-ep rolling median)")
            best_idx = int(np.argmin(vals))
            ax_top.plot(steps[best_idx], vals[best_idx], marker="*", markersize=14,
                        markeredgecolor="black", markerfacecolor=color, linestyle="none",
                        label=f"{opt} best ckpt: {vals[best_idx]:.3f} @ ep {int(steps[best_idx])}")
            ax_bot.plot(steps, np.minimum.accumulate(vals), color=color,
                        linewidth=1.8, label=f"{opt} best-so-far")
        ax_top.set_title(f"seed {seed}", fontsize=11)
        ax_top.set_ylim(0.4, 2.0)
        ax_top.legend(fontsize=7, loc="upper right")
        ax_bot.set_ylim(0.4, 1.0)
        ax_bot.set_xlabel("epoch")
        ax_bot.legend(fontsize=7, loc="upper right")
        for ax in (ax_top, ax_bot):
            ax.grid(True, linewidth=0.4, alpha=0.5)
    axes[0][0].set_ylabel("val MAE")
    axes[1][0].set_ylabel("best val MAE so far")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = sweep_dir / f"{net.lower()}_seed_sweep_adam_vs_adamw.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
