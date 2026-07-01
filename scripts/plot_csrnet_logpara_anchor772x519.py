#!/usr/bin/env python3
"""Plot results of the CSRNet LOG_PARA=25.05 (100@772x519) bs6 LR-grid sweep.

Counterpart to plot_mobilecount_logpara2550.py / _overfit.py. Reads
csrnet_logpara_results.json from the sweep dir (arg, or the latest matching dir)
and writes two PNGs next to it:
  - best_mae_vs_lr.png       : best MAE (and MSE) vs LR, Adam vs AdamW
  - csrnet_logpara25_overfit.png : one panel per LR overlaying Adam vs AdamW val
        MAE; raw curve faint, rolling median bold, a star at the best checkpoint
        and a dotted best-MAE floor. The curve rising after the star is overfit.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"Adam": "tab:blue", "AdamW": "tab:orange"}
SMOOTH_WINDOW = 51  # epochs, rolling median
WARMUP = 30  # epochs ignored when auto-scaling the overfit-panel y-axis


def find_dir() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    cands = sorted(glob.glob("exp/csrnet_logpara100_anchor772x519_*"))
    if not cands:
        sys.exit("no sweep dir found")
    return cands[-1]


def mae_series(run):
    pts = sorted((m["step"], m["value"]) for m in run["metrics"] if m["tag"] == "mae")
    return np.array([s for s, _ in pts]), np.array([v for _, v in pts])


def rolling_median(values, window):
    if len(values) < window:
        return values.copy()
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(len(values))])


def lr_label(lr):
    return f"{lr:.0e}".replace("e-0", "e-")


def plot_best_vs_lr(results, sample, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    for opt in ("Adam", "AdamW"):
        rows = sorted((r for r in results if r["optimizer"] == opt), key=lambda r: r["lr"])
        lrs = [r["lr"] for r in rows]
        ax.plot(lrs, [r["best_mae"] for r in rows], "o-", color=COLORS[opt], label=f"{opt} MAE")
        ax.plot(lrs, [r["best_mse"] for r in rows], "s--", color=COLORS[opt], alpha=0.5,
                label=f"{opt} MSE")
        for r in rows:
            ax.annotate(f"{r['best_mae']:.3f}", (r["lr"], r["best_mae"]),
                        textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("best validation MAE / MSE (counts, log scale)")
    ax.set_title(f"CSRNet bs{sample['batch_size']}, LOG_PARA={sample['log_para']:.2f} "
                 f"(100@772x519), Tenebrio {sample['variant']}\n"
                 f"best metric over {sample['max_epoch']} epochs, seed {sample['seed']}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "best_mae_vs_lr.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


def plot_overfit(cells, net, sample, out_dir):
    lrs = sorted({lr for (n, _, lr) in cells if n == net}, reverse=True)
    ncols = 2 if len(lrs) <= 4 else 3
    nrows = math.ceil(len(lrs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows), squeeze=False)
    fig.suptitle(
        f"{net} on Tenebrio {sample['variant']} — val MAE vs epoch, Adam vs AdamW "
        f"(LOG_PARA={sample['log_para']:.2f} = 100@772x519, step γ=0.995, "
        f"bs {sample['batch_size']}, seed {sample['seed']}, {sample['max_epoch']} ep)",
        fontsize=13,
    )

    flat = list(axes.flat)
    for ax in flat[len(lrs):]:  # hide unused panels
        ax.set_visible(False)
    for ax, lr in zip(flat, lrs):
        lo_candidates, hi_candidates = [], []
        for opt in ("Adam", "AdamW"):
            run = cells.get((net, opt, lr))
            if run is None:
                continue
            steps, vals = mae_series(run)
            color = COLORS[opt]
            smooth = rolling_median(vals, SMOOTH_WINDOW)
            ax.plot(steps, vals, color=color, linewidth=0.6, alpha=0.18)
            ax.plot(steps, smooth, color=color, linewidth=1.8,
                    label=f"{opt} ({SMOOTH_WINDOW}-ep rolling median)")

            best_idx = int(np.argmin(vals))
            best_ep, best_mae = int(steps[best_idx]), float(vals[best_idx])
            final_mae = float(vals[-1])
            ax.plot(best_ep, best_mae, marker="*", markersize=15,
                    markeredgecolor="black", markerfacecolor=color, linestyle="none",
                    label=(f"{opt} best {best_mae:.3f} @ ep {best_ep} "
                           f"(final {final_mae:.3f}, Δ{final_mae - best_mae:+.3f})"))
            ax.axhline(best_mae, color=color, linewidth=0.8, linestyle=":", alpha=0.7)

            warm = smooth[WARMUP:] if len(smooth) > WARMUP else smooth
            lo_candidates.append(float(np.min(vals)))
            hi_candidates.append(float(np.max(warm)))

        ax.set_title(f"lr = {lr_label(lr)}", fontsize=11)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MAE")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=7.5, loc="upper right")
        if lo_candidates:
            lo, hi = min(lo_candidates), max(hi_candidates)
            ax.set_ylim(max(0.0, lo - 0.05 * (hi - lo + 1e-6)), hi + 0.05 * (hi - lo + 1e-6))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(out_dir, f"{net.lower()}_logpara25_overfit.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    d = find_dir()
    results = json.load(open(os.path.join(d, "csrnet_logpara_results.json")))
    results = [r for r in results if "error" not in r]
    if not results:
        sys.exit("no successful runs in results JSON")
    print(f"plotting {len(results)} runs from {d}")

    sample = results[0]
    plot_best_vs_lr(results, sample, d)

    cells = {(r["net"], r["optimizer"], r["lr"]): r for r in results}
    for net in sorted({r["net"] for r in results}):
        plot_overfit(cells, net, sample, d)


if __name__ == "__main__":
    main()
