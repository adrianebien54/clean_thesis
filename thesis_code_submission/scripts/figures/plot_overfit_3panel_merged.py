#!/usr/bin/env python3
"""Merged overfit figure: the two 1x3 "*_overfit_3panel.png" figures stacked.

Row 1 = CSRNet (lr 1e-4, 1e-5, 1e-6), row 2 = MobileCount (lr 1e-3, 1e-4,
1e-5), cropped at MAX_EPOCH. Same styling as the originals: faint raw val-MAE
curve, bold 51-epoch rolling median, star at the best checkpoint, dotted line
at the best-MAE floor, y-limits shared within each row, per-net colors
(CSRNet orange/blue, MobileCount green/red).

Two outputs: overfit_3panel_merged.* with per-row y-limits, and
overfit_3panel_merged_sharedscale.* with one y-scale across both nets.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "exp" / "360p_lr_optim_grid"

ROWS = [
    {
        "net": "CSRNet",
        "results": GRID / "csrnet_logpara100_anchor772x519_06-20_14-41"
        / "csrnet_logpara_results.json",
        "lrs": [1e-4, 1e-5, 1e-6],
        "colors": {"Adam": "tab:orange", "AdamW": "tab:blue"},
    },
    {
        "net": "MobileCount",
        "results": GRID / "mobilecount_logpara2550_anchor772x519_06-17_14-26"
        / "mobilecount_logpara2550_results.json",
        "lrs": [1e-3, 1e-4, 1e-5],
        "colors": {"Adam": "tab:green", "AdamW": "tab:red"},
    },
]

SMOOTH_WINDOW = 51  # epochs, rolling median
WARMUP = 30  # epochs ignored when auto-scaling the y-axis
MAX_EPOCH = 1200  # crop the x-axis here (curves are flat beyond)


def rolling_median(values, window):
    if len(values) < window:
        return values.copy()
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(len(values))])


def mae_series(run):
    pts = sorted((m["step"], m["value"]) for m in run["metrics"]
                 if m["tag"] == "mae" and m["step"] <= MAX_EPOCH)
    return np.array([s for s, _ in pts]), np.array([v for _, v in pts])


def load_cells(results_file):
    cells = {}
    for r in json.loads(Path(results_file).read_text()):
        if "error" in r:
            continue
        key = (r["optimizer"], r["lr"])
        if key not in cells or r["max_epoch"] > cells[key]["max_epoch"]:
            cells[key] = r
    return cells


def lr_exponent(lr):
    return int(round(np.log10(lr)))


def make_figure(shared_scale):
    ncols = 3
    fig, axes = plt.subplots(2, ncols, figsize=(12.5, 6.8), sharex=True,
                             sharey="row")

    row_ranges = []
    for row, spec in enumerate(ROWS):
        cells = load_cells(spec["results"])
        colors = spec["colors"]
        lo_all, hi_all = [], []
        for col, lr in enumerate(spec["lrs"]):
            ax = axes[row][col]
            for opt in ("Adam", "AdamW"):
                run = cells.get((opt, lr))
                if run is None:
                    continue
                steps, vals = mae_series(run)
                color = colors[opt]
                smooth = rolling_median(vals, SMOOTH_WINDOW)
                ax.plot(steps, vals, color=color, linewidth=0.7, alpha=0.25)
                ax.plot(steps, smooth, color=color, linewidth=2.2)

                best_idx = int(np.argmin(vals))
                ax.plot(int(steps[best_idx]), float(vals[best_idx]), marker="*",
                        markersize=16, markeredgecolor="black",
                        markerfacecolor=color, linestyle="none", zorder=5)
                ax.axhline(float(vals[best_idx]), color=color, linewidth=0.9,
                           linestyle=":", alpha=0.7)

                warm = smooth[WARMUP:] if len(smooth) > WARMUP else smooth
                lo_all.append(float(np.min(vals)))
                hi_all.append(float(np.max(warm)))

            ax.set_title(rf"$\mathbf{{lr = 10^{{{lr_exponent(lr)}}}}}$", fontsize=15)
            ax.tick_params(labelsize=11)
            if row == 1:
                ax.set_xlabel("epoch", fontsize=13)
            if col == 0:
                ax.set_ylabel("val MAE", fontsize=13)

        row_ranges.append((min(lo_all), max(hi_all)))

        legend_handles = [
            Line2D([], [], color=colors["Adam"], linewidth=2.2, label="Adam"),
            Line2D([], [], color=colors["AdamW"], linewidth=2.2, label="AdamW"),
            Line2D([], [], marker="*", markersize=14, markeredgecolor="black",
                   markerfacecolor="white", linestyle="none",
                   label="best checkpoint"),
        ]
        axes[row][0].legend(handles=legend_handles, fontsize=11, loc="upper right")

    if shared_scale:
        lo = min(r[0] for r in row_ranges)
        hi = max(r[1] for r in row_ranges)
        row_ranges = [(lo, hi)] * len(ROWS)
    for row, (lo, hi) in enumerate(row_ranges):
        margin = 0.05 * (hi - lo + 1e-6)
        axes[row][0].set_ylim(lo - margin, hi + margin)

    fig.tight_layout(rect=(0.045, 0, 1, 1))
    for row, spec in enumerate(ROWS):
        pos = axes[row][0].get_position()
        fig.text(pos.x0 - 0.052, (pos.y0 + pos.y1) / 2, spec["net"],
                 rotation=90, ha="center", va="center", fontsize=18,
                 fontweight="bold")

    stem = "overfit_3panel_merged" + ("_sharedscale" if shared_scale else "")
    for ext in ("png", "pdf"):
        out = GRID / f"{stem}.{ext}"
        fig.savefig(out, dpi=200)
        print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure(shared_scale=False)
    make_figure(shared_scale=True)
