#!/usr/bin/env python3
"""Thesis version of the per-resolution val-MAE learning curves: 2x3 grid.

Same data/cells as scripts/plot_resolution_mae_curves.py (best-combo AUG=none,
seed 3035, 386x260 folded in from its original sweeps), restyled for the thesis:
faint raw per-epoch val MAE + bold 51-ep rolling median, star at the best
checkpoint, dotted line at the best level, epochs trimmed to 1200.

Layout: 2 rows x 3 cols, resolutions in reading order (49x33 ... 1544x1038).
Panels in the same row share the y-scale (sharey="row"), so curves are directly
comparable within a row; all panels share the x-axis.

Writes exp/combined_resolution/mae_curves_by_resolution_thesis_2x3.png.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/umrobotics/clean_thesis/crowd_counting"
os.chdir(ROOT)

RESOLUTIONS = [
    ("49x33", 49, 33),
    ("97x65", 97, 65),
    ("193x130", 193, 130),
    ("386x260", 386, 260),
    ("772x519", 772, 519),
    ("1544x1038", 1544, 1038),
]
NETS = [("CSRNet", "tab:blue"), ("MobileCount", "tab:red")]
MAX_EPOCH = 1200
# (variant, net) cells to hide from their panel, e.g. {("49x33", "CSRNet")}
OMIT = set()

BESTCOMBO = "exp/bestcombo_resolutions_06-24_11-49/bestcombo_resolutions_results.json"
CSRNET_386 = "exp/batchsize_sweep_logpara_06-21_22-41/batchsize_logpara_results.json"
MOBILE_386 = ("exp/360p_lr_optim_grid/mobilecount_logpara2550_anchor772x519_06-17_14-26/"
              "mobilecount_logpara2550_results.json")


def _rows(path):
    d = json.load(open(path))
    rows = d if isinstance(d, list) else d.get("runs", [])
    return [r for r in rows if isinstance(r, dict) and "error" not in r]


def best_of(rows):
    return min(rows, key=lambda r: (float(r["best_mae"]), float(r["best_mse"])))


def series(cell, tag):
    pts = sorted((m["step"], m["value"]) for m in cell["metrics"]
                 if m["tag"] == tag and m["step"] <= MAX_EPOCH)
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def rolling_median(y, w=51):
    n, h = len(y), w // 2
    return np.array([np.median(y[max(0, i - h):min(n, i + h + 1)]) for i in range(n)])


def collect_mae():
    """curves[variant][net] = (epochs, mae)"""
    curves = {v: {} for v, _, _ in RESOLUTIONS}
    for r in _rows(BESTCOMBO):
        v, n = str(r.get("variant")), r.get("net")
        if v in curves and n in (nn for nn, _ in NETS):
            curves[v][n] = series(r, "mae")

    cs = [r for r in _rows(CSRNET_386)
          if r.get("net") == "CSRNet" and str(r.get("variant")) == "386x260"
          and int(r.get("batch_size", 0)) == 1]
    curves["386x260"]["CSRNet"] = series(best_of(cs), "mae")

    ms = [r for r in _rows(MOBILE_386)
          if r.get("net") == "MobileCount" and str(r.get("variant")) == "386x260"
          and int(r.get("batch_size", 0)) == 6 and abs(float(r.get("lr", 0)) - 1e-3) < 1e-9]
    curves["386x260"]["MobileCount"] = series(best_of(ms), "mae")
    return curves


def main() -> None:
    curves = collect_mae()
    plt.rcParams.update({
        "axes.titlesize": 21, "axes.titleweight": "bold",
        "axes.labelsize": 19, "xtick.labelsize": 15, "ytick.labelsize": 15,
        "legend.fontsize": 15,
    })
    fig, axes = plt.subplots(2, 3, figsize=(19, 9.5), sharex=True)
    bests = {v: [] for v, _, _ in RESOLUTIONS}
    finals = {v: [] for v, _, _ in RESOLUTIONS}
    for ax, (v, w, h) in zip(axes.ravel(), RESOLUTIONS):
        for net, color in NETS:
            if net not in curves[v] or (v, net) in OMIT:
                continue
            ep, y = curves[v][net]
            med = rolling_median(y, 51)
            bi = int(np.argmin(y))
            best, best_ep, final = float(y[bi]), int(ep[bi]), float(y[-1])
            bests[v].append(best)
            finals[v].append(final)
            ax.plot(ep, y, color=color, lw=0.7, alpha=0.20)
            ax.plot(ep, med, color=color, lw=2.5)
            ax.plot(best_ep, best, marker="*", ms=18, color=color, mec="k",
                    mew=0.8, ls="none", zorder=5)
            ax.axhline(best, color=color, ls=":", lw=1.0, alpha=0.7)
        ax.set_title(f"res = {w}×{h}")
        ax.grid(True, ls=":", alpha=0.4)
    # two y-scales, padded below so the stars clear the axis edge:
    # 49x33 gets the wide best->final band (CSRNet plateaus at ~12.5); the
    # other five panels share one scale spanning their joint best->final band,
    # so every star, plateau and overfit rise stays visible (only the first
    # ~20 epochs of the steepest descents clip at the top).
    # Tick labels appear on 49x33, 97x65 (start of the shared scale) and
    # 386x260 (start of the bottom row).
    top = ["49x33", "97x65", "193x130"]
    rest = [v for v, _, _ in RESOLUTIONS if v != "49x33"]
    hi0 = 1.08 * max(f for v in top for f in finals[v])
    lo0 = min(b for v in top for b in bests[v])
    axes[0, 0].set_ylim(lo0 - 0.04 * (hi0 - lo0), hi0)
    hi1 = 1.08 * max(f for v in rest for f in finals[v])
    lo1 = min(b for v in rest for b in bests[v])
    for ax in (axes[0, 1], axes[0, 2], *axes[1]):
        ax.set_ylim(lo1 - 0.04 * (hi1 - lo1), hi1)
    for ax in (axes[0, 2], axes[1, 1], axes[1, 2]):
        ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel("val MAE")
    axes[1, 0].set_ylabel("val MAE")
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    axes[0, 0].set_xlim(-30, MAX_EPOCH + 30)

    handles = [plt.Line2D([], [], color=c, lw=2.5, label=n) for n, c in NETS]
    handles.append(plt.Line2D([], [], marker="*", ms=18, color="w", mec="k",
                              mew=1.0, ls="none", label="best checkpoint"))
    axes[0, 2].legend(handles=handles, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = "exp/combined_resolution/mae_curves_by_resolution_thesis_2x3.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    for v, _, _ in RESOLUTIONS:
        for net, _ in NETS:
            if net in curves[v]:
                ep, y = curves[v][net]
                bi = int(np.argmin(y))
                print(f"  {v:<11} {net:<11} best {y[bi]:.3f} @ ep {int(ep[bi]):<4} "
                      f"final {y[-1]:.3f}  Δ{y[-1]-y[bi]:+.3f}")


if __name__ == "__main__":
    main()
