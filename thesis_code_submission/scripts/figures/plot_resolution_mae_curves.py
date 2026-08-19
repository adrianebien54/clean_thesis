#!/usr/bin/env python3
"""Per-epoch validation MAE learning curves, faceted by resolution (CSRNet vs MobileCount).

Same style as exp/lr_grid_06-12_20-16/csrnet_lr_grid_overfit.png: faint raw per-epoch
val MAE + a bold 51-epoch rolling median, a star at the best (min-MAE) epoch, a dotted
line at the best level, and a legend annotation "best X @ ep Y (final Z, delta)" where
delta = final-epoch MAE minus best MAE (the overfit gap). One panel per resolution;
within a panel CSRNet (blue) and MobileCount (orange) are directly comparable (MAE is
in count units, LOG_PARA divided out). y-axis is zoomed to the converged/overfit band.

Same 6 resolutions / best-combo AUG=none cells / sources as
scripts/combine_resolution_results.py (386x260 folded in from its original sweeps).

Writes exp/combined_resolution_mae_curves.png.
"""
from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = str(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])
os.chdir(ROOT)

RESOLUTIONS = [
    ("49x33", 49, 33),
    ("97x65", 97, 65),
    ("193x130", 193, 130),
    ("386x260", 386, 260),
    ("772x519", 772, 519),
    ("1544x1038", 1544, 1038),
]
NETS = [("CSRNet", "tab:blue"), ("MobileCount", "tab:orange")]

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
    pts = sorted((m["step"], m["value"]) for m in cell["metrics"] if m["tag"] == tag)
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
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (v, w, h) in zip(axes.ravel(), RESOLUTIONS):
        panel_bests, panel_finals = [], []
        for net, color in NETS:
            if net not in curves[v]:
                continue
            ep, y = curves[v][net]
            med = rolling_median(y, 51)
            bi = int(np.argmin(y))
            best, best_ep, final = float(y[bi]), int(ep[bi]), float(y[-1])
            delta = final - best
            panel_bests.append(best)
            panel_finals.append(final)
            ax.plot(ep, y, color=color, lw=0.6, alpha=0.20)
            ax.plot(ep, med, color=color, lw=1.8,
                    label=f"{net} (51-ep median)")
            ax.plot(best_ep, best, marker="*", ms=15, color=color, mec="k", mew=0.6,
                    ls="none",
                    label=f"{net} best {best:.3f} @ ep {best_ep} (final {final:.3f}, Δ{delta:+.3f})")
            ax.axhline(best, color=color, ls=":", lw=0.8, alpha=0.7)
        # zoom y to span best -> final (the convergence + overfit band) for both nets
        if panel_bests:
            ax.set_ylim(0.90 * min(panel_bests), 1.08 * max(panel_finals))
        ax.set_title(f"{v}  ({w*h:,} px)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MAE")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Best-combo val MAE vs epoch, by resolution "
                 "(Tenebrio, seed 3035, AUG=none, step γ=0.995, 1500 ep)\n"
                 "faint = raw per-epoch, bold = 51-ep rolling median, ★ = best epoch, "
                 "Δ = final − best (overfit gap)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = "exp/combined_resolution/mae_curves_by_resolution.png"
    os.makedirs("exp/combined_resolution", exist_ok=True)
    fig.savefig(out, dpi=150)
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
