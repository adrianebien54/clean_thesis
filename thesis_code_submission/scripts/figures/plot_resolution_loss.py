#!/usr/bin/env python3
"""Per-epoch validation-loss learning curves across all 6 resolutions, per net.

Companion to scripts/combine_resolution_results.py. Same 6 resolutions / same
best-combo AUG=none cells / same sources (386x260 folded in from its original sweeps).

NOTE: loss is MSE on the LOG_PARA-scaled density maps, and LOG_PARA auto-scales with
resolution (~0.4 at 49x33 up to ~10200 at 1544 MobileCount), so absolute loss levels
are NOT comparable across resolutions -- read the CONVERGENCE SHAPE, not the level.
val_loss is logged every epoch for both nets; train_loss is only logged for CSRNet
(MobileCount's bs6 runs hit too few iters/epoch for PRINT_FREQ=200), so this uses
val_loss for an apples-to-apples 2-panel figure.

Writes exp/combined_resolution_loss.png.
"""
from __future__ import annotations

from pathlib import Path
import json
import os

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
NETS = ["CSRNet", "MobileCount"]

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
    return [p[0] for p in pts], [p[1] for p in pts]


def collect_valloss():
    """curves[net][variant] = (epochs, val_loss)"""
    curves = {n: {} for n in NETS}

    for r in _rows(BESTCOMBO):
        v, n = str(r.get("variant")), r.get("net")
        if any(v == rv for rv, _, _ in RESOLUTIONS) and n in NETS:
            curves[n][v] = series(r, "val_loss")

    cs = [r for r in _rows(CSRNET_386)
          if r.get("net") == "CSRNet" and str(r.get("variant")) == "386x260"
          and int(r.get("batch_size", 0)) == 1]
    curves["CSRNet"]["386x260"] = series(best_of(cs), "val_loss")

    ms = [r for r in _rows(MOBILE_386)
          if r.get("net") == "MobileCount" and str(r.get("variant")) == "386x260"
          and int(r.get("batch_size", 0)) == 6 and abs(float(r.get("lr", 0)) - 1e-3) < 1e-9]
    curves["MobileCount"]["386x260"] = series(best_of(ms), "val_loss")
    return curves


def main() -> None:
    curves = collect_valloss()
    order = [v for v, _, _ in RESOLUTIONS]
    colors = plt.cm.viridis([i / (len(order) - 1) for i in range(len(order))])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
    for ax, net in zip(axes, NETS):
        for v, c in zip(order, colors):
            if v not in curves[net]:
                continue
            ep, y = curves[net][v]
            ax.plot(ep, y, color=c, lw=1.0, alpha=0.9, label=v)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation loss (MSE on LOG_PARA-scaled density, log scale)")
        ax.set_title(net)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(title="resolution", fontsize=8, ncol=2)
    fig.suptitle("Per-epoch validation loss by resolution "
                 "(Tenebrio, seed 3035, AUG=none, 1500 ep)\n"
                 "absolute loss is LOG_PARA-scaled per resolution — compare shape, not level",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = "exp/combined_resolution/val_loss_curves.png"
    os.makedirs("exp/combined_resolution", exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for net in NETS:
        for v in order:
            if v in curves[net]:
                ep, y = curves[net][v]
                print(f"  {net:<11} {v:<11} val_loss n={len(y)} last={y[-1]:.4g} min={min(y):.4g}")


if __name__ == "__main__":
    main()
