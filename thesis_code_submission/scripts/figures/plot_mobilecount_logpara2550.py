#!/usr/bin/env python3
"""Plot results of the MobileCount LOG_PARA=638.73 (2550@772x519) bs6 LR-grid sweep.

Reads mobilecount_logpara2550_results.json from the sweep dir (or the latest
matching dir) and writes three PNGs next to it:
  - best_mae_vs_lr.png   : best MAE (and MSE) vs LR, Adam vs AdamW
  - val_mae_curves.png   : per-epoch validation MAE for all 6 runs
  - val_mae_curves_zoom.png : same, y-clipped to show the converged tail
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_dir() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    cands = sorted(glob.glob("exp/mobilecount_logpara2550_anchor772x519_*"))
    if not cands:
        sys.exit("no sweep dir found")
    return cands[-1]


def series(run: dict, tag: str):
    pts = sorted((m["step"], m["value"]) for m in run["metrics"] if m["tag"] == tag)
    return [p[0] for p in pts], [p[1] for p in pts]


def main() -> None:
    d = find_dir()
    results = json.load(open(os.path.join(d, "mobilecount_logpara2550_results.json")))
    results = [r for r in results if "error" not in r]
    print(f"plotting {len(results)} runs from {d}")

    color = {"Adam": "tab:blue", "AdamW": "tab:red"}

    # ---- 1) best MAE / MSE vs LR ------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    for opt in ("Adam", "AdamW"):
        rows = sorted((r for r in results if r["optimizer"] == opt), key=lambda r: r["lr"])
        lrs = [r["lr"] for r in rows]
        ax.plot(lrs, [r["best_mae"] for r in rows], "o-", color=color[opt], label=f"{opt} MAE")
        ax.plot(lrs, [r["best_mse"] for r in rows], "s--", color=color[opt], alpha=0.5,
                label=f"{opt} MSE")
        for r in rows:
            ax.annotate(f"{r['best_mae']:.3f}", (r["lr"], r["best_mae"]),
                        textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("best validation MAE / MSE (counts)")
    ax.set_title("MobileCount bs6, LOG_PARA=638.73 (2550@772x519), Tenebrio 386x260\n"
                 "best metric over 1500 epochs, seed 3035")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(d, "best_mae_vs_lr.png")
    fig.savefig(p1, dpi=130)
    print("wrote", p1)

    # ---- 2) per-epoch val MAE curves --------------------------------------
    def curve_fig(ylim_top, fname, title_suffix):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ls = {1e-3: "-", 1e-4: "--", 1e-5: ":"}
        for r in results:
            ep, mae = series(r, "mae")
            ax.plot(ep, mae, ls.get(r["lr"], "-"), color=color[r["optimizer"]], alpha=0.85,
                    label=f"{r['optimizer']} lr{r['lr']:.0e} (best {r['best_mae']:.3f})")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation MAE (counts)")
        if ylim_top:
            ax.set_ylim(0, ylim_top)
        ax.set_title(f"MobileCount val MAE over training {title_suffix}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        path = os.path.join(d, fname)
        fig.savefig(path, dpi=130)
        print("wrote", path)

    curve_fig(None, "val_mae_curves.png", "(full range)")
    curve_fig(1.5, "val_mae_curves_zoom.png", "(zoom: MAE ≤ 1.5)")


if __name__ == "__main__":
    main()
