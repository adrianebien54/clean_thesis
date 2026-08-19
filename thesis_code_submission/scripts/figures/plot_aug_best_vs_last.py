#!/usr/bin/env python3
"""Best-checkpoint vs last-epoch MAE (train and val), all three arms (thesis).

2x3 small multiples (net x arm: noaug | mc80 | csrnet9). Each panel slopes the
train and validation MAE from the selected (best-val-MAE) checkpoint to the
final epoch-1500 state: under both aug schemes the two curves rise TOGETHER
and end nearly equal (the late degradation is not memorization), while noaug
keeps train below val at both points (MobileCount most visibly). Same
palette/arm colors as the other aug figures; within a panel the split is
encoded as filled+solid (val) vs open+dashed (train).

Reads exp/aug_paper_schemes_386x260_07-04/{trainsplit_results.json,
lastepoch_results.json} (the latter from
scripts/eval_aug_paper_schemes_lastepoch.py); writes
aug_paper_schemes_best_vs_last.{png,pdf} (300 dpi) next to them.
"""
import os
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])

EXP = ROOT / "exp/aug_paper_schemes_386x260_07-04"

# --- palette (dataviz reference instance, light mode; arm colors fixed
#     across all aug figures -- see plot_aug_paper_schemes.py)
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
ARM_ORDER = ["noaug", "mc80", "csrnet9"]
ARM_LABEL = {"noaug": "no aug.", "mc80": "mc80", "csrnet9": "csrnet9"}
ARM_COLOR = {"noaug": "#2a78d6", "mc80": "#1baf7a", "csrnet9": "#eda100"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "axes.titlesize": 9, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 120, "savefig.dpi": 300,
})

best = {(r["net"], r["arm"]): r
        for r in json.load(open(EXP / "trainsplit_results.json"))}
last = {(r["net"], r["arm"]): r
        for r in json.load(open(EXP / "lastepoch_results.json"))}

fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.9), sharey="row")
ylims = {"CSRNet": (3.25, 4.65), "MobileCount": (0.36, 1.45)}

for i, net in enumerate(("CSRNet", "MobileCount")):
    fmt = "{:.2f}" if net == "CSRNet" else "{:.3f}"
    for j, arm in enumerate(ARM_ORDER):
        ax = axes[i, j]
        b, l = best[(net, arm)], last[(net, arm)]
        c = ARM_COLOR[arm]
        val_ys = [b["val_mae"], l["val_mae"]]
        trn_ys = [b["train_mae"], l["train_mae"]]
        series = [("val", val_ys, dict(ls="-", mfc=c)),
                  ("train", trn_ys, dict(ls=(0, (4, 1.6)), mfc="white"))]
        for name, ys, style in series:
            ax.plot([0, 1], ys, color=c, lw=1.8, ls=style["ls"], zorder=3,
                    solid_capstyle="round", dash_capstyle="round")
            ax.plot([0, 1], ys, "o", ms=5.5, mfc=style["mfc"], mec=c, mew=1.3,
                    ls="none", zorder=4)
            for k, txt in enumerate((fmt.format(ys[0]),
                                     f"{name} {fmt.format(ys[1])}")):
                # the higher point of the pair gets its label above, the
                # lower below (val wins the tie), so labels track the points
                other = trn_ys[k] if name == "val" else val_ys[k]
                up = ys[k] >= other if name == "val" else ys[k] > other
                dy = 7 if up else -8
                ax.annotate(txt, (k, ys[k]),
                            xytext=(0, dy), textcoords="offset points",
                            ha="center", va="bottom" if up else "top",
                            fontsize=7.5,
                            color=INK if name == "val" and k else INK2,
                            zorder=5)
        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"best (ep {b['best_epoch']})",
                            f"last (ep {l['last_epoch']})"],
                           fontsize=8, color=INK2)
        ax.set_ylim(*ylims[net])
        ax.set_title(f"{net} — {ARM_LABEL[arm]}", loc="left", fontweight="bold")
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
    axes[i, 0].set_ylabel("MAE")

fig.tight_layout(w_pad=1.6, h_pad=2.2)
for ext in ("png", "pdf"):
    fig.savefig(EXP / f"aug_paper_schemes_best_vs_last.{ext}", bbox_inches="tight")
plt.close(fig)
print("written:", *sorted(p.name for p in EXP.glob("aug_paper_schemes_best_vs_last.*")))
