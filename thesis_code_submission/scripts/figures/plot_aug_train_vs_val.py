#!/usr/bin/env python3
"""Train-vs-validation MAE of the selected checkpoints, aug arms only (thesis).

One slope per arm (mc80, csrnet9) from train-split MAE to validation MAE,
2 panels (CSRNet | MobileCount). A flat or falling line = the checkpoint does
not memorize the training images; the per-arm gap Delta = val - train is
annotated at the midpoint. Test-free companion to Fig 2 of
plot_aug_paper_schemes.py (same palette, arm colors and linestyles).

Reads exp/aug_paper_schemes_386x260_07-04/trainsplit_results.json; writes
aug_paper_schemes_train_vs_val.{png,pdf} (300 dpi) next to it.
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
ARM_ORDER = ["mc80", "csrnet9"]
ARM_SHORT = {"mc80": "mc80", "csrnet9": "csrnet9"}
ARM_COLOR = {"mc80": "#1baf7a", "csrnet9": "#eda100"}
ARM_LS    = {"mc80": (0, (5, 1.6)), "csrnet9": (0, (1, 1.2))}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "axes.titlesize": 9.5, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 120, "savefig.dpi": 300,
})

train = json.load(open(EXP / "trainsplit_results.json"))
rec = {(t["net"], t["arm"]): t for t in train}

fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
panel = {"CSRNet": axes[0], "MobileCount": axes[1]}
ylims = {"CSRNet": (3.3, 4.3), "MobileCount": (0.605, 0.655)}
# Delta label anchor per (net, arm): (fraction along the line, offset in points)
# MobileCount lines cross mid-panel, so its labels sit before the crossing
DELTA_AT = {("CSRNet", "mc80"): (0.5, 7), ("CSRNet", "csrnet9"): (0.5, 7),
            ("MobileCount", "mc80"): (0.28, -10),
            ("MobileCount", "csrnet9"): (0.28, 9)}

for net, ax in panel.items():
    fmt = "{:.2f}" if net == "CSRNet" else "{:.3f}"
    for arm in ARM_ORDER:
        r = rec[(net, arm)]
        ys = [r["train_mae"], r["val_mae"]]
        c = ARM_COLOR[arm]
        ax.plot([0, 1], ys, color=c, lw=1.8, ls=ARM_LS[arm], zorder=3,
                solid_capstyle="round", dash_capstyle="round")
        ax.plot([0, 1], ys, "o", ms=5.5, mfc=c, mec="white", mew=1.1,
                ls="none", zorder=4)
        ax.annotate(fmt.format(ys[0]), (0, ys[0]),
                    xytext=(-7, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=7.5, color=INK2, zorder=5)
        ax.annotate(f"{ARM_SHORT[arm]}  {fmt.format(ys[1])}", (1, ys[1]),
                    xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=INK, zorder=5)
        xf, dy = DELTA_AT[(net, arm)]
        ax.annotate(f"$\\Delta$ {ys[1] - ys[0]:+.3f}",
                    (xf, ys[0] + (ys[1] - ys[0]) * xf),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", va="bottom" if dy > 0 else "top",
                    fontsize=7, color=INK2, style="italic", zorder=5)
    ax.set_xlim(-0.55, 1.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["train", "validation"], fontsize=8.5, color=INK2)
    ax.set_ylim(*ylims[net])
    ax.set_title(net, loc="left", fontweight="bold")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("MAE of selected checkpoint")
fig.tight_layout(w_pad=1.2)
for ext in ("png", "pdf"):
    fig.savefig(EXP / f"aug_paper_schemes_train_vs_val.{ext}", bbox_inches="tight")
plt.close(fig)
print("written:", *sorted(p.name for p in EXP.glob("aug_paper_schemes_train_vs_val.*")))
