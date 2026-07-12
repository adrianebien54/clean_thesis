#!/usr/bin/env python3
"""Figures for the paper-augmentation-schemes experiment (thesis, print-safe).

Fig 1 (aug_paper_schemes_val_curves): best-so-far validation MAE over training,
    2 panels (CSRNet | MobileCount), 3 arms. The running minimum suppresses the
    per-epoch validation noise and shows when each arm stops improving.
Fig 2 (aug_paper_schemes_generalization): train -> val -> test slope chart of
    the selected checkpoints. A rising line = memorization of the training
    images; the no-aug MobileCount line crosses both others (ranking flip).

Reads exp/aug_paper_schemes_386x260_07-04/{aug_paper_schemes_results.json,
testset_results.json,trainsplit_results.json} (the last one from
scripts/eval_aug_paper_schemes_trainsplit.py); writes PNG (300 dpi) + PDF
next to them.
"""
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

EXP = Path("/home/umrobotics/clean_thesis/crowd_counting/exp/aug_paper_schemes_386x260_07-04")

# --- palette (dataviz reference instance, light mode, slots 1-3 in fixed order)
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
ARM_ORDER = ["noaug", "mc80", "csrnet9"]
ARM_LABEL = {"noaug": "no augmentation",
             "mc80": "online 80% crop (mc80)",
             "csrnet9": "offline 18-patch (csrnet9)"}
ARM_SHORT = {"noaug": "no aug.", "mc80": "mc80", "csrnet9": "csrnet9"}
ARM_COLOR = {"noaug": "#2a78d6", "mc80": "#1baf7a", "csrnet9": "#eda100"}
ARM_LS    = {"noaug": "-", "mc80": (0, (5, 1.6)), "csrnet9": (0, (1, 1.2))}

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

runs = json.load(open(EXP / "aug_paper_schemes_results.json"))
test = json.load(open(EXP / "testset_results.json"))
train = json.load(open(EXP / "trainsplit_results.json"))
by = {(r["net"], r["arm"]): r for r in runs}
trec = {(t["net"], t["arm"]): t for t in test}
trn = {(t["net"], t["arm"]): t["train_mae"] for t in train}


def mae_series(rec):
    return np.asarray([x["value"] for x in rec["metrics"] if x["tag"] == "mae"])


# ------------------------------------------- Fig 1: best-so-far staircases
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
panel = {"CSRNet": axes[0], "MobileCount": axes[1]}
ylims = {"CSRNet": (3.25, 6.3), "MobileCount": (0.5, 2.1)}
# per-(net, arm) offsets (points) for the "ep N" best-epoch annotation
ANN = {("CSRNet", "noaug"): (-2, -10), ("CSRNet", "mc80"): (14, -10),
       ("CSRNet", "csrnet9"): (0, 8),
       ("MobileCount", "noaug"): (2, -10), ("MobileCount", "mc80"): (14, 7),
       ("MobileCount", "csrnet9"): (-14, 7)}

for net, ax in panel.items():
    for arm in ARM_ORDER:
        y = mae_series(by[(net, arm)])
        x = np.arange(1, len(y) + 1)
        c = ARM_COLOR[arm]
        ax.plot(x, np.minimum.accumulate(y), color=c, lw=1.6, ls=ARM_LS[arm],
                label=ARM_LABEL[arm], zorder=3,
                solid_capstyle="round", dash_capstyle="round")
        bi = int(np.argmin(y))
        ax.plot(x[bi], y[bi], "o", ms=5, mfc=c, mec="white", mew=1.1, zorder=4)
        dx, dy = ANN[(net, arm)]
        ax.annotate(f"ep {x[bi]}", (x[bi], y[bi]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="bottom" if dy > 0 else "top",
                    fontsize=7, color=INK2, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(1, 1500)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1000"])
    ax.set_ylim(*ylims[net])
    ax.set_title(net, loc="left", fontweight="bold")
    ax.set_xlabel("epoch (log scale)")
    ax.grid(True, axis="y")
    ax.tick_params(length=2.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel("best validation MAE so far")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncols=3, handlelength=2.4,
           bbox_to_anchor=(0.5, 1.02), columnspacing=1.4)
fig.tight_layout(w_pad=2.0, rect=(0, 0, 1, 0.92))
for ext in ("png", "pdf"):
    fig.savefig(EXP / f"aug_paper_schemes_val_curves.{ext}", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------ Fig 2: train -> val -> test slopes
fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
panel = {"CSRNet": axes[0], "MobileCount": axes[1]}
ylims = {"CSRNet": (2.95, 4.25), "MobileCount": (0.42, 0.80)}
# vertical dodge (points) for the left-hand (train) value labels, per (net, arm)
DODGE = {("CSRNet", "noaug"): 3, ("CSRNet", "mc80"): -3, ("CSRNet", "csrnet9"): 0,
         ("MobileCount", "noaug"): 0, ("MobileCount", "mc80"): -4,
         ("MobileCount", "csrnet9"): 4}

for net, ax in panel.items():
    fmt = "{:.2f}" if net == "CSRNet" else "{:.3f}"
    for arm in ARM_ORDER:
        t = trec[(net, arm)]
        ys = [trn[(net, arm)], t["val_mae"], t["test_mae"]]
        c = ARM_COLOR[arm]
        ax.plot([0, 1, 2], ys, color=c, lw=1.8, ls=ARM_LS[arm], zorder=3,
                solid_capstyle="round", dash_capstyle="round")
        ax.plot([0, 1, 2], ys, "o", ms=5.5, mfc=c, mec="white", mew=1.1,
                ls="none", zorder=4)
        ax.annotate(fmt.format(ys[0]), (0, ys[0]),
                    xytext=(-7, DODGE[(net, arm)]),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7.5, color=INK2, zorder=5)
        ax.annotate(f"{ARM_SHORT[arm]}  {fmt.format(ys[2])}", (2, ys[2]),
                    xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=INK, zorder=5)
    if net == "MobileCount":  # the headline: no-aug collapses on unseen data
        v = trec[(net, "noaug")]["val_mae"]
        ax.annotate("+64%\ntrain → test", (0.5, (trn[(net, "noaug")] + v) / 2),
                    xytext=(14, -6), textcoords="offset points",
                    ha="left", va="top", fontsize=7.5, color=INK2, style="italic")
    ax.set_xlim(-0.6, 3.15)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["train", "validation", "test"], fontsize=8.5, color=INK2)
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
    fig.savefig(EXP / f"aug_paper_schemes_generalization.{ext}", bbox_inches="tight")
plt.close(fig)
print("written:", *[p.name for p in sorted(EXP.glob('aug_paper_schemes_val_curves.*'))
                    + sorted(EXP.glob('aug_paper_schemes_generalization.*'))])
