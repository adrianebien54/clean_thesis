#!/usr/bin/env python3
"""Overfitting-vs-instability movement map for the resolution study.

One point per (net, resolution): x = train MAE at ep1500 / train MAE at the
best-val checkpoint, y = the same ratio for val MAE (log-log). The diagnostic
categories become regions: upper-left of x=1 -> train kept improving while val
degraded (memorisation); on/near the diagonal y=x -> both splits degraded
together (optimisation drift/instability, which memorisation cannot produce);
distance from (1,1) = severity. Same net colors/markers as mae_vs_memory.py.

Reads exp/combined_resolution/trainsplit_best_vs_last.json (from
scripts/eval_bestcombo_best_vs_last_trainsplit.py); writes
exp/combined_resolution/overfit_movement_map.{png,pdf}.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])
DATA = ROOT / "exp/combined_resolution/trainsplit_best_vs_last.json"
OUT = ROOT / "exp/combined_resolution/overfit_movement_map"

NETS = [("CSRNet", "tab:blue", "o"), ("MobileCount", "tab:red", "s")]
TEXT_MUTED = "#777777"
RES_LABEL_KW = dict(fontsize=8.5, color="#333333", zorder=5)

# (dx, dy) points and horizontal alignment for the resolution tags
LABEL_OFFSETS = {
    ("CSRNet", "49x33"): (8, -3, "left"),
    ("CSRNet", "97x65"): (-8, -6, "right"),
    ("CSRNet", "193x130"): (8, -3, "left"),
    ("CSRNet", "386x260"): (8, -3, "left"),
    ("CSRNet", "772x519"): (8, -3, "left"),
    ("CSRNet", "1544x1038"): (8, -3, "left"),
    ("MobileCount", "49x33"): (8, 2, "left"),
    ("MobileCount", "97x65"): (-8, 2, "right"),
    ("MobileCount", "193x130"): (8, -3, "left"),
    ("MobileCount", "386x260"): (7, -9, "left"),
    ("MobileCount", "772x519"): (-8, 4, "right"),
    ("MobileCount", "1544x1038"): (0, -14, "center"),
}

rows = json.load(DATA.open())
fig, ax = plt.subplots(figsize=(5.8, 4.9))

# reference geometry: no-change boundary and the equal-degradation diagonal
lo, hi = 0.33, 6.0
ax.axvline(1, color="#c9c9c9", lw=0.9, zorder=1)
ax.plot([lo, hi], [lo, hi], color="#b0b0b0", lw=1.1, ls=(0, (5, 3)), zorder=1)
ax.axvspan(lo, 1, color="#000000", alpha=0.045, zorder=0)

for net, color, marker in NETS:
    pts = sorted((r for r in rows if r["net"] == net),
                 key=lambda r: int(r["variant"].split("x")[0]))
    xs = [r["last_train_mae"] / r["best_train_mae"] for r in pts]
    ys = [r["last_val_mae"] / r["best_val_mae"] for r in pts]
    ax.plot(xs, ys, marker, color=color, ms=8, mec="white", mew=1.5,
            ls="none", label=net, zorder=3)
    for r, x, y in zip(pts, xs, ys):
        dx, dy, ha = LABEL_OFFSETS[(net, r["variant"])]
        tag = r["variant"].replace("x", "×")
        if r["variant"] == "386x260":
            tag += "*"
        ax.annotate(tag, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, **RES_LABEL_KW)

# region annotations
ax.annotate("memorisation", xy=(0.52, 2.6), fontsize=10, color=TEXT_MUTED,
            style="italic", ha="center")
ax.annotate("instability", xy=(2.85, 1.6), fontsize=10, color=TEXT_MUTED,
            style="italic", ha="center", rotation=32, rotation_mode="anchor")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lo, hi)
ax.set_ylim(0.85, 4.6)
xticks = [0.5, 1, 2, 4]
yticks = [1, 2, 4]
ax.xaxis.set_major_locator(FixedLocator(xticks))
ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}×" for t in xticks]))
ax.xaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_locator(FixedLocator(yticks))
ax.yaxis.set_major_formatter(FixedFormatter([f"{t:g}×" for t in yticks]))
ax.yaxis.set_minor_locator(NullLocator())
ax.tick_params(colors=TEXT_MUTED, labelsize=9)
ax.set_xlabel("increase in train MAE", fontsize=10.5)
ax.set_ylabel("increase in validation MAE", fontsize=10.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#c3c2b7")
ax.legend(loc="upper left", frameon=False, fontsize=9.5)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("written:", OUT.with_suffix(".png"), OUT.with_suffix(".pdf"), sep="\n  ")
