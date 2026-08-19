#!/usr/bin/env python3
"""Accuracy-vs-memory trade-off figure: test MAE against peak inference GPU memory.

Connected scatter, one point per input resolution (smallest -> largest along each
line), one line per net. Both axes log: memory spans 27 MB -> 1.7 GB and MAE
0.23 -> 5.5, so linear axes would collapse most points into a corner. Lower-left
is better on both axes.

Reads exp/combined_resolution/inference_memory.json (from measure_inference_memory.py,
which pairs testset_results.json MAEs with measured batch-1 forward-pass peaks).
Writes exp/combined_resolution/mae_vs_memory.png and .pdf.
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])
DATA = ROOT / "exp/combined_resolution/inference_memory.json"
OUT = ROOT / "exp/combined_resolution/mae_vs_memory"

NETS = [("CSRNet", "tab:blue", "o"), ("MobileCount", "tab:red", "s")]
TEXT_MUTED = "#777777"
RES_LABEL_KW = dict(fontsize=9, color="#333333")

# (dx, dy) in points, per (net, variant) -- keeps the resolution tags clear of
# the lines and of each other.
LABEL_OFFSETS = {
    ("CSRNet", "49x33"): (8, 3),
    ("CSRNet", "97x65"): (8, 3),
    ("CSRNet", "193x130"): (8, 3),
    ("CSRNet", "386x260"): (8, 3),
    ("CSRNet", "772x519"): (6, 6),
    ("CSRNet", "1544x1038"): (-4, 8),
    ("MobileCount", "49x33"): (7, -2),
    ("MobileCount", "97x65"): (7, -2),
    ("MobileCount", "193x130"): (7, -2),
    ("MobileCount", "386x260"): (5, -11),
    ("MobileCount", "772x519"): (5, 5),
    ("MobileCount", "1544x1038"): (5, -11),
}


def main() -> None:
    rows = json.load(DATA.open())
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for net, color, marker in NETS:
        pts = sorted((r for r in rows if r["net"] == net), key=lambda r: r["pixels"])
        xs = [r["peak_mb"] for r in pts]
        ys = [r["test_mae"] for r in pts]
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=8,
                mec="white", mew=1.5, solid_capstyle="round",
                label=net, zorder=3)
        for r in pts:
            dx, dy = LABEL_OFFSETS[(net, r["variant"])]
            ax.annotate(r["variant"].replace("x", "×"),
                        (r["peak_mb"], r["test_mae"]),
                        textcoords="offset points", xytext=(dx, dy),
                        zorder=4, **RES_LABEL_KW)

    ax.set_xscale("log")
    ax.set_yscale("log")
    xticks = [25, 50, 100, 200, 500, 1000, 2000]
    yticks = [0.25, 0.5, 1, 2, 5]
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:,}" for t in xticks]))
    ax.yaxis.set_major_locator(FixedLocator(yticks))
    ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in yticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

    ax.grid(True, which="major", color="#e0e0e0", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bbbbbb")
    ax.tick_params(colors="#555555", labelsize=9)

    ax.set_xlabel("Peak GPU memory during inference in MB (log scale)", fontsize=10)
    ax.set_ylabel("Test MAE (log scale)", fontsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.annotate("better", xy=(0.035, 0.06), xycoords="axes fraction",
                fontsize=9, color=TEXT_MUTED, style="italic")
    ax.annotate("", xy=(0.012, 0.025), xytext=(0.075, 0.10), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color=TEXT_MUTED, lw=1))

    fig.tight_layout()
    fig.savefig(f"{OUT}.png", dpi=200)
    fig.savefig(f"{OUT}.pdf")
    print(f"wrote {OUT}.png and .pdf")


if __name__ == "__main__":
    main()
