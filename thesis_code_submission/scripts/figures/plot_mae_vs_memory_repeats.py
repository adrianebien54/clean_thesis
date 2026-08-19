#!/usr/bin/env python3
"""Accuracy-vs-memory trade-off figure built from the same-seed repeat sweep.

Same layout as plot_mae_vs_memory.py (log-log connected scatter, one point per input
resolution, one line per net, lower-left is better on both axes), but each point is the
mean test MAE across the same-seed repeats (seed 3035, 800 epochs) rather than a single
run. The per-cell spread is not drawn; it is kept in the JSON/table companions as
test_mae_std / min / max.

Peak memory is *reused*, not re-measured: measure_inference_memory.py feeds a
torch.randn(1, 3, H, W) tensor through a fixed architecture, so the recorded peak depends
only on (net, resolution) and is identical for every repeat of a cell. Only peak_mb /
weights_mb / params_m are taken from that file; its test MAEs belong to the single
1500-epoch run and are deliberately dropped here.

Reads $SWEEP_DIR/testset_repeats_results.json (per-rep rows written by
eval_bestcombo_repeats_testset.py) and exp/combined_resolution/inference_memory.json.
Writes $SWEEP_DIR/mae_vs_memory_repeats.png/.pdf plus inference_memory_repeats.json and
inference_memory_repeats_table.md.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])
SWEEP_DIR = ROOT / os.environ.get(
    "SWEEP_DIR", "./exp/bestcombo_resolutions_repeats_800ep")
DATA = SWEEP_DIR / "testset_repeats_results.json"
MEMORY_JSON = ROOT / "exp/combined_resolution/inference_memory.json"
MEMORY_SOURCE = "exp/combined_resolution/inference_memory.json"
OUT = SWEEP_DIR / "mae_vs_memory_repeats"
OUT_JSON = SWEEP_DIR / "inference_memory_repeats.json"
OUT_TABLE = SWEEP_DIR / "inference_memory_repeats_table.md"

NETS = [("CSRNet", "tab:blue", "o"), ("MobileCount", "tab:red", "s")]
TEXT_MUTED = "#777777"
RES_LABEL_KW = dict(fontsize=9, color="#333333")

# (dx, dy) in points, per (net, variant) -- keeps the resolution tags clear of the
# lines and of each other. Retuned relative to plot_mae_vs_memory.py: the repeat means
# move the two low-resolution MobileCount points close together (2.71 and 2.55 MAE at
# 13.4 and 13.9 MB), so they are split above/below.
LABEL_OFFSETS = {
    ("CSRNet", "49x33"): (10, 0),
    ("CSRNet", "97x65"): (10, 2),
    ("CSRNet", "193x130"): (9, 3),
    ("CSRNet", "386x260"): (8, 4),
    ("CSRNet", "772x519"): (6, 7),
    ("CSRNet", "1544x1038"): (-6, 9),
    ("MobileCount", "49x33"): (8, 6),
    ("MobileCount", "97x65"): (8, -12),
    ("MobileCount", "193x130"): (8, -3),
    ("MobileCount", "386x260"): (6, -13),
    ("MobileCount", "772x519"): (7, 4),
    ("MobileCount", "1544x1038"): (5, -13),
}


def build_rows() -> list[dict]:
    """Join per-rep test MAEs with the per-cell memory measurements."""
    reps = json.load(DATA.open())
    memory = {(r["net"], r["variant"]): r for r in json.load(MEMORY_JSON.open())}

    by: dict[tuple, list[dict]] = {}
    for r in reps:
        by.setdefault((r["net"], r["variant"]), []).append(r)

    missing = sorted(k for k in by if k not in memory)
    if missing:
        raise KeyError(f"no memory measurement for {missing} in {MEMORY_SOURCE}")

    rows = []
    for (net, variant), rs in by.items():
        maes = [r["test_mae"] for r in rs]
        m = memory[(net, variant)]
        rows.append({
            "variant": variant, "net": net, "pixels": rs[0]["pixels"],
            "peak_mb": m["peak_mb"], "weights_mb": m["weights_mb"],
            "params_m": m["params_m"],
            "test_mae_mean": float(np.mean(maes)),
            "test_mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
            "test_mae_min": min(maes), "test_mae_max": max(maes),
            "n_reps": len(rs),
            "memory_source": MEMORY_SOURCE,
        })
    rows.sort(key=lambda r: (r["net"], r["pixels"]))
    return rows


def write_table(rows: list[dict]) -> None:
    by = {(r["variant"], r["net"]): r for r in rows}
    variants = sorted({r["variant"] for r in rows},
                      key=lambda v: next(r["pixels"] for r in rows if r["variant"] == v))
    n_reps = sorted({r["n_reps"] for r in rows})
    n_txt = "/".join(str(n) for n in n_reps)
    lines = [
        "# Accuracy vs peak inference memory (same-seed repeat sweep, 800 epochs)",
        "",
        f"Test MAE is the mean +- 1 sample std over n={n_txt} same-seed (3035) repeats",
        "per cell. Peak memory is the batch-1 max_memory_allocated during one warmed-up",
        f"test_forward, reused from {MEMORY_SOURCE}: it depends only on the network and",
        "the input shape, so it is identical across repeats of a cell.",
        "",
        "| Resolution | CSRNet peak MB | CSRNet test MAE | MobileCount peak MB | MobileCount test MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for v in variants:
        c, m = by[(v, "CSRNet")], by[(v, "MobileCount")]
        lines.append(
            f"| {v} | {c['peak_mb']:.1f} | {c['test_mae_mean']:.4f} +- {c['test_mae_std']:.4f} "
            f"| {m['peak_mb']:.1f} | {m['test_mae_mean']:.4f} +- {m['test_mae_std']:.4f} |")
    c, m = by[(variants[0], "CSRNet")], by[(variants[0], "MobileCount")]
    lines += ["",
              f"Model weights: CSRNet {c['params_m']:.2f} M params ({c['weights_mb']:.1f} MB), "
              f"MobileCount {m['params_m']:.2f} M params ({m['weights_mb']:.1f} MB)."]
    with OUT_TABLE.open("w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    rows = build_rows()
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for net, color, marker in NETS:
        pts = sorted((r for r in rows if r["net"] == net), key=lambda r: r["pixels"])
        xs = [r["peak_mb"] for r in pts]
        ys = [r["test_mae_mean"] for r in pts]
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=8,
                mec="white", mew=1.5, solid_capstyle="round",
                label=net, zorder=3)
        for r in pts:
            dx, dy = LABEL_OFFSETS[(net, r["variant"])]
            ax.annotate(r["variant"].replace("x", "×"),
                        (r["peak_mb"], r["test_mae_mean"]),
                        textcoords="offset points", xytext=(dx, dy),
                        zorder=4, **RES_LABEL_KW)

    ax.set_xscale("log")
    ax.set_yscale("log")
    xticks = [25, 50, 100, 200, 500, 1000, 2000]
    yticks = [0.25, 0.5, 1, 2, 5, 7]
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

    with OUT_JSON.open("w") as f:
        json.dump(rows, f, indent=2)
    write_table(rows)

    print(f"{'net':<12} {'variant':<11} {'peak_MB':>9} {'test_MAE':>18} {'n':>3}")
    print("-" * 57)
    for r in rows:
        mae = f"{r['test_mae_mean']:.4f} ± {r['test_mae_std']:.4f}"
        print(f"{r['net']:<12} {r['variant']:<11} {r['peak_mb']:>9.1f} {mae:>18} "
              f"{r['n_reps']:>3}")
    print(f"\nwrote {OUT}.png and .pdf")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
