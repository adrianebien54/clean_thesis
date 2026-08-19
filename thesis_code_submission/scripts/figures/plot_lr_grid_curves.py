#!/usr/bin/env python3
"""Plot val-MAE curves for the full optimizer x LR grid to expose overfitting.

Reads the grid result JSONs (lr_grid_results.json + lr_grid_continuations_results.json),
dedupes each {net, optimizer, lr} cell to its longest run, and produces one PNG per
architecture: a 2x2 panel (one per LR) overlaying Adam vs AdamW val MAE. Raw curve is
faint, a rolling median is bold, and a star marks the best checkpoint. The curve rising
after the star is the overfit signal; a dotted line marks the best-MAE floor.

Style follows plot_seed_sweep_curves.py.
"""

import json
import math
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "exp" / "lr_grid_06-12_20-16"
# Auto-discover every grid results file (base grid + high-LR edge + continuations).
RESULT_FILES = sorted((ROOT / "exp").glob("lr_grid*/lr_grid_results.json")) + [
    ROOT / "exp" / "lr_grid_continuations_results.json",
]

COLORS = {"Adam": "tab:blue", "AdamW": "tab:orange"}
SMOOTH_WINDOW = 51  # epochs, rolling median
WARMUP = 30  # epochs ignored when auto-scaling the y-axis


def rolling_median(values, window):
    if len(values) < window:
        return values.copy()
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    return np.array([np.median(padded[i:i + window]) for i in range(len(values))])


def mae_series(run):
    pts = sorted((m["step"], m["value"]) for m in run["metrics"] if m["tag"] == "mae")
    return np.array([s for s, _ in pts]), np.array([v for _, v in pts])


def load_cells():
    """Return {(net, optimizer, lr): run}, keeping the longest run per cell."""
    cells = {}
    for f in RESULT_FILES:
        for r in json.loads(Path(f).read_text()):
            if "error" in r:
                continue
            key = (r["net"], r["optimizer"], r["lr"])
            if key not in cells or r["max_epoch"] > cells[key]["max_epoch"]:
                cells[key] = r
    return cells


def lr_label(lr):
    return f"{lr:.0e}".replace("e-0", "e-")


def plot_net(net, cells, out_dir):
    lrs = sorted({lr for (n, _, lr) in cells if n == net}, reverse=True)
    ncols = 2 if len(lrs) <= 4 else 3
    nrows = math.ceil(len(lrs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows),
                             squeeze=False)
    sample = next(r for (n, _, _), r in cells.items() if n == net)
    fig.suptitle(
        f"{net} on Tenebrio {sample['variant']} — val MAE vs epoch, Adam vs AdamW "
        f"(step γ=0.995, bs {sample['batch_size']}, seed {sample['seed']}, 1500 ep)",
        fontsize=13,
    )

    flat = list(axes.flat)
    for ax in flat[len(lrs):]:  # hide unused panels
        ax.set_visible(False)
    for ax, lr in zip(flat, lrs):
        lo_candidates, hi_candidates = [], []
        for opt in ("Adam", "AdamW"):
            run = cells.get((net, opt, lr))
            if run is None:
                continue
            steps, vals = mae_series(run)
            color = COLORS[opt]
            smooth = rolling_median(vals, SMOOTH_WINDOW)
            ax.plot(steps, vals, color=color, linewidth=0.6, alpha=0.18)
            ax.plot(steps, smooth, color=color, linewidth=1.8,
                    label=f"{opt} ({SMOOTH_WINDOW}-ep rolling median)")

            best_idx = int(np.argmin(vals))
            best_ep, best_mae = int(steps[best_idx]), float(vals[best_idx])
            final_mae = float(vals[-1])
            ax.plot(best_ep, best_mae, marker="*", markersize=15,
                    markeredgecolor="black", markerfacecolor=color, linestyle="none",
                    label=(f"{opt} best {best_mae:.3f} @ ep {best_ep} "
                           f"(final {final_mae:.3f}, Δ{final_mae - best_mae:+.3f})"))
            ax.axhline(best_mae, color=color, linewidth=0.8, linestyle=":", alpha=0.7)

            warm = smooth[WARMUP:] if len(smooth) > WARMUP else smooth
            lo_candidates.append(float(np.min(vals)))
            hi_candidates.append(float(np.max(warm)))

        ax.set_title(f"lr = {lr_label(lr)}", fontsize=11)
        ax.set_xlabel("epoch")
        ax.set_ylabel("val MAE")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.legend(fontsize=7.5, loc="upper right")
        if lo_candidates:
            lo, hi = min(lo_candidates), max(hi_candidates)
            ax.set_ylim(max(0.0, lo - 0.05 * (hi - lo + 1e-6)), hi + 0.05 * (hi - lo + 1e-6))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / f"{net.lower()}_lr_grid_overfit.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    cells = load_cells()
    nets = sorted({n for (n, _, _) in cells})
    for net in nets:
        plot_net(net, cells, GRID_DIR)


if __name__ == "__main__":
    main()
