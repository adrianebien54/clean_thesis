#!/usr/bin/env python3
"""Best-val vs end-of-budget validation MAE for every repeat of the cross-resolution sweep.

Companion to eval_bestcombo_repeats_testset.py. Where that script scores the
best-val-MAE checkpoint on the TEST split, this one asks a different question:
how far has validation error drifted by the end of the fixed 800-epoch budget?

Everything is read straight out of the runner's logged tensorboard curves in
bestcombo_repeats_results.json -- no checkpoints are loaded. That matters
because the repeats sweep pruned every checkpoint except the best-val one, so
the final-epoch weights no longer exist on disk; only the scalars survive.

Consequently this is VALIDATION-ONLY. The train-split counterpart lives in
exp/combined_resolution/trainsplit_best_vs_last.json and covers the single
1500-epoch run (see eval_bestcombo_best_vs_last_trainsplit.py), which is the
only sweep that retained latest_state.pth.

Per-rep rows go to best_vs_last_repeats.json; the aggregate (mean +- sample
std over the reps of each cell) goes to best_vs_last_repeats_table.md.

Env overrides:
  SWEEP_DIR sweep to summarise (default ./exp/bestcombo_resolutions_repeats_800ep)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/umrobotics/clean_thesis/crowd_counting")

SWEEP_DIR = ROOT / os.environ.get(
    "SWEEP_DIR", "./exp/bestcombo_resolutions_repeats_800ep")
SWEEP_JSON = SWEEP_DIR / "bestcombo_repeats_results.json"
OUT_JSON = SWEEP_DIR / "best_vs_last_repeats.json"
OUT_TABLE = SWEEP_DIR / "best_vs_last_repeats_table.md"

# (label, width, height) smallest -> largest; also fixes the table row order
RESOLUTIONS = [
    ("49x33", 49, 33),
    ("97x65", 97, 65),
    ("193x130", 193, 130),
    ("386x260", 386, 260),
    ("772x519", 772, 519),
    ("1544x1038", 1544, 1038),
]
PIXELS = {v: w * h for v, w, h in RESOLUTIONS}


def series(record: dict, tag: str) -> dict[int, float]:
    """{step: value} for one tensorboard tag. Step is the 1-based epoch that
    trainer.validate_V1 logs, i.e. the number in the checkpoint filename."""
    return {int(m["step"]): float(m["value"])
            for m in record["metrics"] if m["tag"] == tag}


def best_and_last(record: dict) -> dict:
    mae = series(record, "mae")
    if not mae:
        raise ValueError("no mae metrics in record")
    mse = series(record, "mse")

    best_epoch = min(mae, key=lambda e: mae[e])
    last_epoch = max(mae)
    return {
        "best_epoch": best_epoch,
        "best_val_mae": mae[best_epoch],
        "best_val_mse": mse.get(best_epoch),
        "last_epoch": last_epoch,
        "last_val_mae": mae[last_epoch],
        "last_val_mse": mse.get(last_epoch),
        "ratio": mae[last_epoch] / mae[best_epoch],
        "n_epochs_logged": len(mae),
    }


def _agg(rows: list[dict], key: str, prec: int = 4) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return "--"
    mean = float(np.mean(vals))
    if len(vals) < 2:
        return f"{mean:.{prec}f}"
    return f"{mean:.{prec}f} ± {float(np.std(vals, ddof=1)):.{prec}f}"


def write_table(results: list[dict], path: Path) -> None:
    by: dict[tuple, list[dict]] = {}
    for r in results:
        by.setdefault((r["variant"], r["net"]), []).append(r)

    budget = sorted({r["last_epoch"] for r in results})
    budget_str = "/".join(str(b) for b in budget)

    lines = [
        "# Cross-resolution repeats: VALIDATION MAE at the best epoch vs at the "
        "end of the budget",
        "",
        "Every (net, resolution) cell repeated with IDENTICAL config and seed 3035;",
        "spread = framework nondeterminism (cuDNN kernels, unseeded DataLoader workers).",
        f"Read from the logged val curves; end of budget = epoch {budget_str}.",
        "Validation split only -- the repeats sweep pruned the final-epoch weights, so",
        "no train-split counterpart can be computed from it (see",
        "exp/combined_resolution/trainsplit_best_vs_last.json for the 1500-epoch",
        "single run, which retained latest_state.pth).",
        "std is the sample std (ddof=1), omitted at n=1.",
        "",
        "| Resolution | Pixels | n | CSRNet best | CSRNet last | CSRNet last/best "
        "| n | MobileCount best | MobileCount last | MobileCount last/best |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, w, h in RESOLUTIONS:
        c = by.get((variant, "CSRNet"), [])
        m = by.get((variant, "MobileCount"), [])
        if not c and not m:
            continue
        lines.append(
            f"| {variant} | {w*h:,} "
            f"| {len(c)} | {_agg(c, 'best_val_mae')} | {_agg(c, 'last_val_mae')} "
            f"| {_agg(c, 'ratio', 2)} "
            f"| {len(m)} | {_agg(m, 'best_val_mae')} | {_agg(m, 'last_val_mae')} "
            f"| {_agg(m, 'ratio', 2)} |")
    with path.open("w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    if not SWEEP_JSON.exists():
        sys.exit(f"no sweep results at {SWEEP_JSON}")
    records = [r for r in json.load(SWEEP_JSON.open())
               if isinstance(r, dict) and "error" not in r]

    results: list[dict] = []
    print(f"{'net':<12} {'variant':<11} {'rep':>4} {'best_ep':>8} {'best_mae':>9} "
          f"{'last_ep':>8} {'last_mae':>9} {'x':>6}")
    print("-" * 74)
    for rec in records:
        row = {"variant": rec["variant"], "net": rec["net"], "rep": rec["rep"],
               "pixels": PIXELS[rec["variant"]], "max_epoch": rec["max_epoch"]}
        row.update(best_and_last(rec))
        results.append(row)
        print(f"{row['net']:<12} {row['variant']:<11} {row['rep']:>4} "
              f"{row['best_epoch']:>8} {row['best_val_mae']:>9.4f} "
              f"{row['last_epoch']:>8} {row['last_val_mae']:>9.4f} {row['ratio']:>6.2f}")

    results.sort(key=lambda r: (r["net"], r["pixels"], r["rep"]))
    with OUT_JSON.open("w") as f:
        json.dump(results, f, indent=2)
    write_table(results, OUT_TABLE)
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_TABLE}")


if __name__ == "__main__":
    main()
