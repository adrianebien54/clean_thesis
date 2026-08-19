#!/usr/bin/env python3
"""Consolidate the batch-size sweep into one final table.

The batch-size results are spread across three sources because the sweep was
relaunched and because bs6 is the shared anchor from the LR grid:

  1. CSRNet bs1 AdamW@4e-5  -- finished in the first (aborted) attempt
       exp/batchsize_sweep_06-13_23-09/batchsize_sweep_results.json
  2. The other 10 runs (bs2/4/8 + all MobileCount) -- the 06-15 relaunch
       exp/batchsize_sweep_06-15_13-52/batchsize_sweep_results.json
  3. bs6 anchors (each net's bs6 optimum) -- pulled from the LR grid
       CSRNet      AdamW@1e-4  exp/lr_grid_06-12_20-16/lr_grid_results.json
       MobileCount AdamW@1e-3  exp/lr_grid_continuations_results.json

Writes two files into the 06-15 sweep dir:
  batchsize_final_merged.json  -- one record per (net, bs, optimizer, lr)
  batchsize_final_table.md     -- human-readable per-net tables
"""

from __future__ import annotations

import os
import json
from pathlib import Path

ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])
OUT_DIR = ROOT / "exp/batchsize_sweep_06-15_13-52"

# (file, filter predicate) -> records to keep from each source.
SOURCES = [
    ("exp/batchsize_sweep_06-13_23-09/batchsize_sweep_results.json",
     lambda r: r.get("net") == "CSRNet" and r.get("batch_size") == 1),
    ("exp/batchsize_sweep_06-15_13-52/batchsize_sweep_results.json",
     lambda r: True),
    ("exp/lr_grid_06-12_20-16/lr_grid_results.json",
     lambda r: r.get("net") == "CSRNet" and r.get("optimizer") == "AdamW"
     and abs(float(r.get("lr", 0)) - 1e-4) < 1e-12 and r.get("batch_size") == 6),
    ("exp/lr_grid_continuations_results.json",
     lambda r: r.get("net") == "MobileCount" and r.get("optimizer") == "AdamW"
     and abs(float(r.get("lr", 0)) - 1e-3) < 1e-12 and r.get("batch_size") == 6),
]

FIELDS = ("net", "optimizer", "batch_size", "lr", "best_mae", "best_mse", "max_epoch")


def load(path: str) -> list[dict]:
    d = json.load((ROOT / path).open())
    return d if isinstance(d, list) else d.get("runs", d)


def main() -> None:
    merged: list[dict] = []
    for path, keep in SOURCES:
        for r in load(path):
            if not isinstance(r, dict) or "error" in r or not keep(r):
                continue
            rec = {k: r.get(k) for k in FIELDS}
            rec["source"] = path
            merged.append(rec)

    # sort: net, then batch size, then optimizer
    merged.sort(key=lambda r: (r["net"], r["batch_size"], r["optimizer"]))

    out_json = OUT_DIR / "batchsize_final_merged.json"
    out_json.write_text(json.dumps(merged, indent=2))

    # markdown: one block per net, mark each net's best MAE
    lines = ["# Batch-size sweep — consolidated final table",
             "",
             "Tenebrio 386x260, seed 3035, 1500 ep. Each batch size runs at its",
             "sqrt(bs/6)-scaled LR off the net's bs6 optimum. bs6 = LR-grid anchor.",
             ""]
    for net in sorted({r["net"] for r in merged}):
        rows = [r for r in merged if r["net"] == net]
        best = min(rows, key=lambda r: r["best_mae"])
        lines += [f"## {net}", "",
                  "| bs | optimizer | lr | MAE | MSE | source |",
                  "|----|-----------|----|-----|-----|--------|"]
        for r in rows:
            star = " ✅" if r is best else ""
            tag = "LR grid (anchor)" if "lr_grid" in r["source"] else "bs sweep"
            lines.append(f"| {r['batch_size']} | {r['optimizer']} | {r['lr']:.1e} | "
                         f"{r['best_mae']:.4f}{star} | {r['best_mse']:.4f} | {tag} |")
        lines += ["", f"**Best {net}: bs{best['batch_size']} {best['optimizer']} "
                  f"@ {best['lr']:.1e} -> MAE {best['best_mae']:.4f}**", ""]

    out_md = OUT_DIR / "batchsize_final_table.md"
    out_md.write_text("\n".join(lines))

    print(f"merged {len(merged)} records")
    print(f"  -> {out_json}")
    print(f"  -> {out_md}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
