#!/usr/bin/env python3
"""Consolidate the new-LOG_PARA batch-size sweep into one final table (AdamW only).

bs1/2/4/8 come from the batch sweep (sweep_batchsize_logpara.py); bs6 is the shared
anchor pulled from each net's new-LOG_PARA LR grid under exp/360p_lr_optim_grid/:

  CSRNet      lp25  AdamW@1e-4  exp/360p_lr_optim_grid/csrnet_logpara100_anchor772x519_06-20_14-41/csrnet_logpara_results.json
  MobileCount lp639 AdamW@1e-3  exp/360p_lr_optim_grid/mobilecount_logpara2550_anchor772x519_06-17_14-26/mobilecount_logpara2550_results.json

Writes two files into the latest batch sweep dir (exp/batchsize_sweep_logpara_*):
  batchsize_logpara_merged.json  -- one record per (net, bs, optimizer, lr)
  batchsize_logpara_table.md     -- human-readable per-net tables
"""

from __future__ import annotations

import os
import json
from pathlib import Path

ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])

# latest batch sweep dir (newest by mtime)
SWEEP_DIRS = sorted((p for p in (ROOT / "exp").glob("batchsize_sweep_logpara_*") if p.is_dir()),
                    key=lambda p: p.stat().st_mtime)
if not SWEEP_DIRS:
    raise SystemExit("no exp/batchsize_sweep_logpara_* dir found -- run the sweep first")
OUT_DIR = SWEEP_DIRS[-1]

# (file, filter predicate) -> records to keep from each source.
SOURCES = [
    (str((OUT_DIR / "batchsize_logpara_results.json").relative_to(ROOT)),
     lambda r: True),
    ("exp/360p_lr_optim_grid/csrnet_logpara100_anchor772x519_06-20_14-41/csrnet_logpara_results.json",
     lambda r: r.get("net") == "CSRNet" and r.get("optimizer") == "AdamW"
     and abs(float(r.get("lr", 0)) - 1e-4) < 1e-12 and r.get("batch_size") == 6),
    ("exp/360p_lr_optim_grid/mobilecount_logpara2550_anchor772x519_06-17_14-26/mobilecount_logpara2550_results.json",
     lambda r: r.get("net") == "MobileCount" and r.get("optimizer") == "AdamW"
     and abs(float(r.get("lr", 0)) - 1e-3) < 1e-12 and r.get("batch_size") == 6),
]

FIELDS = ("net", "optimizer", "batch_size", "lr", "log_para", "best_mae", "best_mse", "max_epoch")


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

    out_json = OUT_DIR / "batchsize_logpara_merged.json"
    out_json.write_text(json.dumps(merged, indent=2))

    # markdown: one block per net, mark each net's best MAE
    lines = ["# Batch-size sweep (new LOG_PARA, AdamW only) — consolidated final table",
             "",
             "Tenebrio 386x260, seed 3035, 1500 ep. Each batch size runs at its",
             "sqrt(bs/6)-scaled LR off the net's bs6 optimum. bs6 = new-LOG_PARA LR-grid anchor.",
             "CSRNet lp25 (LOG_PARA 25.03), MobileCount lp639 (LOG_PARA 638.73).",
             ""]
    for net in sorted({r["net"] for r in merged}):
        rows = [r for r in merged if r["net"] == net]
        best = min(rows, key=lambda r: r["best_mae"])
        lp = next((r["log_para"] for r in rows if r.get("log_para") is not None), None)
        lp_str = f" (LOG_PARA {lp:.2f})" if lp is not None else ""
        lines += [f"## {net}{lp_str}", "",
                  "| bs | optimizer | lr | MAE | MSE | source |",
                  "|----|-----------|----|-----|-----|--------|"]
        for r in rows:
            star = " ✅" if r is best else ""
            tag = "LR grid (anchor)" if "aa_final" in r["source"] else "bs sweep"
            lines.append(f"| {r['batch_size']} | {r['optimizer']} | {r['lr']:.1e} | "
                         f"{r['best_mae']:.4f}{star} | {r['best_mse']:.4f} | {tag} |")
        lines += ["", f"**Best {net}: bs{best['batch_size']} {best['optimizer']} "
                  f"@ {best['lr']:.1e} -> MAE {best['best_mae']:.4f}**", ""]

    out_md = OUT_DIR / "batchsize_logpara_table.md"
    out_md.write_text("\n".join(lines))

    print(f"merged {len(merged)} records  (into {OUT_DIR.name})")
    print(f"  -> {out_json}")
    print(f"  -> {out_md}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
