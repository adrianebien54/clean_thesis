#!/usr/bin/env python3
"""Evaluate every repeat of the cross-resolution sweep on the TEST split; report mean +- std.

Companion to run_bestcombo_resolutions_repeats.py: for every completed
(variant, net, rep) cell it loads the best-val-MAE epoch's checkpoint (the one
the runner's pruning kept) and scores it on the held-out TEST split (112 imgs),
replicating trainer.validate_V1 (pred_cnt = sum(pred)/LOG_PARA). Per-rep rows
go to testset_repeats_results.json; the aggregate table (mean +- sample std,
n per cell) goes to testset_repeats_table.md.

Safe to run mid-sweep: only cells present in the runner's results JSON are
evaluated, already-evaluated (variant, net, rep) triples are skipped, and each
row is wrapped in try/except (e.g. CUDA OOM while a big training cell holds the
GPU) so a failed row can simply be retried on the next invocation.

Unlike eval_bestcombo_resolutions_testset.py, val MSE is taken at the best-MAE
epoch (same TB step), not the global best MSE (which can be a different epoch).

Env overrides:
  SWEEP_DIR sweep to evaluate (default ./exp/bestcombo_resolutions_repeats_800ep)
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader

ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import config
config.cfg.GPU_ID = [0]

import misc.transforms as own_transforms
from datasets.Tenebrio.setting import cfg_data
from datasets.Tenebrio.Tenebrio import Tenebrio
from models.CC import CrowdCounter

SWEEP_DIR = ROOT / os.environ.get(
    "SWEEP_DIR", "./exp/bestcombo_resolutions_repeats_800ep")
SWEEP_JSON = SWEEP_DIR / "bestcombo_repeats_results.json"
OUT_JSON = SWEEP_DIR / "testset_repeats_results.json"
OUT_TABLE = SWEEP_DIR / "testset_repeats_table.md"

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


def best_mae_epoch(record: dict) -> tuple[int, float, float]:
    """(epoch, mae, mse_at_that_epoch) for the minimum recorded val mae. Epoch is
    the tensorboard step, which trainer.validate_V1 logs as (epoch+1) -- i.e. the
    1-based epoch used in the checkpoint filename all_ep_<epoch>_..."""
    mae_pts = [(int(m["step"]), float(m["value"]))
               for m in record["metrics"] if m["tag"] == "mae"]
    if not mae_pts:
        raise ValueError("no mae metrics in record")
    epoch, mae = min(mae_pts, key=lambda t: t[1])
    mse = next((float(m["value"]) for m in record["metrics"]
                if m["tag"] == "mse" and int(m["step"]) == epoch), None)
    return epoch, mae, mse


def find_checkpoint(run_dir: Path, epoch: int) -> Path:
    hits = sorted(run_dir.glob(f"all_ep_{epoch}_mae_*.pth"))
    if not hits:
        raise FileNotFoundError(f"no checkpoint all_ep_{epoch}_mae_*.pth in {run_dir}")
    return hits[0]


_loader_cache: dict[tuple, DataLoader] = {}


def build_test_loader(variant: str, log_para: float) -> DataLoader:
    key = (variant, round(log_para, 6))
    if key in _loader_cache:
        return _loader_cache[key]
    data_path = ROOT / f"exp/data/Tenebrio/{variant}"
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD),
    ])
    gt_transform = standard_transforms.Compose([own_transforms.LabelNormalize(log_para)])
    test_set = Tenebrio(str(data_path / "test"), "test",
                        main_transform=None,
                        img_transform=img_transform,
                        gt_transform=gt_transform)
    loader = DataLoader(test_set, batch_size=1, num_workers=8,
                        shuffle=False, drop_last=False)
    _loader_cache[key] = loader
    return loader


@torch.no_grad()
def eval_on_test(net_name: str, ckpt: Path, loader: DataLoader, log_para: float) -> dict:
    net = CrowdCounter(config.cfg.GPU_ID, net_name)
    state = torch.load(str(ckpt), map_location="cuda", weights_only=True)
    net.load_state_dict(state)
    net.cuda()
    net.eval()

    abs_errs, sq_errs = [], []
    for img, gt_map in loader:
        img = img.cuda()
        pred_map = net.test_forward(img)
        pred_cnt = float(pred_map.sum().item()) / log_para
        gt_cnt = float(gt_map.sum().item()) / log_para
        d = gt_cnt - pred_cnt
        abs_errs.append(abs(d))
        sq_errs.append(d * d)

    del net
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "n": len(abs_errs),
        "mae": float(np.mean(abs_errs)),
        "mse": float(math.sqrt(np.mean(sq_errs))),
    }


def main() -> None:
    if not SWEEP_JSON.exists():
        sys.exit(f"no sweep results at {SWEEP_JSON}")
    records = [r for r in json.load(SWEEP_JSON.open())
               if isinstance(r, dict) and "error" not in r]

    results: list[dict] = []
    if OUT_JSON.exists():
        results = json.load(OUT_JSON.open())
    done = {(r["variant"], r["net"], r["rep"]) for r in results}

    print(f"{'net':<12} {'variant':<11} {'rep':>4} {'epoch':>6} {'val_mae':>9} "
          f"{'test_mae':>9} {'test_mse':>9} {'N':>4}")
    print("-" * 72)
    failed = 0
    for rec in records:
        key = (rec["variant"], rec["net"], rec["rep"])
        if key in done:
            continue
        try:
            log_para = float(rec["log_para"])
            epoch, val_mae, val_mse = best_mae_epoch(rec)
            ckpt = find_checkpoint(SWEEP_DIR / rec["exp_name"], epoch)
            loader = build_test_loader(rec["variant"], log_para)
            test = eval_on_test(rec["net"], ckpt, loader, log_para)
        except Exception as e:
            print(f"{rec['net']:<12} {rec['variant']:<11} {rec['rep']:>4} "
                  f"FAILED: {type(e).__name__}: {e}")
            failed += 1
            continue

        row = {
            "variant": rec["variant"], "net": rec["net"], "rep": rec["rep"],
            "pixels": PIXELS[rec["variant"]],
            "best_epoch": epoch, "log_para": log_para,
            "val_mae": val_mae, "val_mse": val_mse,
            "test_mae": test["mae"], "test_mse": test["mse"], "test_n": test["n"],
            "checkpoint": str(ckpt.relative_to(ROOT)),
        }
        results.append(row)
        with OUT_JSON.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"{rec['net']:<12} {rec['variant']:<11} {rec['rep']:>4} {epoch:>6} "
              f"{val_mae:>9.4f} {test['mae']:>9.4f} {test['mse']:>9.4f} {test['n']:>4}")

    write_table(results, OUT_TABLE)
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_TABLE}")
    if failed:
        print(f"{failed} cell(s) FAILED -- rerun this script to retry them")


def _agg(rows: list[dict], key: str) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return "--"
    mean = float(np.mean(vals))
    if len(vals) < 2:
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {float(np.std(vals, ddof=1)):.4f}"


def write_table(results: list[dict], path: Path) -> None:
    by: dict[tuple, list[dict]] = {}
    for r in results:
        by.setdefault((r["variant"], r["net"]), []).append(r)

    lines = [
        "# Cross-resolution repeats: TEST-split MAE, mean ± std across same-seed runs",
        "",
        "Every (net, resolution) cell repeated with IDENTICAL config and seed 3035;",
        "spread = framework nondeterminism (cuDNN kernels, unseeded DataLoader workers).",
        "800 epochs, best-val-MAE checkpoint per rep evaluated on the held-out TEST",
        "split. CSRNet AdamW lr4e-5 bs1 | MobileCount AdamW lr1e-3 bs6 | AUG=none |",
        "LOG_PARA anchored at 772x519. std is the sample std (ddof=1), omitted at n=1.",
        "",
        "| Resolution | Pixels | n | CSRNet val MAE | CSRNet test MAE | CSRNet test MSE "
        "| n | MobileCount val MAE | MobileCount test MAE | MobileCount test MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, w, h in RESOLUTIONS:
        c = by.get((variant, "CSRNet"), [])
        m = by.get((variant, "MobileCount"), [])
        if not c and not m:
            continue
        lines.append(
            f"| {variant} | {w*h:,} "
            f"| {len(c)} | {_agg(c, 'val_mae')} | {_agg(c, 'test_mae')} | {_agg(c, 'test_mse')} "
            f"| {len(m)} | {_agg(m, 'val_mae')} | {_agg(m, 'test_mae')} | {_agg(m, 'test_mse')} |")
    n_test = results[0]["test_n"] if results else 0
    lines += ["", f"Test split: {n_test} images per resolution."]
    with path.open("w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
