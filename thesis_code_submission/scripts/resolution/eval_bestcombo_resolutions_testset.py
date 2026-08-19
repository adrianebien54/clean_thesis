#!/usr/bin/env python3
"""Evaluate each resolution's best-val-MAE checkpoint on the Tenebrio TEST split.

The resolution study (exp/combined_resolution/resolution_table.md) reports the best
*validation* MAE reached during training for each net x resolution. This script takes
the checkpoint from the exact epoch that hit that best val MAE and scores it on the
held-out TEST split, so the resolution curve can be reported on test data.

For every (net, resolution) it:
  1. selects the matching run record from that sweep's results.json,
  2. finds the epoch with the minimum val `mae` in the recorded metrics (== best_mae),
  3. locates that epoch's saved checkpoint (all_ep_<epoch>_mae_*.pth),
  4. loads it and computes count MAE/MSE on <DATA_PATH>/test, replicating
     trainer.validate_V1 exactly (pred_cnt = sum(pred)/LOG_PARA, per-image abs/sq error),
  5. records both the val and test MAE/MSE.

Covers all 6 resolutions x 2 nets (CSRNet AdamW lr4e-5 bs1, MobileCount AdamW lr1e-3 bs6,
AUG=none, seed 3035, LOG_PARA anchored at 772x519). 386x260 checkpoints come from their
original sweeps (see combine_resolution_results.py); the other five from
bestcombo_resolutions_06-24_11-49.

Writes exp/combined_resolution/testset_results.json and testset_table.md.
"""
from __future__ import annotations

import gc
import glob
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

BESTCOMBO_DIR = ROOT / "exp/bestcombo_resolutions_06-24_11-49"
BESTCOMBO_JSON = BESTCOMBO_DIR / "bestcombo_resolutions_results.json"
CSRNET_386_DIR = ROOT / "exp/batchsize_sweep_logpara_06-21_22-41"
CSRNET_386_JSON = CSRNET_386_DIR / "batchsize_logpara_results.json"
MOBILE_386_DIR = ROOT / "exp/360p_lr_optim_grid/mobilecount_logpara2550_anchor772x519_06-17_14-26"
MOBILE_386_JSON = MOBILE_386_DIR / "mobilecount_logpara2550_results.json"

# (label, width, height) smallest -> largest
RESOLUTIONS = [
    ("49x33", 49, 33),
    ("97x65", 97, 65),
    ("193x130", 193, 130),
    ("386x260", 386, 260),
    ("772x519", 772, 519),
    ("1544x1038", 1544, 1038),
]


def _rows(path):
    d = json.load(open(path))
    rows = d if isinstance(d, list) else d.get("runs", [])
    return [r for r in rows if isinstance(r, dict) and "error" not in r]


def select_run(variant: str, net: str) -> dict:
    """Return a spec dict {record, run_dir} for this (variant, net) target."""
    if variant == "386x260" and net == "CSRNet":
        rec = next(r for r in _rows(CSRNET_386_JSON)
                   if r["net"] == "CSRNet" and str(r["variant"]) == "386x260"
                   and int(r["batch_size"]) == 1)
        run_dir = CSRNET_386_DIR / "386x260_CSRNet_AdamW_lr4.0e-05_seed3035_bs1_lp25"
    elif variant == "386x260" and net == "MobileCount":
        rec = next(r for r in _rows(MOBILE_386_JSON)
                   if r["net"] == "MobileCount" and str(r["variant"]) == "386x260"
                   and int(r["batch_size"]) == 6 and r["optimizer"] == "AdamW"
                   and abs(float(r["lr"]) - 1e-3) < 1e-9)
        run_dir = MOBILE_386_DIR / "386x260_MobileCount_AdamW_lr1.0e-03_seed3035_bs6_lp639"
    else:
        rec = next(r for r in _rows(BESTCOMBO_JSON)
                   if r["net"] == net and str(r["variant"]) == variant)
        if net == "CSRNet":
            name = f"{variant}_CSRNet_AdamW_lr4.0e-05_seed3035_bs1"
        else:
            name = f"{variant}_MobileCount_AdamW_lr1.0e-03_seed3035_bs6"
        run_dir = BESTCOMBO_DIR / name
    return {"record": rec, "run_dir": run_dir}


def best_mae_epoch(record: dict) -> tuple[int, float]:
    """(epoch, mae) for the minimum recorded val mae. Epoch is the tensorboard step,
    which trainer.validate_V1 logs as (epoch+1) -- i.e. the 1-based epoch used in the
    checkpoint filename all_ep_<epoch>_..."""
    mae_pts = [(int(m["step"]), float(m["value"]))
               for m in record["metrics"] if m["tag"] == "mae"]
    if not mae_pts:
        raise ValueError("no mae metrics in record")
    epoch, mae = min(mae_pts, key=lambda t: t[1])
    return epoch, mae


def find_checkpoint(run_dir: Path, epoch: int) -> Path:
    hits = sorted(run_dir.glob(f"all_ep_{epoch}_mae_*.pth"))
    if not hits:
        raise FileNotFoundError(f"no checkpoint all_ep_{epoch}_mae_*.pth in {run_dir}")
    return hits[0]


def build_test_loader(variant: str, log_para: float) -> DataLoader:
    data_path = ROOT / f"exp/data/Tenebrio/{variant}"
    mean_std = cfg_data.MEAN_STD
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std),
    ])
    gt_transform = standard_transforms.Compose([own_transforms.LabelNormalize(log_para)])
    test_set = Tenebrio(str(data_path / "test"), "test",
                        main_transform=None,
                        img_transform=img_transform,
                        gt_transform=gt_transform)
    return DataLoader(test_set, batch_size=1, num_workers=8, shuffle=False, drop_last=False)


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
    results = []
    print(f"{'net':<12} {'variant':<11} {'epoch':>6} {'log_para':>10} "
          f"{'val_mae':>9} {'test_mae':>9} {'val_mse':>9} {'test_mse':>9} {'N':>4}")
    print("-" * 92)
    for variant, w, h in RESOLUTIONS:
        for net in ("CSRNet", "MobileCount"):
            spec = select_run(variant, net)
            rec = spec["record"]
            run_dir = spec["run_dir"]
            log_para = float(rec["log_para"])
            epoch, val_mae = best_mae_epoch(rec)
            ckpt = find_checkpoint(run_dir, epoch)
            loader = build_test_loader(variant, log_para)
            test = eval_on_test(net, ckpt, loader, log_para)

            row = {
                "variant": variant, "net": net, "pixels": w * h,
                "best_epoch": epoch, "log_para": log_para,
                "val_mae": val_mae, "val_mse": float(rec["best_mse"]),
                "test_mae": test["mae"], "test_mse": test["mse"],
                "test_n": test["n"],
                "checkpoint": str(ckpt.relative_to(ROOT)),
            }
            results.append(row)
            print(f"{net:<12} {variant:<11} {epoch:>6} {log_para:>10.2f} "
                  f"{val_mae:>9.4f} {test['mae']:>9.4f} {float(rec['best_mse']):>9.4f} "
                  f"{test['mse']:>9.4f} {test['n']:>4}")

    out_dir = ROOT / "exp/combined_resolution"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "testset_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    write_table(results, out_dir / "testset_table.md")
    print(f"\nwrote {out_dir/'testset_results.json'}")
    print(f"wrote {out_dir/'testset_table.md'}")


def write_table(results: list[dict], path: Path) -> None:
    by = {(r["variant"], r["net"]): r for r in results}
    lines = [
        "# Best-epoch TEST-split MAE across input resolutions (Tenebrio, seed 3035, AUG=none)",
        "",
        "Each cell's best-*validation*-MAE checkpoint (the epoch reported in",
        "resolution_table.md) evaluated on the held-out TEST split. CSRNet AdamW lr4e-5 bs1 |",
        "MobileCount AdamW lr1e-3 bs6 | LOG_PARA anchored at 772x519. 386x260 checkpoints",
        "from their original sweeps; all others from bestcombo_resolutions_06-24_11-49.",
        "",
        "| Resolution | Pixels | CSRNet val MAE | CSRNet test MAE | CSRNet test MSE "
        "| MobileCount val MAE | MobileCount test MAE | MobileCount test MSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, w, h in RESOLUTIONS:
        c = by[(variant, "CSRNet")]
        m = by[(variant, "MobileCount")]
        star = " *" if variant == "386x260" else ""
        lines.append(
            f"| {variant}{star} | {w*h:,} | {c['val_mae']:.4f} | {c['test_mae']:.4f} "
            f"| {c['test_mse']:.4f} | {m['val_mae']:.4f} | {m['test_mae']:.4f} "
            f"| {m['test_mse']:.4f} |")
    n_test = by[(RESOLUTIONS[0][0], "CSRNet")]["test_n"]
    lines += [
        "",
        f"Test split: {n_test} images per resolution. \\* 386x260 checkpoints from the "
        "batchsize_logpara (CSRNet) and mobilecount_logpara2550 (MobileCount) sweeps.",
    ]
    with path.open("w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
