#!/usr/bin/env python3
"""Evaluate the paper-aug-scheme runs' best-val-MAE checkpoints on the Tenebrio TEST split.

Companion to scripts/run_aug_paper_schemes_386x260.py. For each of the 6 runs
(2 nets x {noaug, mc80, csrnet9}) it:
  1. reads the run record from <SWEEP_DIR>/aug_paper_schemes_results.json,
  2. finds the epoch with the minimum recorded val `mae` (== best_mae),
  3. locates that epoch's checkpoint (all_ep_<epoch>_mae_*.pth) in the run dir,
  4. scores it on exp/data/Tenebrio/386x260/test (112 full images, no augmentation),
     replicating trainer.validate_V1 (pred_cnt = sum(pred)/LOG_PARA),
  5. records val and test MAE/MSE.

All arms evaluate on the SAME full-image test split — including csrnet9, whose training
patches share the parent resolution's LOG_PARA (asserted here against the run record).

Usage: SWEEP_DIR=./exp/aug_paper_schemes_386x260_<stamp> .venv/bin/python \
           scripts/eval_aug_paper_schemes_testset.py
   or: .venv/bin/python scripts/eval_aug_paper_schemes_testset.py <sweep_dir>

Writes <SWEEP_DIR>/testset_results.json and <SWEEP_DIR>/testset_table.md.
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

ROOT = Path("/home/umrobotics/clean_thesis/crowd_counting")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import config
config.cfg.GPU_ID = [0]

import misc.transforms as own_transforms
from datasets.Tenebrio.setting import cfg_data
from datasets.Tenebrio.Tenebrio import Tenebrio
from models.CC import CrowdCounter

VARIANT = "386x260"
SEED = 3035
ANCHOR = 772 * 519
EXPECTED_LOG_PARA = {"CSRNet": 100.0 * 386 * 260 / ANCHOR,
                     "MobileCount": 2550.0 * 386 * 260 / ANCHOR}
ARMS = ["noaug", "mc80", "csrnet9"]
NETS = ["CSRNet", "MobileCount"]


def sweep_dir() -> Path:
    if len(sys.argv) > 1:
        d = Path(sys.argv[1])
    elif os.environ.get("SWEEP_DIR"):
        d = Path(os.environ["SWEEP_DIR"])
    else:
        sys.exit("usage: eval_aug_paper_schemes_testset.py <sweep_dir>  (or set SWEEP_DIR)")
    d = d if d.is_absolute() else ROOT / d
    if not (d / "aug_paper_schemes_results.json").exists():
        sys.exit(f"no aug_paper_schemes_results.json in {d}")
    return d


def select_run(rows: list[dict], net: str, arm: str) -> dict:
    hits = [r for r in rows if r.get("net") == net and r.get("arm") == arm
            and "error" not in r]
    if not hits:
        raise KeyError(f"no completed record for {net}/{arm}")
    return hits[-1]  # latest wins if a cell was re-run


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


def build_test_loader(log_para: float) -> DataLoader:
    data_path = ROOT / f"exp/data/Tenebrio/{VARIANT}"
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
    sdir = sweep_dir()
    rows = json.load((sdir / "aug_paper_schemes_results.json").open())

    results = []
    print(f"{'net':<12} {'arm':<9} {'epoch':>6} {'log_para':>10} "
          f"{'val_mae':>9} {'test_mae':>9} {'val_mse':>9} {'test_mse':>9} {'N':>4}")
    print("-" * 88)
    for net in NETS:
        for arm in ARMS:
            rec = select_run(rows, net, arm)
            log_para = float(rec["log_para"])
            expected = EXPECTED_LOG_PARA[net]
            assert abs(log_para - expected) < 0.01, (
                f"{net}/{arm}: record log_para {log_para:.4f} != expected {expected:.4f}")
            run_dir = sdir / (f"{VARIANT}_{net}_Adam_lr{float(rec['lr']):.1e}"
                              f"_seed{SEED}_bs6_{arm}")
            epoch, val_mae = best_mae_epoch(rec)
            ckpt = find_checkpoint(run_dir, epoch)
            test = eval_on_test(net, ckpt, build_test_loader(log_para), log_para)

            row = {
                "net": net, "arm": arm,
                "best_epoch": epoch, "log_para": log_para,
                "val_mae": val_mae, "val_mse": float(rec["best_mse"]),
                "test_mae": test["mae"], "test_mse": test["mse"],
                "test_n": test["n"],
                "checkpoint": str(ckpt.relative_to(ROOT)),
            }
            results.append(row)
            print(f"{net:<12} {arm:<9} {epoch:>6} {log_para:>10.2f} "
                  f"{val_mae:>9.4f} {test['mae']:>9.4f} {float(rec['best_mse']):>9.4f} "
                  f"{test['mse']:>9.4f} {test['n']:>4}")

    with (sdir / "testset_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    write_table(results, sdir / "testset_table.md")
    print(f"\nwrote {sdir/'testset_results.json'}")
    print(f"wrote {sdir/'testset_table.md'}")


def write_table(results: list[dict], path: Path) -> None:
    by = {(r["net"], r["arm"]): r for r in results}
    arm_label = {"noaug": "no augmentation", "mc80": "mc80 (online 80% crop)",
                 "csrnet9": "csrnet9 (offline 18-patch)"}
    lines = [
        "# Paper augmentation schemes @ 386x260 — TEST-split MAE (Tenebrio, seed 3035)",
        "",
        "Original (untuned) HPs: Adam, wd 1e-4, bs 6, step gamma=0.995/epoch, 1500 epochs;",
        "LR 1e-5 (CSRNet) / 1e-4 (MobileCount); LOG_PARA anchored at 772x519",
        "(CSRNet 25.05, MobileCount 638.73, identical across arms). Each cell = the",
        "best-val-MAE epoch's checkpoint evaluated on the held-out 386x260 test split.",
        "",
        "| Aug scheme | CSRNet val MAE | CSRNet test MAE | CSRNet test MSE | CSRNet ep "
        "| MobileCount val MAE | MobileCount test MAE | MobileCount test MSE | MobileCount ep |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        c = by[("CSRNet", arm)]
        m = by[("MobileCount", arm)]
        lines.append(
            f"| {arm_label[arm]} | {c['val_mae']:.4f} | {c['test_mae']:.4f} "
            f"| {c['test_mse']:.4f} | {c['best_epoch']} "
            f"| {m['val_mae']:.4f} | {m['test_mae']:.4f} "
            f"| {m['test_mse']:.4f} | {m['best_epoch']} |")
    n_test = results[0]["test_n"]
    lines += ["", f"Test split: {n_test} full 386x260 images for every arm "
                  "(csrnet9 patches are train-only)."]
    with path.open("w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
