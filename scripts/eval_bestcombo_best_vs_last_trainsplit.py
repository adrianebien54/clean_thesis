#!/usr/bin/env python3
"""Best-checkpoint vs last-epoch TRAIN-split MAE for the resolution study.

Overfitting-vs-drift diagnostic for the thesis resolution chapter (same logic
as scripts/eval_aug_paper_schemes_lastepoch.py): if MobileCount's late val-MAE
rise at 772x519/1544x1038 were memorization, the last-epoch state would score
train far below val; if train and val rise together it is optimization drift.

For each (net, variant) in {CSRNet, MobileCount} x all 6 resolutions: scores
the best-val-MAE checkpoint AND latest_state.pth on the 840 train images (val
numbers come from the logged curves). Run dirs/records as in
scripts/eval_bestcombo_resolutions_testset.py. Skips (net, variant) pairs
already present in the output JSON, so it can be re-run incrementally.

Writes exp/combined_resolution/trainsplit_best_vs_last.json.
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

BESTCOMBO_DIR = ROOT / "exp/bestcombo_resolutions_06-24_11-49"
BESTCOMBO_JSON = BESTCOMBO_DIR / "bestcombo_resolutions_results.json"
CSRNET_386_DIR = ROOT / "exp/batchsize_sweep_logpara_06-21_22-41"
CSRNET_386_JSON = CSRNET_386_DIR / "batchsize_logpara_results.json"
MOBILE_386_DIR = ROOT / "exp/360p_lr_optim_grid/mobilecount_logpara2550_anchor772x519_06-17_14-26"
MOBILE_386_JSON = MOBILE_386_DIR / "mobilecount_logpara2550_results.json"

VARIANTS = ["49x33", "97x65", "193x130", "386x260", "772x519", "1544x1038"]
OUT = ROOT / "exp/combined_resolution/trainsplit_best_vs_last.json"


def _rows(path):
    d = json.load(open(path))
    rows = d if isinstance(d, list) else d.get("runs", [])
    return [r for r in rows if isinstance(r, dict) and "error" not in r]


def select_run(variant: str, net: str) -> dict:
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


def mae_curve(record):
    return dict((int(m["step"]), float(m["value"]))
                for m in record["metrics"] if m["tag"] == "mae")


def build_train_loader(variant: str, log_para: float) -> DataLoader:
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD),
    ])
    gt_transform = standard_transforms.Compose([own_transforms.LabelNormalize(log_para)])
    ds = Tenebrio(str(ROOT / f"exp/data/Tenebrio/{variant}/train"), "test",
                  main_transform=None, img_transform=img_transform,
                  gt_transform=gt_transform)
    return DataLoader(ds, batch_size=1, num_workers=8, shuffle=False, drop_last=False)


@torch.no_grad()
def train_mae(net_name: str, state_dict, loader, log_para: float) -> tuple[float, float]:
    net = CrowdCounter(config.cfg.GPU_ID, net_name)
    net.load_state_dict(state_dict)
    net.cuda(); net.eval()
    abs_errs, sq_errs = [], []
    for img, gt_map in loader:
        pred = net.test_forward(img.cuda())
        d = float(gt_map.sum().item()) / log_para - float(pred.sum().item()) / log_para
        abs_errs.append(abs(d)); sq_errs.append(d * d)
    del net
    gc.collect(); torch.cuda.empty_cache()
    return float(np.mean(abs_errs)), float(math.sqrt(np.mean(sq_errs)))


out = json.load(OUT.open()) if OUT.exists() else []
done = {(r["net"], r["variant"]) for r in out}
print(f"{'net':<12} {'variant':<10} {'ckpt':>5} {'epoch':>6} {'train_mae':>10} {'val_mae':>9}")
print("-" * 60)
for variant in VARIANTS:
    for net_name in ("CSRNet", "MobileCount"):
        if (net_name, variant) in done:
            continue
        spec = select_run(variant, net_name)
        rec, run_dir = spec["record"], spec["run_dir"]
        log_para = float(rec["log_para"])
        curve = mae_curve(rec)
        best_ep = min(curve, key=curve.get)
        loader = build_train_loader(variant, log_para)

        hits = sorted(run_dir.glob(f"all_ep_{best_ep}_mae_*.pth"))
        best_sd = torch.load(str(hits[0]), map_location="cuda", weights_only=True)
        last = torch.load(str(run_dir / "latest_state.pth"), map_location="cuda",
                          weights_only=False)  # trainer state: our own file
        last_ep = last["epoch"] + 1

        row = dict(net=net_name, variant=variant, log_para=log_para,
                   best_epoch=best_ep, last_epoch=last_ep, train_n=840)
        for tag, sd, ep in (("best", best_sd, best_ep), ("last", last["net"], last_ep)):
            tm, tmse = train_mae(net_name, sd, loader, log_para)
            row[f"{tag}_train_mae"], row[f"{tag}_train_mse"] = tm, tmse
            row[f"{tag}_val_mae"] = curve[ep]
            print(f"{net_name:<12} {variant:<10} {tag:>5} {ep:>6} {tm:>10.4f} {curve[ep]:>9.4f}")
        out.append(row)
        del best_sd, last
        gc.collect(); torch.cuda.empty_cache()

out.sort(key=lambda r: (VARIANTS.index(r["variant"]), r["net"]))
with OUT.open("w") as f:
    json.dump(out, f, indent=1)
print("\nwrote", OUT)
