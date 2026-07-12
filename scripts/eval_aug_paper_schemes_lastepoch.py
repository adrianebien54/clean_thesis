#!/usr/bin/env python3
"""Score the 6 paper-aug LAST-epoch checkpoints (latest_state.pth) on the TRAIN split.

Companion to scripts/eval_aug_paper_schemes_trainsplit.py (identical protocol:
840 full 386x260 train images, batch 1), but loads each run's end-of-training
state instead of the best-val-MAE checkpoint. Together with the logged
last-epoch validation MAE this quantifies post-best-epoch overfitting:
train MAE keeps falling after the selected epoch while val MAE rises.

Writes exp/aug_paper_schemes_386x260_07-04/lastepoch_results.json with the
last-epoch train MAE/MSE and the logged last-epoch val MAE per run.
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

SWEEP = ROOT / "exp/aug_paper_schemes_386x260_07-04"
tests = json.load((SWEEP / "testset_results.json").open())
runs = json.load((SWEEP / "aug_paper_schemes_results.json").open())
val_curve = {(r["net"], r["arm"]): [x["value"] for x in r["metrics"] if x["tag"] == "mae"]
             for r in runs}

mean_std = cfg_data.MEAN_STD
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std),
])

out = []
print(f"{'net':<12} {'arm':<9} {'epoch':>6} {'train_mae':>10} {'val_mae':>9}")
print("-" * 50)
for t in tests:
    log_para = t["log_para"]
    gt_transform = standard_transforms.Compose([own_transforms.LabelNormalize(log_para)])
    ds = Tenebrio(str(ROOT / "exp/data/Tenebrio/386x260/train"), "test",
                  main_transform=None, img_transform=img_transform,
                  gt_transform=gt_transform)
    loader = DataLoader(ds, batch_size=1, num_workers=8, shuffle=False, drop_last=False)

    run_dir = (ROOT / t["checkpoint"]).parent
    # trainer state (net + optimizer + numpy scalars): our own file, needs weights_only=False
    state = torch.load(str(run_dir / "latest_state.pth"), map_location="cuda",
                       weights_only=False)
    last_epoch = state["epoch"] + 1  # trainer epochs are 0-indexed

    net = CrowdCounter(config.cfg.GPU_ID, t["net"])
    net.load_state_dict(state["net"])
    net.cuda(); net.eval()

    abs_errs, sq_errs = [], []
    with torch.no_grad():
        for img, gt_map in loader:
            pred = net.test_forward(img.cuda())
            d = float(gt_map.sum().item()) / log_para - float(pred.sum().item()) / log_para
            abs_errs.append(abs(d)); sq_errs.append(d * d)
    val_mae = val_curve[(t["net"], t["arm"])][last_epoch - 1]
    rec = dict(net=t["net"], arm=t["arm"], last_epoch=last_epoch,
               train_mae=float(np.mean(abs_errs)),
               train_mse=float(math.sqrt(np.mean(sq_errs))),
               train_n=len(abs_errs),
               val_mae=val_mae)
    out.append(rec)
    print(f"{rec['net']:<12} {rec['arm']:<9} {rec['last_epoch']:>6} "
          f"{rec['train_mae']:>10.4f} {rec['val_mae']:>9.4f}")

    del net, state
    gc.collect(); torch.cuda.empty_cache()

with open(SWEEP / "lastepoch_results.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote", SWEEP / "lastepoch_results.json")
