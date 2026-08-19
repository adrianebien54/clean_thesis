#!/usr/bin/env python3
"""Score the 6 paper-aug selected checkpoints on the TRAIN split (840 full images).

Same protocol as scripts/eval_aug_paper_schemes_testset.py, but on train/ —
measures memorization: train MAE far below val/test MAE = overfitting.
All arms (incl. csrnet9, which trained on patches of these images) are scored
on the same 840 full 386x260 train images.
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

SWEEP = ROOT / "exp/aug_paper_schemes_386x260_07-04"
tests = json.load((SWEEP / "testset_results.json").open())

mean_std = cfg_data.MEAN_STD
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std),
])

out = []
print(f"{'net':<12} {'arm':<9} {'epoch':>6} {'train_mae':>10} {'val_mae':>9} {'test_mae':>9}")
print("-" * 60)
for t in tests:
    log_para = t["log_para"]
    gt_transform = standard_transforms.Compose([own_transforms.LabelNormalize(log_para)])
    ds = Tenebrio(str(ROOT / "exp/data/Tenebrio/386x260/train"), "test",
                  main_transform=None, img_transform=img_transform,
                  gt_transform=gt_transform)
    loader = DataLoader(ds, batch_size=1, num_workers=8, shuffle=False, drop_last=False)

    net = CrowdCounter(config.cfg.GPU_ID, t["net"])
    state = torch.load(str(ROOT / t["checkpoint"]), map_location="cuda", weights_only=True)
    net.load_state_dict(state)
    net.cuda(); net.eval()

    abs_errs, sq_errs = [], []
    with torch.no_grad():
        for img, gt_map in loader:
            pred = net.test_forward(img.cuda())
            d = float(gt_map.sum().item()) / log_para - float(pred.sum().item()) / log_para
            abs_errs.append(abs(d)); sq_errs.append(d * d)
    rec = dict(net=t["net"], arm=t["arm"], best_epoch=t["best_epoch"],
               train_mae=float(np.mean(abs_errs)),
               train_mse=float(math.sqrt(np.mean(sq_errs))),
               train_n=len(abs_errs),
               val_mae=t["val_mae"], test_mae=t["test_mae"])
    out.append(rec)
    print(f"{rec['net']:<12} {rec['arm']:<9} {rec['best_epoch']:>6} "
          f"{rec['train_mae']:>10.4f} {rec['val_mae']:>9.4f} {rec['test_mae']:>9.4f}")

    del net
    gc.collect(); torch.cuda.empty_cache()

with open(SWEEP / "trainsplit_results.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote", SWEEP / "trainsplit_results.json")
