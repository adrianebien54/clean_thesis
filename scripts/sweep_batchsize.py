#!/usr/bin/env python3
"""Batch-size sweep with sqrt-scaled learning rate (seed 3035, Tenebrio 386x260).

Batch size is the last hyperparameter to tune after the optimizer x LR grid.
Because the optimal LR scales with batch size (sqrt-rule for Adam-family), each
batch size is evaluated at its own sqrt(bs/6)-scaled LR rather than a frozen one,
so the comparison across batch sizes is fair. bs6 is already covered by the LR
grid and is NOT re-run here.

Anchors (best cells from the bs6 LR grid):
  CSRNet      AdamW @ 1e-4  (0.34)
  MobileCount AdamW @ 1e-3  (0.40)

Optimizer coverage differs by net, driven by the bs6 grid:
  CSRNet      -- AdamW won every measured cell; scaled LRs (4e-5..1.2e-4) sit
                 between two measured AdamW wins -> AdamW only.
  MobileCount -- Adam wins the low-LR tail (1e-5, 1e-6), AdamW wins at ~1e-3,
                 and the 1e-5->1e-3 gap is unmeasured. The scaled LRs for
                 bs1/2/4 (4e-4, 6e-4, 8e-4) fall in that gap -> run BOTH
                 optimizers. bs8 scales to 1.2e-3 (> 1e-3, AdamW's confirmed
                 region) -> AdamW only.

11 runs total, 1500 epochs each (budget where gamma=0.995/epoch step decay has
collapsed LR to ~0.05% of base). Seeds python random / numpy / torch (3035)
before each run, matching sweep_lr_grid.py. Writes batchsize_sweep_results.json
under ./exp/batchsize_sweep_<timestamp>/ incrementally after each run.

Set MAX_EPOCH env var to override the epoch budget (smoke test: MAX_EPOCH=2).
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/umrobotics/clean_thesis/crowd_counting")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import config

config.cfg.DATASET = "Tenebrio"
config.cfg.GPU_ID = [0]
config.cfg.MAX_EPOCH = int(os.environ.get("MAX_EPOCH", "1500"))
config.cfg.VAL_FREQ = 1
config.cfg.VAL_DENSE_START = -1
config.cfg.PRINT_FREQ = 200
config.cfg.PRE_GCC = False
config.cfg.RESUME = False
config.cfg.WEIGHT_DECAY = 1e-4
config.cfg.LR_SCHEDULE = "step"
config.cfg.LR_DECAY = 0.995
config.cfg.NUM_EPOCH_LR_DECAY = 1
config.cfg.LR_DECAY_START = -1

SWEEP_STAMP = time.strftime("%m-%d_%H-%M", time.localtime())
config.cfg.EXP_PATH = f"./exp/batchsize_sweep_{SWEEP_STAMP}"

VARIANT = "386x260"
SEED = 3035
# bs6 is already measured by the LR grid; sqrt(bs/6)-scaled LR per row.
# NOTE: CSRNet bs1 AdamW lr4e-5 already completed (best_mae 0.2860) in the
# 06-13_23-09 sweep dir; its result lives in that dir's results.json. Dropped
# here so the relaunch does not redo it. The other 10 runs all still need to
# run (CSRNet bs2 died at ep151 unrecorded in the first attempt).
RUNS = [
    {"net": "CSRNet",      "bs": 2, "lr": 6e-5, "optimizer": "AdamW"},
    {"net": "CSRNet",      "bs": 4, "lr": 8e-5, "optimizer": "AdamW"},
    {"net": "CSRNet",      "bs": 8, "lr": 1.2e-4, "optimizer": "AdamW"},
    {"net": "MobileCount", "bs": 1, "lr": 4e-4, "optimizer": "Adam"},
    {"net": "MobileCount", "bs": 1, "lr": 4e-4, "optimizer": "AdamW"},
    {"net": "MobileCount", "bs": 2, "lr": 6e-4, "optimizer": "Adam"},
    {"net": "MobileCount", "bs": 2, "lr": 6e-4, "optimizer": "AdamW"},
    {"net": "MobileCount", "bs": 4, "lr": 8e-4, "optimizer": "Adam"},
    {"net": "MobileCount", "bs": 4, "lr": 8e-4, "optimizer": "AdamW"},
    {"net": "MobileCount", "bs": 8, "lr": 1.2e-3, "optimizer": "AdamW"},
]


class RecordingWriter:
    def __init__(self, real_writer):
        self._real = real_writer
        self.records: list[dict] = []

    def add_scalar(self, tag, value, step):
        self._real.add_scalar(tag, value, step)
        try:
            self.records.append({"tag": tag, "value": float(value), "step": int(step)})
        except Exception:
            self.records.append({"tag": tag, "value": float(value), "step": str(step)})

    def __getattr__(self, name):
        return getattr(self._real, name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one(run: dict) -> dict:
    from datasets.Tenebrio.setting import cfg_data
    cfg_data.DATA_PATH = f"./exp/data/Tenebrio/{VARIANT}"
    cfg_data.TRAIN_BATCH_SIZE = run["bs"]
    cfg_data.VAL_BATCH_SIZE = 1

    config.cfg.NET = run["net"]
    config.cfg.LR = run["lr"]
    config.cfg.OPTIMIZER = run["optimizer"]
    config.cfg.SEED = SEED
    config.cfg.EXP_NAME = (f"{VARIANT}_{run['net']}_{run['optimizer']}"
                           f"_lr{run['lr']:.1e}_seed{SEED}_bs{run['bs']}")

    seed_everything(SEED)

    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))
    print(f"  [{run['net']}/{run['optimizer']}/bs{run['bs']}/lr{run['lr']:.1e}] "
          f"optimizer = {type(trainer.optimizer).__name__}  lr={config.cfg.LR}  "
          f"batch_size={cfg_data.TRAIN_BATCH_SIZE}  "
          f"weight_decay={config.cfg.WEIGHT_DECAY}  seed={SEED}")

    rec_writer = RecordingWriter(trainer.writer)
    trainer.writer = rec_writer

    t0 = time.time()
    trainer.forward()
    duration = time.time() - t0

    result = {
        "net": run["net"],
        "optimizer": run["optimizer"],
        "seed": SEED,
        "variant": VARIANT,
        "batch_size": run["bs"],
        "lr": run["lr"],
        "weight_decay": config.cfg.WEIGHT_DECAY,
        "max_epoch": config.cfg.MAX_EPOCH,
        "duration_sec": duration,
        "metrics": rec_writer.records,
        "best_mae": float(trainer.train_record["best_mae"]),
        "best_mse": float(trainer.train_record["best_mse"]),
        "best_model_name": trainer.train_record["best_model_name"],
    }
    try:
        trainer.writer._real.close()
    except Exception:
        pass
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    print(f"=== Batch-size sweep on {VARIANT}: {len(RUNS)} runs x "
          f"{config.cfg.MAX_EPOCH} epochs, seed {SEED} ===")
    print(f"=== Output under {config.cfg.EXP_PATH} ===\n")

    results: list[dict] = []
    for run in RUNS:
        label = f"{run['net']}/{run['optimizer']}/bs{run['bs']}/lr{run['lr']:.1e}"
        print(f"\n{'=' * 60}\nRUN: {label}\n{'=' * 60}")
        try:
            results.append(train_one(run))
        except Exception as e:
            print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
            results.append({"net": run["net"], "optimizer": run["optimizer"],
                            "bs": run["bs"], "lr": run["lr"],
                            "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        out_dir = Path(config.cfg.EXP_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "batchsize_sweep_results.json").open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f'{"net":<13} {"optimizer":<10} {"bs":<4} {"lr":<10} {"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r["net"]:<13} {r["optimizer"]:<10} {r["bs"]:<4} {r["lr"]:<10.1e} ERROR: {r["error"]}')
        else:
            print(f'{r["net"]:<13} {r["optimizer"]:<10} {r["batch_size"]:<4} {r["lr"]:<10.1e} '
                  f'{r["best_mae"]:<12.4f} {r["best_mse"]:<12.4f} {r["duration_sec"]:>6.0f}s')


if __name__ == "__main__":
    main()
