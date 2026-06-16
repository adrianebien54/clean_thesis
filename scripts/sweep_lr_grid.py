#!/usr/bin/env python3
"""Fill in the missing cells of the optimizer x LR grid (seed 3035, bs6).

Full grid is {MobileCount, CSRNet} x {Adam, AdamW} x {1e-3, 1e-4, 1e-5, 1e-6}
on Tenebrio 386x260. Existing seeded runs cover MobileCount@1e-3/1e-4 and
CSRNet@1e-5 (800 ep, being extended to 1500 by continue_seed3035_to1500.py).
This script runs the 10 remaining cells from scratch:

  MobileCount x {Adam, AdamW} x {1e-5, 1e-6}
  CSRNet      x {Adam, AdamW} x {1e-3, 1e-4, 1e-6}

1500 epochs each — the budget where step decay gamma=0.995/epoch has
collapsed the LR to 0.995^1500 ~ 0.05% of base. CSRNet@1e-3 may diverge
(pretrained VGG backbone); it runs anyway to document the boundary.

Unlike compare_optimizers_archs.py this script seeds python random / numpy /
torch before each run (seed 3035), matching sweep_optimizer_seeds.py.
Writes lr_grid_results.json under ./exp/lr_grid_<timestamp>/ incrementally
after each run.

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
config.cfg.EXP_PATH = f"./exp/lr_grid_{SWEEP_STAMP}"

VARIANT = "386x260"
BATCH_SIZE = 6
SEED = 3035
RUNS = [
    {"net": "MobileCount", "lr": 1e-5, "optimizer": "Adam"},
    {"net": "MobileCount", "lr": 1e-5, "optimizer": "AdamW"},
    {"net": "MobileCount", "lr": 1e-6, "optimizer": "Adam"},
    {"net": "MobileCount", "lr": 1e-6, "optimizer": "AdamW"},
    {"net": "CSRNet",      "lr": 1e-3, "optimizer": "Adam"},
    {"net": "CSRNet",      "lr": 1e-3, "optimizer": "AdamW"},
    {"net": "CSRNet",      "lr": 1e-4, "optimizer": "Adam"},
    {"net": "CSRNet",      "lr": 1e-4, "optimizer": "AdamW"},
    {"net": "CSRNet",      "lr": 1e-6, "optimizer": "Adam"},
    {"net": "CSRNet",      "lr": 1e-6, "optimizer": "AdamW"},
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
    cfg_data.TRAIN_BATCH_SIZE = BATCH_SIZE
    cfg_data.VAL_BATCH_SIZE = 1

    config.cfg.NET = run["net"]
    config.cfg.LR = run["lr"]
    config.cfg.OPTIMIZER = run["optimizer"]
    config.cfg.SEED = SEED
    config.cfg.EXP_NAME = (f"{VARIANT}_{run['net']}_{run['optimizer']}"
                           f"_lr{run['lr']:.0e}_seed{SEED}_bs{BATCH_SIZE}")

    seed_everything(SEED)

    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))
    print(f"  [{run['net']}/{run['optimizer']}/lr{run['lr']:.0e}] "
          f"optimizer = {type(trainer.optimizer).__name__}  lr={config.cfg.LR}  "
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
        "batch_size": BATCH_SIZE,
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
    print(f"=== Optimizer x LR grid on {VARIANT}: {len(RUNS)} runs x "
          f"{config.cfg.MAX_EPOCH} epochs, seed {SEED} ===")
    print(f"=== Output under {config.cfg.EXP_PATH} ===\n")

    results: list[dict] = []
    for run in RUNS:
        label = f"{run['net']}/{run['optimizer']}/lr{run['lr']:.0e}"
        print(f"\n{'=' * 60}\nRUN: {label}\n{'=' * 60}")
        try:
            results.append(train_one(run))
        except Exception as e:
            print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
            results.append({"net": run["net"], "optimizer": run["optimizer"],
                            "lr": run["lr"], "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        out_dir = Path(config.cfg.EXP_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "lr_grid_results.json").open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f'{"net":<13} {"optimizer":<10} {"lr":<8} {"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r["net"]:<13} {r["optimizer"]:<10} {r["lr"]:<8.0e} ERROR: {r["error"]}')
        else:
            print(f'{r["net"]:<13} {r["optimizer"]:<10} {r["lr"]:<8.0e} {r["best_mae"]:<12.4f} '
                  f'{r["best_mse"]:<12.4f} {r["duration_sec"]:>6.0f}s')


if __name__ == "__main__":
    main()
