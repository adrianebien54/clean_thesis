#!/usr/bin/env python3
"""Seed-variance sweep: Adam vs AdamW for MobileCount on Tenebrio 386x260.

Motivated by optimizer_arch_compare_06-10_18-12, where the two optimizers'
best val MAEs (0.509 vs 0.536) were within validation noise on one seed each.
Runs 2 optimizers x 3 seeds at 800 epochs (Adam's best-so-far went flat by
~ep 700; AdamW peaks ~ep 300), so best-checkpoint MAE can be reported as
mean +/- std per optimizer.

Other settings match the original comparison: lr 1e-4, weight_decay 1e-4,
step decay gamma=0.995/epoch, batch=6, validate every epoch.

Seeds python `random` (used by misc/transforms RandomCrop), numpy, and torch
CPU/CUDA before each run. Writes optimizer_seed_sweep_results.json under
./exp/optimizer_seed_sweep_<timestamp>/ incrementally after each run.

Env overrides: MAX_EPOCH (epoch budget; smoke test: MAX_EPOCH=2), NET
(architecture, default MobileCount), LR (base LR, default 1e-4 — use 1e-5
for CSRNet per the original comparison).
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

config.cfg.NET = os.environ.get("NET", "MobileCount")
config.cfg.DATASET = "Tenebrio"
config.cfg.GPU_ID = [0]
config.cfg.MAX_EPOCH = int(os.environ.get("MAX_EPOCH", "800"))
config.cfg.VAL_FREQ = 1
config.cfg.VAL_DENSE_START = -1
config.cfg.PRINT_FREQ = 200
config.cfg.PRE_GCC = False
config.cfg.RESUME = False
config.cfg.LR = float(os.environ.get("LR", "1e-4"))
config.cfg.WEIGHT_DECAY = 1e-4
config.cfg.LR_SCHEDULE = "step"
config.cfg.LR_DECAY = 0.995
config.cfg.NUM_EPOCH_LR_DECAY = 1
config.cfg.LR_DECAY_START = -1

SWEEP_STAMP = time.strftime("%m-%d_%H-%M", time.localtime())
config.cfg.EXP_PATH = f"./exp/optimizer_seed_sweep_{config.cfg.NET}_{SWEEP_STAMP}"

VARIANT = "386x260"
BATCH_SIZE = 6
SEEDS = [3035, 42, 7]  # 3035 is the codebase default (config.py SEED)
OPTIMIZERS = ["Adam", "AdamW"]
RUNS = [{"optimizer": opt, "seed": s} for opt in OPTIMIZERS for s in SEEDS]


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

    config.cfg.OPTIMIZER = run["optimizer"]
    config.cfg.SEED = run["seed"]
    config.cfg.EXP_NAME = f"{VARIANT}_{config.cfg.NET}_{run['optimizer']}_seed{run['seed']}_bs{BATCH_SIZE}"

    seed_everything(run["seed"])

    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))
    print(f"  [{run['optimizer']}/seed{run['seed']}] optimizer = {type(trainer.optimizer).__name__}  "
          f"lr={config.cfg.LR}  weight_decay={config.cfg.WEIGHT_DECAY}  seed={run['seed']}")

    rec_writer = RecordingWriter(trainer.writer)
    trainer.writer = rec_writer

    t0 = time.time()
    trainer.forward()
    duration = time.time() - t0

    result = {
        "net": config.cfg.NET,
        "optimizer": run["optimizer"],
        "seed": run["seed"],
        "variant": VARIANT,
        "batch_size": BATCH_SIZE,
        "lr": config.cfg.LR,
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
    print(f"=== Optimizer seed sweep: {config.cfg.NET} on {VARIANT} (lr={config.cfg.LR}), "
          f"{len(RUNS)} runs x {config.cfg.MAX_EPOCH} epochs ===")
    print(f"=== Output under {config.cfg.EXP_PATH} ===\n")

    results: list[dict] = []
    for run in RUNS:
        label = f"{run['optimizer']}/seed{run['seed']}"
        print(f"\n{'=' * 60}\nRUN: {label}\n{'=' * 60}")
        try:
            results.append(train_one(run))
        except Exception as e:
            print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
            results.append({"optimizer": run["optimizer"], "seed": run["seed"],
                            "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        out_dir = Path(config.cfg.EXP_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "optimizer_seed_sweep_results.json").open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f'{"optimizer":<10} {"seed":<6} {"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r["optimizer"]:<10} {r["seed"]:<6} ERROR: {r["error"]}')
        else:
            print(f'{r["optimizer"]:<10} {r["seed"]:<6} {r["best_mae"]:<12.4f} '
                  f'{r["best_mse"]:<12.4f} {r["duration_sec"]:>6.0f}s')
    for opt in OPTIMIZERS:
        maes = [r["best_mae"] for r in results if r.get("optimizer") == opt and "error" not in r]
        if maes:
            print(f'{opt}: best MAE mean {np.mean(maes):.4f} +/- {np.std(maes):.4f}  (n={len(maes)})')


if __name__ == "__main__":
    main()
