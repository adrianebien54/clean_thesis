#!/usr/bin/env python3
"""Batch-size sweep at each net's new per-model LOG_PARA (AdamW only; seed 3035, Tenebrio 386x260).

Re-runs the batch-size sweep (sweep_batchsize.py) at the density scale that the
new-LOG_PARA LR grids selected as each net's robust optimum, AdamW only:

  CSRNet      LOG_PARA_BASE = 100  @ 772x519  -> 25.03  at 386x260  (lp25)
  MobileCount LOG_PARA_BASE = 2550 @ 772x519  -> 638.73 at 386x260  (lp639)

LOG_PARA is a runtime multiplier on the density target, auto-scaled per resolution
in loading_data() via LOG_PARA = LOG_PARA_BASE * (w*h) / LOG_PARA_BASE_AREA, so we
set LOG_PARA_BASE / LOG_PARA_BASE_AREA per run (both scoped to this sweep; setting.py
stays at 100 / 386*260). The trainer divides LOG_PARA back out, so reported MAE/MSE
stay in count units; only the loss/gradient scale changes.

Each batch size runs at its own sqrt(bs/6)-scaled LR off the net's bs6 optimum, so the
comparison across batch sizes is fair. The bs6 anchors are unchanged from the old sweep
(CSRNet AdamW@1e-4 -> 0.320, MobileCount AdamW@1e-3 -> 0.385 at the new LOG_PARA) and are
NOT re-run here -- they live in the new-LOG_PARA LR grids under exp/360p_lr_optim_grid/ and are
merged in by merge_batchsize_logpara_results.py.

8 runs total, 1500 epochs each (budget where gamma=0.995/epoch step decay has collapsed
LR to ~0.05% of base). Seeds python random / numpy / torch (3035) before each run, matching
sweep_batchsize.py. Writes batchsize_logpara_results.json under
./exp/batchsize_sweep_logpara_<timestamp>/ incrementally after each run.

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

ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])
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
config.cfg.EXP_PATH = f"./exp/batchsize_sweep_logpara_{SWEEP_STAMP}"

VARIANT = "386x260"
SEED = 3035
ANCHOR = 772 * 519  # LOG_PARA_BASE_AREA for both nets (re-anchored scheme)
# bs6 is the LR-grid optimum (in exp/360p_lr_optim_grid/), NOT re-run here. sqrt(bs/6)-scaled
# LR per row off each net's bs6 anchor: CSRNet AdamW@1e-4, MobileCount AdamW@1e-3.
RUNS = [
    # CSRNet -- LOG_PARA_BASE=100 @ 772x519 -> 25.03 at 386x260 (lp25)
    {"net": "CSRNet",      "bs": 1, "lr": 4e-5,   "optimizer": "AdamW", "lp_base": 100.0,  "lp_area": ANCHOR, "lp_tag": "lp25"},
    {"net": "CSRNet",      "bs": 2, "lr": 6e-5,   "optimizer": "AdamW", "lp_base": 100.0,  "lp_area": ANCHOR, "lp_tag": "lp25"},
    {"net": "CSRNet",      "bs": 4, "lr": 8e-5,   "optimizer": "AdamW", "lp_base": 100.0,  "lp_area": ANCHOR, "lp_tag": "lp25"},
    {"net": "CSRNet",      "bs": 8, "lr": 1.2e-4, "optimizer": "AdamW", "lp_base": 100.0,  "lp_area": ANCHOR, "lp_tag": "lp25"},
    # MobileCount -- LOG_PARA_BASE=2550 @ 772x519 -> 638.73 at 386x260 (lp639)
    {"net": "MobileCount", "bs": 1, "lr": 4e-4,   "optimizer": "AdamW", "lp_base": 2550.0, "lp_area": ANCHOR, "lp_tag": "lp639"},
    {"net": "MobileCount", "bs": 2, "lr": 6e-4,   "optimizer": "AdamW", "lp_base": 2550.0, "lp_area": ANCHOR, "lp_tag": "lp639"},
    {"net": "MobileCount", "bs": 4, "lr": 8e-4,   "optimizer": "AdamW", "lp_base": 2550.0, "lp_area": ANCHOR, "lp_tag": "lp639"},
    {"net": "MobileCount", "bs": 8, "lr": 1.2e-3, "optimizer": "AdamW", "lp_base": 2550.0, "lp_area": ANCHOR, "lp_tag": "lp639"},
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
    cfg_data.LOG_PARA_BASE = run["lp_base"]       # 100 (CSRNet) / 2550 (MobileCount)
    cfg_data.LOG_PARA_BASE_AREA = run["lp_area"]  # 772*519, read by _compute_log_para()

    config.cfg.NET = run["net"]
    config.cfg.LR = run["lr"]
    config.cfg.OPTIMIZER = run["optimizer"]
    config.cfg.SEED = SEED
    config.cfg.EXP_NAME = (f"{VARIANT}_{run['net']}_{run['optimizer']}"
                           f"_lr{run['lr']:.1e}_seed{SEED}_bs{run['bs']}_{run['lp_tag']}")

    seed_everything(SEED)

    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))
    print(f"  [{run['net']}/{run['optimizer']}/bs{run['bs']}/lr{run['lr']:.1e}] "
          f"optimizer = {type(trainer.optimizer).__name__}  lr={config.cfg.LR}  "
          f"batch_size={cfg_data.TRAIN_BATCH_SIZE}  log_para={cfg_data.LOG_PARA:.2f}  "
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
        "log_para_base": run["lp_base"],
        "log_para_base_area": run["lp_area"],
        "log_para": float(cfg_data.LOG_PARA),
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
    print(f"=== Batch-size sweep (new LOG_PARA, AdamW only) on {VARIANT}: {len(RUNS)} runs x "
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
        with (out_dir / "batchsize_logpara_results.json").open("w") as f:
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
