#!/usr/bin/env python3
"""Re-run the CSRNet optimizer x LR grid with a smaller density scale (seed 3035, Tenebrio 386x260).

Repeats the CSRNet portion of the optimizer x LR grid (sweep_lr_grid.py) with a
smaller density scale. LOG_PARA is a runtime multiplier on the density target
(misc/transforms.LabelNormalize), not baked into the stored HDF5 maps, so nothing
is regenerated -- only the multiplier changes. The trainer divides LOG_PARA back
out, so reported MAE/MSE stay in count units; only the loss/gradient scale shrinks.

This is the low-end counterpart to sweep_mobilecount_logpara2550.py (which probed
MobileCount at the *larger* scale 638.73): here we probe CSRNet at the *smaller*
scale produced by re-anchoring the per-resolution scheme at 772x519.

Tenebrio auto-scales LOG_PARA per resolution in loading_data() via
  LOG_PARA = LOG_PARA_BASE * (w*h) / LOG_PARA_BASE_AREA
so setting LOG_PARA directly would be overwritten. We anchor LOG_PARA_BASE = 100 at
772x519 (LOG_PARA_BASE_AREA = 772*519, both scoped to this sweep; setting.py is
left at 100 / 386*260). The scheme scales by area, and 386x260 is 1/4 the area of
772x519, so at this 386x260 variant LOG_PARA = 100 * (386*260)/(772*519) = 25.03.

Runs (8): the bs6 LR grid -- CSRNet x {Adam, AdamW} x {1e-3, 1e-4, 1e-5, 1e-6}.

1500 epochs each -- the budget where gamma=0.995/epoch step decay has collapsed the
LR to 0.995^1500 ~ 0.05% of base. CSRNet@1e-3 may diverge (pretrained VGG backbone);
it runs anyway to document the boundary. Seeds python random / numpy / torch (3035)
before each run, matching sweep_lr_grid.py. Writes csrnet_logpara_results.json under
./exp/csrnet_logpara100_anchor772x519_<timestamp>/ incrementally after each run.

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
config.cfg.EXP_PATH = f"./exp/csrnet_logpara100_anchor772x519_{SWEEP_STAMP}"

VARIANT = "386x260"
SEED = 3035
# 100 anchored at 772x519; the per-resolution (area) scheme scales it down to
# LOG_PARA = 100 * (386*260)/(772*519) = 25.03 at this 386x260 variant.
LOG_PARA_BASE = 100.0
LOG_PARA_BASE_AREA = 772 * 519
LP_TAG = "lp25"  # applied LOG_PARA at 386x260 (25.03), rounded, for run names

LR_GRID_RUNS = [
    # 8 LR-grid cells (bs6) -- CSRNet x {Adam, AdamW} x {1e-3, 1e-4, 1e-5, 1e-6}
    {"net": "CSRNet", "bs": 6, "lr": 1e-3, "optimizer": "Adam"},
    {"net": "CSRNet", "bs": 6, "lr": 1e-3, "optimizer": "AdamW"},
    {"net": "CSRNet", "bs": 6, "lr": 1e-4, "optimizer": "Adam"},
    {"net": "CSRNet", "bs": 6, "lr": 1e-4, "optimizer": "AdamW"},
    {"net": "CSRNet", "bs": 6, "lr": 1e-5, "optimizer": "Adam"},
    {"net": "CSRNet", "bs": 6, "lr": 1e-5, "optimizer": "AdamW"},
    {"net": "CSRNet", "bs": 6, "lr": 1e-6, "optimizer": "Adam"},
    {"net": "CSRNet", "bs": 6, "lr": 1e-6, "optimizer": "AdamW"},
]

RUNS = LR_GRID_RUNS


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
    cfg_data.LOG_PARA_BASE = LOG_PARA_BASE            # 100, anchored at 772x519 below
    cfg_data.LOG_PARA_BASE_AREA = LOG_PARA_BASE_AREA  # read by _compute_log_para()

    config.cfg.NET = run["net"]
    config.cfg.LR = run["lr"]
    config.cfg.OPTIMIZER = run["optimizer"]
    config.cfg.SEED = SEED
    config.cfg.EXP_NAME = (f"{VARIANT}_{run['net']}_{run['optimizer']}"
                           f"_lr{run['lr']:.1e}_seed{SEED}_bs{run['bs']}_{LP_TAG}")

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
        "log_para_base": LOG_PARA_BASE,
        "log_para_base_area": LOG_PARA_BASE_AREA,
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
    print(f"=== CSRNet LOG_PARA_BASE=100 @ 772x519 (-> 25.03 at {VARIANT}) sweep: "
          f"{len(RUNS)} runs x {config.cfg.MAX_EPOCH} epochs, seed {SEED} ===")
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
        with (out_dir / "csrnet_logpara_results.json").open("w") as f:
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
