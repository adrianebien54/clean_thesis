#!/usr/bin/env python3
"""Compare augmentation schemes (basic vs extended) at 360p (386x260), Tenebrio.

Each net is trained at its `none`-optimal opt/LR/batch-size/LOG_PARA combo (tuned at
386x260, from the new-LOG_PARA grids -- identical to scripts/run_best_combo_resolutions.py),
holding HPs fixed and varying ONLY the augmentation scheme:

  CSRNet      AdamW lr=4e-5 bs1  lp_base=100   (anchor 772x519 -> LOG_PARA ~25  @ 386x260)
  MobileCount AdamW lr=1e-3 bs6  lp_base=2550  (anchor 772x519 -> LOG_PARA ~639 @ 386x260)

The no-augmentation baseline is NOT re-run here -- reuse the existing 386x260 bests measured
with AUG='none' (RandomCrop only): CSRNet 0.2876, MobileCount 0.3850.

4 runs (2 nets x {basic, extended}), 1500 epochs each (standard budget), seed 3035, MobileCount
first (cheaper). Records best VAL MAE (trainer.train_record), same metric as the baseline.
Writes aug_schemes_360p_results.json under ./exp/aug_schemes_360p_<timestamp>/ incrementally.

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
config.cfg.EXP_PATH = f"./exp/aug_schemes_360p_{SWEEP_STAMP}"

SEED = 3035
VARIANT = "386x260"
ANCHOR = 772 * 519  # LOG_PARA_BASE_AREA for both nets (re-anchored scheme)

# each net's best cell from the new-LOG_PARA grids (tuned at 386x260, no augmentation)
BEST = {
    "CSRNet":      {"optimizer": "AdamW", "lr": 4e-5, "bs": 1, "lp_base": 100.0},
    "MobileCount": {"optimizer": "AdamW", "lr": 1e-3, "bs": 6, "lp_base": 2550.0},
}

AUG_SCHEMES = ["basic", "extended"]

# MobileCount first (cheaper), basic before extended
RUNS = [
    {"net": net, "aug": aug, **BEST[net]}
    for net in ("MobileCount", "CSRNet")
    for aug in AUG_SCHEMES
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
    cfg_data.LOG_PARA_BASE = run["lp_base"]   # 100 (CSRNet) / 2550 (MobileCount)
    cfg_data.LOG_PARA_BASE_AREA = ANCHOR      # 772*519, read by _compute_log_para()
    cfg_data.AUG = run["aug"]                 # 'basic' | 'extended' -- read by loading_data()

    config.cfg.NET = run["net"]
    config.cfg.LR = run["lr"]
    config.cfg.OPTIMIZER = run["optimizer"]
    config.cfg.SEED = SEED
    config.cfg.EXP_NAME = (f"{VARIANT}_{run['net']}_{run['optimizer']}"
                           f"_lr{run['lr']:.1e}_seed{SEED}_bs{run['bs']}_{run['aug']}")

    seed_everything(SEED)

    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))
    print(f"  [{run['net']}/{run['aug']}/bs{run['bs']}/lr{run['lr']:.1e}] "
          f"optimizer = {type(trainer.optimizer).__name__}  lr={config.cfg.LR}  "
          f"batch_size={cfg_data.TRAIN_BATCH_SIZE}  log_para={cfg_data.LOG_PARA:.2f}  "
          f"aug={cfg_data.AUG}  weight_decay={config.cfg.WEIGHT_DECAY}  seed={SEED}")

    rec_writer = RecordingWriter(trainer.writer)
    trainer.writer = rec_writer

    t0 = time.time()
    trainer.forward()
    duration = time.time() - t0

    result = {
        "net": run["net"],
        "aug": run["aug"],
        "optimizer": run["optimizer"],
        "seed": SEED,
        "variant": VARIANT,
        "batch_size": run["bs"],
        "lr": run["lr"],
        "log_para_base": run["lp_base"],
        "log_para_base_area": ANCHOR,
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
    print(f"=== Aug schemes (basic vs extended) @ {VARIANT}: {len(RUNS)} runs x "
          f"{config.cfg.MAX_EPOCH} epochs, seed {SEED} ===")
    print(f"=== Baseline (AUG=none, not re-run): CSRNet 0.2876, MobileCount 0.3850 ===")
    print(f"=== Output under {config.cfg.EXP_PATH} ===\n")

    results: list[dict] = []
    for run in RUNS:
        label = f"{run['net']}/{run['aug']}/bs{run['bs']}/lr{run['lr']:.1e}"
        print(f"\n{'=' * 60}\nRUN: {label}\n{'=' * 60}")
        try:
            results.append(train_one(run))
        except Exception as e:
            print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
            results.append({"net": run["net"], "aug": run["aug"],
                            "bs": run["bs"], "lr": run["lr"],
                            "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        out_dir = Path(config.cfg.EXP_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "aug_schemes_360p_results.json").open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}\nSUMMARY (val best_mae; baseline none: CSRNet 0.2876 / MobileCount 0.3850)\n{'=' * 60}")
    print(f'{"net":<13} {"aug":<10} {"bs":<4} {"lr":<10} {"log_para":<10} {"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r["net"]:<13} {r["aug"]:<10} {r["bs"]:<4} {r["lr"]:<10.1e} ERROR: {r["error"]}')
        else:
            print(f'{r["net"]:<13} {r["aug"]:<10} {r["batch_size"]:<4} {r["lr"]:<10.1e} '
                  f'{r["log_para"]:<10.2f} {r["best_mae"]:<12.4f} {r["best_mse"]:<12.4f} {r["duration_sec"]:>6.0f}s')


if __name__ == "__main__":
    main()
