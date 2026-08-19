#!/usr/bin/env python3
"""OFAT: tune the LR-schedule *shape* for MobileCount on Tenebrio 386x260.

One factor varied (LR_SCHEDULE shape); everything else frozen:
  optimizer = Adam, base LR = 1e-4, batch = 6, 2000 epochs, weight_decay = 1e-4.

Schedule candidates (all start from base LR = 1e-4):
  - constant      : LR held at 1e-4 the whole run
  - cosine        : 1e-4 -> 0 over 2000 epochs (CosineAnnealingLR)
  - step_g0.999   : StepLR gamma=0.999/epoch  -> ends ~1.35e-5 (13.5% of base)
  - step_g0.9985  : StepLR gamma=0.9985/epoch -> ends ~5.0e-6  (5% of base)

Writes lr_schedule_results.json under ./exp/lr_schedule_<timestamp>/ with best
val MAE/MSE per schedule. Each run logs its EFFECTIVE config to its .txt.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import config

config.cfg.NET = "MobileCount"
config.cfg.DATASET = "Tenebrio"
config.cfg.GPU_ID = [0]
config.cfg.MAX_EPOCH = 2000
config.cfg.VAL_FREQ = 1
config.cfg.VAL_DENSE_START = -1
config.cfg.PRINT_FREQ = 200
config.cfg.LR = 1e-4
config.cfg.OPTIMIZER = "Adam"
config.cfg.WEIGHT_DECAY = 1e-4
config.cfg.LR_DECAY_START = -1
config.cfg.PRE_GCC = False
config.cfg.RESUME = False

SWEEP_STAMP = time.strftime("%m-%d_%H-%M", time.localtime())
config.cfg.EXP_PATH = f"./exp/lr_schedule_{SWEEP_STAMP}"

VARIANT = "386x260"
BATCH_SIZE = 6

# (label, cfg overrides for the schedule shape)
SCHEDULES: list[tuple[str, dict]] = [
    ("constant",     {"LR_SCHEDULE": "constant"}),
    ("cosine",       {"LR_SCHEDULE": "cosine", "LR_MIN": 0.0}),
    ("step_g0.999",  {"LR_SCHEDULE": "step", "LR_DECAY": 0.999,  "NUM_EPOCH_LR_DECAY": 1}),
    ("step_g0.9985", {"LR_SCHEDULE": "step", "LR_DECAY": 0.9985, "NUM_EPOCH_LR_DECAY": 1}),
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


def train_one(label: str, overrides: dict) -> dict:
    from datasets.Tenebrio.setting import cfg_data
    cfg_data.DATA_PATH = f"./exp/data/Tenebrio/{VARIANT}"
    cfg_data.TRAIN_BATCH_SIZE = BATCH_SIZE
    cfg_data.VAL_BATCH_SIZE = 1

    # reset the schedule-related knobs each run, then apply this run's overrides
    config.cfg.LR_SCHEDULE = "step"
    config.cfg.LR_DECAY = 0.995
    config.cfg.NUM_EPOCH_LR_DECAY = 1
    config.cfg.LR_MIN = 0.0
    for k, v in overrides.items():
        setattr(config.cfg, k, v)

    config.cfg.EXP_NAME = f"{VARIANT}_sched-{label}_lr1e-04"

    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))
    print(f"  [{label}] schedule={config.cfg.LR_SCHEDULE} "
          f"optimizer={type(trainer.optimizer).__name__} base_lr={config.cfg.LR} "
          f"scheduler={type(trainer.scheduler).__name__}")

    rec_writer = RecordingWriter(trainer.writer)
    trainer.writer = rec_writer

    t0 = time.time()
    trainer.forward()
    duration = time.time() - t0

    result = {
        "schedule": label,
        "overrides": overrides,
        "variant": VARIANT,
        "batch_size": BATCH_SIZE,
        "base_lr": config.cfg.LR,
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
    print(f"=== LR-schedule shape sweep — MobileCount on {VARIANT} × {config.cfg.MAX_EPOCH} epochs ===")
    print(f"=== {len(SCHEDULES)} shapes, base LR={config.cfg.LR}, optimizer={config.cfg.OPTIMIZER} ===")
    print(f"=== Output under {config.cfg.EXP_PATH} ===\n")

    results: list[dict] = []
    sweep_t0 = time.time()
    for label, overrides in SCHEDULES:
        print(f"\n{'=' * 60}\nSCHEDULE: {label}  {overrides}\n{'=' * 60}")
        try:
            results.append(train_one(label, overrides))
        except Exception as e:
            print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
            results.append({"schedule": label, "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        out_dir = Path(config.cfg.EXP_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "lr_schedule_results.json").open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n=== Sweep total time: {(time.time() - sweep_t0) / 60:.1f} min ===")
    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f'{"schedule":<14} {"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r["schedule"]:<14} ERROR: {r["error"]}')
        else:
            print(f'{r["schedule"]:<14} {r["best_mae"]:<12.4f} {r["best_mse"]:<12.4f} '
                  f'{r["duration_sec"]:>6.0f}s')


if __name__ == "__main__":
    main()
