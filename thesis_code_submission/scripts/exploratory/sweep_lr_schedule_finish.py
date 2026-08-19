#!/usr/bin/env python3
"""Finish the interrupted LR-schedule sweep in exp/lr_schedule_06-04_08-22/.

The original run (scripts/sweep_lr_schedule.py) completed `constant` and
`cosine` but died during `step_g0.999` (stopped ~ep739, never written to
results) and never started `step_g0.9985`. This script re-runs only those two
step schedules into the SAME experiment dir, seeding `results` from the
existing lr_schedule_results.json so the final file holds all four in order.

Everything else is identical to the original sweep: MobileCount, Tenebrio
386x260, Adam, base LR 1e-4, batch 6, weight_decay 1e-4, 2000 epochs.
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

# Reuse the existing sweep directory rather than minting a new timestamp.
EXP_PATH = "./exp/lr_schedule_06-04_08-22"
config.cfg.EXP_PATH = EXP_PATH

VARIANT = "386x260"
BATCH_SIZE = 6

# Only the schedules that did not complete in the original run.
SCHEDULES: list[tuple[str, dict]] = [
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
    out_dir = Path(EXP_PATH)
    results_path = out_dir / "lr_schedule_results.json"

    # Seed with the already-completed runs (constant, cosine) so the merged
    # file ends up holding all four schedules.
    results: list[dict] = []
    if results_path.exists():
        results = json.load(results_path.open())
    done = {r.get("schedule") for r in results}
    print(f"=== Finishing LR-schedule sweep in {EXP_PATH} ===")
    print(f"=== Already complete: {sorted(done)} ===")

    sweep_t0 = time.time()
    for label, overrides in SCHEDULES:
        if label in done:
            print(f"\n[{label}] already in results — skipping")
            continue
        print(f"\n{'=' * 60}\nSCHEDULE: {label}  {overrides}\n{'=' * 60}")
        try:
            results.append(train_one(label, overrides))
        except Exception as e:
            print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
            results.append({"schedule": label, "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        out_dir.mkdir(parents=True, exist_ok=True)
        with results_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n=== Finish total time: {(time.time() - sweep_t0) / 60:.1f} min ===")
    print(f"\n{'=' * 60}\nSUMMARY (all schedules)\n{'=' * 60}")
    print(f'{"schedule":<14} {"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r["schedule"]:<14} ERROR: {r["error"]}')
        else:
            dur = r.get("duration_sec", float("nan"))
            print(f'{r["schedule"]:<14} {r["best_mae"]:<12.4f} {r["best_mse"]:<12.4f} '
                  f'{dur:>6.0f}s')


if __name__ == "__main__":
    main()
