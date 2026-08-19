#!/usr/bin/env python3
"""Compare Adam vs AdamW for MobileCount on the Tenebrio 386x260 variant.

Two runs, identical in every way except the optimizer
(500 epochs, LR=1e-4, batch=6, validate every epoch).

Writes optimizer_compare_results.json under
./exp/optimizer_compare_<timestamp>/ with best val MAE/MSE per optimizer.
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
config.cfg.MAX_EPOCH = 500
config.cfg.VAL_FREQ = 1
config.cfg.VAL_DENSE_START = -1
config.cfg.PRINT_FREQ = 50
config.cfg.LR = 1e-4
config.cfg.PRE_GCC = False
config.cfg.RESUME = False
config.cfg.WEIGHT_DECAY = 1e-4

SWEEP_STAMP = time.strftime("%m-%d_%H-%M", time.localtime())
config.cfg.EXP_PATH = f"./exp/optimizer_compare_{SWEEP_STAMP}"

VARIANT = "386x260"
BATCH_SIZE = 6
OPTIMIZERS = ["Adam", "AdamW"]


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


def train_one(optimizer_name: str) -> dict:
    from datasets.Tenebrio.setting import cfg_data
    cfg_data.DATA_PATH = f"./exp/data/Tenebrio/{VARIANT}"
    cfg_data.TRAIN_BATCH_SIZE = BATCH_SIZE
    cfg_data.VAL_BATCH_SIZE = 1

    config.cfg.OPTIMIZER = optimizer_name
    config.cfg.EXP_NAME = f"{VARIANT}_{optimizer_name}_bs{BATCH_SIZE}"

    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))
    print(f"  [{optimizer_name}] optimizer = {type(trainer.optimizer).__name__}  "
          f"lr={config.cfg.LR}  weight_decay={config.cfg.WEIGHT_DECAY}")

    rec_writer = RecordingWriter(trainer.writer)
    trainer.writer = rec_writer

    t0 = time.time()
    trainer.forward()
    duration = time.time() - t0

    result = {
        "optimizer": optimizer_name,
        "variant": VARIANT,
        "batch_size": BATCH_SIZE,
        "lr": config.cfg.LR,
        "weight_decay": config.cfg.WEIGHT_DECAY,
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
    print(f"=== Optimizer comparison — MobileCount on {VARIANT} × {config.cfg.MAX_EPOCH} epochs ===")
    print(f"=== Output under {config.cfg.EXP_PATH} ===\n")

    results: list[dict] = []
    for opt_name in OPTIMIZERS:
        print(f"\n{'=' * 60}\nOPTIMIZER: {opt_name}\n{'=' * 60}")
        try:
            results.append(train_one(opt_name))
        except Exception as e:
            print(f"  [{opt_name}] FAILED: {type(e).__name__}: {e}")
            results.append({"optimizer": opt_name, "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        out_dir = Path(config.cfg.EXP_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "optimizer_compare_results.json").open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f'{"optimizer":<10} {"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r["optimizer"]:<10} ERROR: {r["error"]}')
        else:
            print(f'{r["optimizer"]:<10} {r["best_mae"]:<12.4f} {r["best_mse"]:<12.4f} '
                  f'{r["duration_sec"]:>6.0f}s')


if __name__ == "__main__":
    main()
