#!/usr/bin/env python3
"""Continue the six seed-3035 800-epoch runs to 1500 epochs.

Part of the full optimizer x LR grid (seed 3035, bs6): the seeded sweeps
stopped at 800 epochs while the gamma=0.995/epoch decay only collapses the
LR (to 0.05% of base) around epoch 1500, and several Adam runs found their
best checkpoint at ep 734-750 — i.e. still improving. Resumes each run from
its latest_state.pth (epoch 801 onward) in its original directory:

  MobileCount lr=1e-3: exp/optimizer_seed_sweep_MobileCount_06-12_09-50/
  MobileCount lr=1e-4: exp/optimizer_seed_sweep_MobileCount_06-11_18-30/
  CSRNet      lr=1e-5: exp/optimizer_seed_sweep_CSRNet_06-12_01-15/

(x Adam/AdamW each, seed3035 runs only). Pre-800 metric history is recovered
from each run's tfevents and merged with the new records into
exp/lr_grid_continuations_results.json — the original per-sweep
optimizer_seed_sweep_results.json files are left untouched so the 800-epoch
3-seed sweeps stay internally consistent.

Re-runnable: completed (net, optimizer, lr) entries in the results file are
skipped; a crashed continuation resumes from wherever latest_state.pth is.

Set MAX_EPOCH env var to override the epoch budget (smoke test: MAX_EPOCH=802).
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
config.cfg.WEIGHT_DECAY = 1e-4
config.cfg.LR_SCHEDULE = "step"
config.cfg.LR_DECAY = 0.995
config.cfg.NUM_EPOCH_LR_DECAY = 1
config.cfg.LR_DECAY_START = -1

VARIANT = "386x260"
BATCH_SIZE = 6
SEED = 3035
RESULTS_PATH = Path("./exp/lr_grid_continuations_results.json")

RUNS = [
    {"net": "MobileCount", "lr": 1e-3, "optimizer": opt,
     "exp_path": "./exp/optimizer_seed_sweep_MobileCount_06-12_09-50"}
    for opt in ("Adam", "AdamW")
] + [
    {"net": "MobileCount", "lr": 1e-4, "optimizer": opt,
     "exp_path": "./exp/optimizer_seed_sweep_MobileCount_06-11_18-30"}
    for opt in ("Adam", "AdamW")
] + [
    {"net": "CSRNet", "lr": 1e-5, "optimizer": opt,
     "exp_path": "./exp/optimizer_seed_sweep_CSRNet_06-12_01-15"}
    for opt in ("Adam", "AdamW")
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


def recover_tfevents_records(run_dir: Path) -> list[dict]:
    """Read scalars already logged by the 800-epoch run from its tfevents file(s).

    Must be called BEFORE the resumed trainer opens a new writer in the same
    directory, so only pre-existing events are picked up.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    records: list[dict] = []
    for tag in ea.Tags()["scalars"]:
        for ev in ea.Scalars(tag):
            records.append({"tag": tag, "value": float(ev.value), "step": int(ev.step)})
    records.sort(key=lambda r: (r["step"], r["tag"]))
    return records


def train_one(run: dict) -> dict:
    from datasets.Tenebrio.setting import cfg_data
    cfg_data.DATA_PATH = f"./exp/data/Tenebrio/{VARIANT}"
    cfg_data.TRAIN_BATCH_SIZE = BATCH_SIZE
    cfg_data.VAL_BATCH_SIZE = 1

    config.cfg.NET = run["net"]
    config.cfg.LR = run["lr"]
    config.cfg.OPTIMIZER = run["optimizer"]
    config.cfg.SEED = SEED
    config.cfg.EXP_PATH = run["exp_path"]
    config.cfg.EXP_NAME = f"{VARIANT}_{run['net']}_{run['optimizer']}_seed{SEED}_bs{BATCH_SIZE}"

    run_dir = Path(run["exp_path"]) / config.cfg.EXP_NAME
    resume_path = run_dir / "latest_state.pth"
    if not resume_path.exists():
        raise FileNotFoundError(f"no latest_state.pth in {run_dir}")
    old_records = recover_tfevents_records(run_dir)
    config.cfg.RESUME = True
    config.cfg.RESUME_PATH = str(resume_path)
    print(f"  [{run['net']}/{run['optimizer']}] resuming from {resume_path} "
          f"({len(old_records)} pre-existing metric records recovered)")

    seed_everything(SEED)

    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))
    resumed_from_epoch = trainer.epoch + 1  # 1-based epoch the run continues at
    print(f"  [{run['net']}/{run['optimizer']}] resumed at epoch {resumed_from_epoch}, "
          f"train_record = {trainer.train_record}")
    print(f"  [{run['net']}/{run['optimizer']}] optimizer = {type(trainer.optimizer).__name__}  "
          f"lr={config.cfg.LR}  current_lr={trainer.optimizer.param_groups[0]['lr']:.3e}  "
          f"weight_decay={config.cfg.WEIGHT_DECAY}")

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
        "duration_sec_note": "duration covers only the resumed segment",
        "resumed_from_epoch": resumed_from_epoch,
        "metrics": old_records + rec_writer.records,
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
    results: list[dict] = []
    if RESULTS_PATH.exists():
        # Drop stale error entries from earlier attempts; those runs re-run below.
        results = [r for r in json.load(RESULTS_PATH.open()) if "error" not in r]
    done = {(r["net"], r["optimizer"], r["lr"]) for r in results
            if r.get("max_epoch") == config.cfg.MAX_EPOCH}
    print(f"=== Continuing {len(RUNS)} seed-{SEED} runs to {config.cfg.MAX_EPOCH} epochs ===")
    print(f"=== Already complete: {sorted(done)} ===")

    sweep_t0 = time.time()
    for run in RUNS:
        label = f"{run['net']}/{run['optimizer']}/lr{run['lr']:.0e}"
        if (run["net"], run["optimizer"], run["lr"]) in done:
            print(f"\n[{label}] already in results — skipping")
            continue
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

        with RESULTS_PATH.open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n=== Continuation total time: {(time.time() - sweep_t0) / 60:.1f} min ===")
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
