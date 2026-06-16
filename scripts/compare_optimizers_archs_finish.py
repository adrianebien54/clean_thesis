#!/usr/bin/env python3
"""Finish the interrupted optimizer x architecture sweep in
exp/optimizer_arch_compare_06-10_18-12/.

The original run (scripts/compare_optimizers_archs.py) completed both
MobileCount runs but died during CSRNet/Adam (~ep771, never written to
results) and never started CSRNet/AdamW. This script resumes CSRNet/Adam
from its latest_state.pth (epoch 772 onward), recovers the pre-crash
metric history from the run's tfevents file, then runs CSRNet/AdamW from
scratch — appending both to the existing optimizer_arch_compare_results.json.

Everything else is identical to the original sweep: Tenebrio 386x260,
CSRNet lr=1e-5, step decay gamma=0.995/epoch, 1500 epochs, batch=6.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

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
config.cfg.PRINT_FREQ = 50
config.cfg.PRE_GCC = False
config.cfg.WEIGHT_DECAY = 1e-4
config.cfg.LR_SCHEDULE = "step"
config.cfg.LR_DECAY = 0.995
config.cfg.NUM_EPOCH_LR_DECAY = 1
config.cfg.LR_DECAY_START = -1

# Reuse the existing sweep directory rather than minting a new timestamp.
EXP_PATH = "./exp/optimizer_arch_compare_06-10_18-12"
config.cfg.EXP_PATH = EXP_PATH

VARIANT = "386x260"
BATCH_SIZE = 6

# Only the runs that did not complete in the original sweep.
RUNS = [
    {"net": "CSRNet", "lr": 1e-5, "optimizer": "Adam", "resume": True},
    {"net": "CSRNet", "lr": 1e-5, "optimizer": "AdamW", "resume": False},
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


def recover_tfevents_records(run_dir: Path) -> list[dict]:
    """Read scalars logged by the crashed run from its tfevents file(s).

    Must be called BEFORE the resumed trainer opens a new writer in the same
    directory, so only pre-crash events are picked up.
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
    config.cfg.EXP_NAME = f"{VARIANT}_{run['net']}_{run['optimizer']}_bs{BATCH_SIZE}"

    run_dir = Path(EXP_PATH) / config.cfg.EXP_NAME
    old_records: list[dict] = []
    resumed_from_epoch = None
    if run["resume"]:
        resume_path = run_dir / "latest_state.pth"
        old_records = recover_tfevents_records(run_dir)
        config.cfg.RESUME = True
        config.cfg.RESUME_PATH = str(resume_path)
        print(f"  [{run['net']}/{run['optimizer']}] resuming from {resume_path} "
              f"({len(old_records)} pre-crash metric records recovered)")
    else:
        config.cfg.RESUME = False

    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))
    if run["resume"]:
        resumed_from_epoch = trainer.epoch + 1  # 1-based epoch the run continues at
        print(f"  [{run['net']}/{run['optimizer']}] resumed train_record = {trainer.train_record}")
    print(f"  [{run['net']}/{run['optimizer']}] optimizer = {type(trainer.optimizer).__name__}  "
          f"lr={config.cfg.LR}  weight_decay={config.cfg.WEIGHT_DECAY}")

    rec_writer = RecordingWriter(trainer.writer)
    trainer.writer = rec_writer

    t0 = time.time()
    trainer.forward()
    duration = time.time() - t0

    result = {
        "net": run["net"],
        "optimizer": run["optimizer"],
        "variant": VARIANT,
        "batch_size": BATCH_SIZE,
        "lr": run["lr"],
        "weight_decay": config.cfg.WEIGHT_DECAY,
        "max_epoch": config.cfg.MAX_EPOCH,
        "duration_sec": duration,
        "metrics": old_records + rec_writer.records,
        "best_mae": float(trainer.train_record["best_mae"]),
        "best_mse": float(trainer.train_record["best_mse"]),
        "best_model_name": trainer.train_record["best_model_name"],
    }
    if resumed_from_epoch is not None:
        result["resumed_from_epoch"] = resumed_from_epoch
        result["duration_sec_note"] = "duration covers only the resumed segment"
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
    results_path = out_dir / "optimizer_arch_compare_results.json"

    # Seed with the already-completed MobileCount runs so the merged file
    # ends up holding all four architecture/optimizer combinations.
    results: list[dict] = []
    if results_path.exists():
        # Drop stale error entries from earlier attempts; those runs re-run below.
        results = [r for r in json.load(results_path.open()) if "error" not in r]
    done = {(r.get("net"), r.get("optimizer")) for r in results}
    print(f"=== Finishing optimizer x architecture sweep in {EXP_PATH} ===")
    print(f"=== Already complete: {sorted(done)} ===")

    sweep_t0 = time.time()
    for run in RUNS:
        label = f"{run['net']}/{run['optimizer']}"
        if (run["net"], run["optimizer"]) in done:
            print(f"\n[{label}] already in results — skipping")
            continue
        print(f"\n{'=' * 60}\nRUN: {label}  (lr={run['lr']}, resume={run['resume']})\n{'=' * 60}")
        try:
            results.append(train_one(run))
        except Exception as e:
            print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
            results.append({"net": run["net"], "optimizer": run["optimizer"],
                            "lr": run["lr"], "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        out_dir.mkdir(parents=True, exist_ok=True)
        with results_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n=== Finish total time: {(time.time() - sweep_t0) / 60:.1f} min ===")
    print(f"\n{'=' * 60}\nSUMMARY (all runs)\n{'=' * 60}")
    print(f'{"net":<13} {"optimizer":<10} {"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r["net"]:<13} {r["optimizer"]:<10} ERROR: {r["error"]}')
        else:
            print(f'{r["net"]:<13} {r["optimizer"]:<10} {r["best_mae"]:<12.4f} '
                  f'{r["best_mse"]:<12.4f} {r["duration_sec"]:>6.0f}s')


if __name__ == "__main__":
    main()
