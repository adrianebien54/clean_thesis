#!/usr/bin/env python3
"""Finish the interrupted bestcombo_resolutions_06-24_11-49 sweep: the two 1544x1038 cells.

The 10-cell sweep from scripts/run_best_combo_resolutions.py (5 resolutions x 2 nets,
1500 epochs each, seed 3035, Tenebrio) died on 2026-06-25 ~15:43 during the highest
resolution. 8 cells completed; the two 1544x1038 cells remain:

  CSRNet      AdamW lr=4e-5 bs1  lp_base=100   -> INTERRUPTED at epoch 327/1500
                                                  resume from latest_state.pth (epochs 328->1500)
  MobileCount AdamW lr=1e-3 bs6  lp_base=2550  -> never started, run fresh (epochs 0->1500)

This writes BACK INTO the existing sweep dir and APPENDS the two cells to the existing
bestcombo_resolutions_results.json, so the sweep ends with all 10 cells recorded. It is
idempotent: a cell already present (by variant_net) in the JSON is skipped, so a relaunch
after a crash is safe (the resumed cell picks up from its rewritten latest_state.pth).

Config / harness mirror scripts/run_best_combo_resolutions.py; the resume wiring mirrors
scripts/continue_seed3035_to1500.py.

Env overrides:
  MAX_EPOCH      epoch budget (default 1500; smoke test: e.g. 329 -> 1 resumed epoch)
  BESTCOMBO_DIR  sweep dir to write into (default the real interrupted dir)
  ONLY_NET       run only this net's cell (CSRNet | MobileCount), for smoke tests
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

# --- module-level config: identical to run_best_combo_resolutions.py:46-58 ---
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

# Write back into the existing (interrupted) sweep dir -- do NOT timestamp a new one.
config.cfg.EXP_PATH = os.environ.get(
    "BESTCOMBO_DIR", "./exp/bestcombo_resolutions_06-24_11-49"
)

SEED = 3035
ANCHOR = 772 * 519  # LOG_PARA_BASE_AREA for both nets (re-anchored scheme)

# Only the two missing 1544x1038 cells, in sweep order (CSRNet before MobileCount).
RUNS = [
    {"net": "CSRNet", "variant": "1544x1038", "optimizer": "AdamW",
     "lr": 4e-5, "bs": 1, "lp_base": 100.0, "resume": True},
    {"net": "MobileCount", "variant": "1544x1038", "optimizer": "AdamW",
     "lr": 1e-3, "bs": 6, "lp_base": 2550.0, "resume": False},
]

_only = os.environ.get("ONLY_NET")
if _only:
    RUNS = [r for r in RUNS if r["net"] == _only]


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
    """Read scalars already logged by the interrupted run from its tfevents file(s).

    Must be called BEFORE the resumed trainer opens a new writer in the same
    directory, so only pre-existing events are picked up. Mirrors
    scripts/continue_seed3035_to1500.py:101-116.
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
    cfg_data.DATA_PATH = f"./exp/data/Tenebrio/{run['variant']}"
    cfg_data.TRAIN_BATCH_SIZE = run["bs"]
    cfg_data.VAL_BATCH_SIZE = 1
    cfg_data.LOG_PARA_BASE = run["lp_base"]   # 100 (CSRNet) / 2550 (MobileCount)
    cfg_data.LOG_PARA_BASE_AREA = ANCHOR      # 772*519, read by _compute_log_para()

    config.cfg.NET = run["net"]
    config.cfg.LR = run["lr"]
    config.cfg.OPTIMIZER = run["optimizer"]
    config.cfg.SEED = SEED
    config.cfg.EXP_NAME = (f"{run['variant']}_{run['net']}_{run['optimizer']}"
                           f"_lr{run['lr']:.1e}_seed{SEED}_bs{run['bs']}")

    run_dir = Path(config.cfg.EXP_PATH) / config.cfg.EXP_NAME
    if run["resume"]:
        resume_path = run_dir / "latest_state.pth"
        if not resume_path.exists():
            raise FileNotFoundError(f"no latest_state.pth in {run_dir}")
        old_records = recover_tfevents_records(run_dir)   # pre-resume TB scalars
        config.cfg.RESUME = True
        config.cfg.RESUME_PATH = str(resume_path)
        print(f"  [{run['net']}/{run['variant']}] resuming from {resume_path} "
              f"({len(old_records)} pre-existing metric records recovered)")
    else:
        old_records = []
        config.cfg.RESUME = False          # cfg persists across cells in-process; reset it
        config.cfg.RESUME_PATH = ""

    seed_everything(SEED)

    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))
    resumed_from_epoch = trainer.epoch + 1  # 1-based epoch the loop continues at
    print(f"  [{run['net']}/{run['variant']}/bs{run['bs']}/lr{run['lr']:.1e}] "
          f"optimizer = {type(trainer.optimizer).__name__}  lr={config.cfg.LR}  "
          f"current_lr={trainer.optimizer.param_groups[0]['lr']:.3e}  "
          f"batch_size={cfg_data.TRAIN_BATCH_SIZE}  log_para={cfg_data.LOG_PARA:.2f}  "
          f"weight_decay={config.cfg.WEIGHT_DECAY}  seed={SEED}  "
          f"resume={config.cfg.RESUME}  start_epoch={resumed_from_epoch}")

    rec_writer = RecordingWriter(trainer.writer)
    trainer.writer = rec_writer

    t0 = time.time()
    trainer.forward()
    duration = time.time() - t0

    result = {
        "net": run["net"],
        "optimizer": run["optimizer"],
        "seed": SEED,
        "variant": run["variant"],
        "batch_size": run["bs"],
        "lr": run["lr"],
        "log_para_base": run["lp_base"],
        "log_para_base_area": ANCHOR,
        "log_para": float(cfg_data.LOG_PARA),
        "weight_decay": config.cfg.WEIGHT_DECAY,
        "max_epoch": config.cfg.MAX_EPOCH,
        "duration_sec": duration,
        "metrics": old_records + rec_writer.records,
        "best_mae": float(trainer.train_record["best_mae"]),
        "best_mse": float(trainer.train_record["best_mse"]),
        "best_model_name": trainer.train_record["best_model_name"],
    }
    if run["resume"]:
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
    out_dir = Path(config.cfg.EXP_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "bestcombo_resolutions_results.json"

    results: list[dict] = json.load(results_path.open()) if results_path.exists() else []
    done = {f"{r.get('variant', '')}_{r.get('net', '')}"
            for r in results if "error" not in r}

    print(f"=== Finishing bestcombo sweep: {len(RUNS)} cell(s) x "
          f"{config.cfg.MAX_EPOCH} epochs, seed {SEED} ===")
    print(f"=== Writing into {config.cfg.EXP_PATH} "
          f"({len(results)} cells already recorded) ===\n")

    for run in RUNS:
        key = f"{run['variant']}_{run['net']}"
        label = f"{run['net']}/{run['variant']}/bs{run['bs']}/lr{run['lr']:.1e}"
        if key in done:
            print(f"\n{'=' * 60}\nSKIP (already in results.json): {label}\n{'=' * 60}")
            continue
        print(f"\n{'=' * 60}\nRUN: {label}\n{'=' * 60}")
        try:
            results.append(train_one(run))
        except Exception as e:
            print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
            results.append({"net": run["net"], "variant": run["variant"],
                            "bs": run["bs"], "lr": run["lr"],
                            "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        with results_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}\nSUMMARY ({len(results)} cells in {results_path.name})\n{'=' * 60}")
    print(f'{"net":<13} {"variant":<11} {"bs":<4} {"lr":<10} {"log_para":<10} '
          f'{"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r.get("net"):<13} {r.get("variant"):<11} ERROR: {r["error"]}')
        else:
            print(f'{r["net"]:<13} {r["variant"]:<11} {r["batch_size"]:<4} {r["lr"]:<10.1e} '
                  f'{r["log_para"]:<10.2f} {r["best_mae"]:<12.4f} {r["best_mse"]:<12.4f} '
                  f'{r["duration_sec"]:>6.0f}s')


if __name__ == "__main__":
    main()
