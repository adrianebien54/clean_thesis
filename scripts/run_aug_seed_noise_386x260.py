#!/usr/bin/env python3
"""Seed-noise estimation for the paper-augmentation comparison at 386x260.

Re-runs cells of run_aug_paper_schemes_386x260.py under IDENTICAL configuration
(original untuned HPs: Adam, wd 1e-4, bs 6, step gamma=0.995, 1500 epochs,
772x519-anchored LOG_PARA) with additional seeds, so the across-seed spread of
the best-val-MAE statistic is measured in the exact regime the thesis claims
"differences are within seed noise" for. Supersedes the 06-11/06-12 seed sweeps
(800 epochs, un-anchored LOG_PARA=100), whose config no longer matches anything
reported.

Default matrix: MobileCount x {noaug, mc80} x seeds {42, 7, 1234, 9001}
(8 runs, ~1 h each). Seed 3035 already exists in
exp/aug_paper_schemes_386x260_07-04/ and is merged at analysis time, giving
n=5 per cell. Runs are ordered seed-major so an interrupted sweep still yields
complete noaug/mc80 pairs per seed (the tie claim is a paired comparison).

Env overrides:
  SEEDS     comma list (default "42,7,1234,9001")
  NETS      comma list (default "MobileCount"; add CSRNet for ~3-4 h/run cells)
  ARMS      comma list (default "noaug,mc80"; csrnet9 costs ~9 h/run MobileCount)
  MAX_EPOCH epoch budget (smoke test: MAX_EPOCH=2)
  SWEEP_DIR relaunch into an existing dir; completed (net, arm, seed) cells in
            aug_seed_noise_results.json are skipped, half-finished cells resume
            from latest_state.pth.

Writes aug_seed_noise_results.json under $SWEEP_DIR (default
./exp/aug_seed_noise_386x260_<timestamp>/) incrementally.
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
config.cfg.EXP_PATH = os.environ.get(
    "SWEEP_DIR", f"./exp/aug_seed_noise_386x260_{SWEEP_STAMP}")

VARIANT = "386x260"
ANCHOR = 772 * 519       # LOG_PARA_BASE_AREA anchor (re-anchored scheme)
FULL_AREA = 386 * 260

# original (untuned) hyperparameters; identical to run_aug_paper_schemes_386x260.py
NETS = {
    "CSRNet":      {"lr": 1e-5, "lp_base": 100.0},
    "MobileCount": {"lr": 1e-4, "lp_base": 2550.0},
}
EXPECTED_LOG_PARA = {n: p["lp_base"] * FULL_AREA / ANCHOR for n, p in NETS.items()}

ARMS = {
    "noaug":   {"data_path": f"./exp/data/Tenebrio/{VARIANT}",          "aug": "raw",  "base_area": ANCHOR},
    "mc80":    {"data_path": f"./exp/data/Tenebrio/{VARIANT}",          "aug": "none", "base_area": ANCHOR},
    "csrnet9": {"data_path": f"./exp/data/Tenebrio/{VARIANT}_csrnet9",  "aug": "raw",  "base_area": ANCHOR / 4},
}

SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,7,1234,9001").split(",")]
RUN_NETS = [n.strip() for n in os.environ.get("NETS", "MobileCount").split(",")]
RUN_ARMS = [a.strip() for a in os.environ.get("ARMS", "noaug,mc80").split(",")]
assert all(n in NETS for n in RUN_NETS), f"unknown net in NETS={RUN_NETS}"
assert all(a in ARMS for a in RUN_ARMS), f"unknown arm in ARMS={RUN_ARMS}"

RESULTS_NAME = "aug_seed_noise_results.json"

# seed-major: an interrupted sweep still yields complete per-seed arm pairs
RUNS = [
    {"net": net, "arm": arm, "seed": seed, **NETS[net], **ARMS[arm]}
    for net in RUN_NETS
    for seed in SEEDS
    for arm in RUN_ARMS
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
    """Read scalars already logged by an interrupted run from its tfevents file(s).

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
    cfg_data.DATA_PATH = run["data_path"]
    cfg_data.TRAIN_BATCH_SIZE = 6
    cfg_data.VAL_BATCH_SIZE = 1
    cfg_data.LOG_PARA_BASE = run["lp_base"]
    cfg_data.LOG_PARA_BASE_AREA = run["base_area"]
    cfg_data.AUG = run["aug"]

    seed = run["seed"]
    config.cfg.NET = run["net"]
    config.cfg.LR = run["lr"]
    config.cfg.OPTIMIZER = "Adam"
    config.cfg.SEED = seed
    config.cfg.EXP_NAME = (f"{VARIANT}_{run['net']}_Adam"
                           f"_lr{run['lr']:.1e}_seed{seed}_bs6_{run['arm']}")

    # resume a half-finished cell from its latest_state.pth; RESUME is sticky module
    # state, so it must be reset explicitly for fresh runs
    run_dir = Path(config.cfg.EXP_PATH) / config.cfg.EXP_NAME
    latest_state = run_dir / "latest_state.pth"
    resumed = latest_state.exists()
    old_records: list[dict] = []
    if resumed:
        old_records = recover_tfevents_records(run_dir)
        config.cfg.RESUME = True
        config.cfg.RESUME_PATH = str(latest_state)
        print(f"  resuming from {latest_state} "
              f"({len(old_records)} pre-existing metric records recovered)")
    else:
        config.cfg.RESUME = False

    seed_everything(seed)

    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))

    expected_lp = EXPECTED_LOG_PARA[run["net"]]
    assert abs(cfg_data.LOG_PARA - expected_lp) < 0.01, (
        f"LOG_PARA anchoring wrong: got {cfg_data.LOG_PARA:.4f}, "
        f"expected {expected_lp:.4f} for {run['net']}/{run['arm']}")

    print(f"  [{run['net']}/{run['arm']}/seed{seed}] "
          f"optimizer = {type(trainer.optimizer).__name__}  "
          f"lr={config.cfg.LR}  batch_size={cfg_data.TRAIN_BATCH_SIZE}  "
          f"log_para={cfg_data.LOG_PARA:.2f}  aug={cfg_data.AUG}  "
          f"weight_decay={config.cfg.WEIGHT_DECAY}  seed={seed}  "
          f"train_iters/epoch={len(trainer.train_loader)}")

    rec_writer = RecordingWriter(trainer.writer)
    trainer.writer = rec_writer

    t0 = time.time()
    trainer.forward()
    duration = time.time() - t0

    result = {
        "net": run["net"],
        "arm": run["arm"],
        "aug": run["aug"],
        "data_path": run["data_path"],
        "optimizer": "Adam",
        "seed": seed,
        "variant": VARIANT,
        "batch_size": 6,
        "lr": run["lr"],
        "log_para_base": run["lp_base"],
        "log_para_base_area": run["base_area"],
        "log_para": float(cfg_data.LOG_PARA),
        "weight_decay": config.cfg.WEIGHT_DECAY,
        "max_epoch": config.cfg.MAX_EPOCH,
        "resumed": resumed,
        "duration_sec": duration,
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


def preflight() -> None:
    checks = [(ROOT / f"exp/data/Tenebrio/{VARIANT}/train/img", 840)]
    if "csrnet9" in RUN_ARMS:
        checks.append((ROOT / f"exp/data/Tenebrio/{VARIANT}_csrnet9/train/img", 15120))
    for img_dir, expected in checks:
        n = len(list(img_dir.glob("*.png"))) if img_dir.is_dir() else 0
        if n != expected:
            sys.exit(f"preflight: {img_dir} has {n} PNGs, expected {expected}")
    if "csrnet9" in RUN_ARMS:
        for split in ("val", "test"):
            if not (ROOT / f"exp/data/Tenebrio/{VARIANT}_csrnet9/{split}/img").is_dir():
                sys.exit(f"preflight: csrnet9 {split} symlink does not resolve")


def main() -> None:
    preflight()

    out_dir = Path(config.cfg.EXP_PATH)
    results_path = out_dir / RESULTS_NAME
    results: list[dict] = []
    if results_path.exists():
        results = json.load(results_path.open())
    done = {(r["net"], r["arm"], r["seed"]) for r in results if "error" not in r}

    print(f"=== Aug seed-noise sweep @ {VARIANT}, original HPs (Adam, bs6, wd1e-4): "
          f"{len(RUNS)} runs x {config.cfg.MAX_EPOCH} epochs ===")
    print(f"=== nets={RUN_NETS}  arms={RUN_ARMS}  seeds={SEEDS} ===")
    print(f"=== LOG_PARA (772x519-anchored): "
          + "  ".join(f"{n} {lp:.2f}" for n, lp in EXPECTED_LOG_PARA.items()) + " ===")
    print(f"=== Output under {config.cfg.EXP_PATH} ===")
    if done:
        print(f"=== Already completed (skipped): {sorted(done)} ===")
    print()

    for run in RUNS:
        label = f"{run['net']}/{run['arm']}/seed{run['seed']}"
        if (run["net"], run["arm"], run["seed"]) in done:
            print(f"SKIP (done): {label}")
            continue
        print(f"\n{'=' * 60}\nRUN: {label}\n{'=' * 60}")
        try:
            results.append(train_one(run))
        except Exception as e:
            print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
            results.append({"net": run["net"], "arm": run["arm"], "seed": run["seed"],
                            "lr": run["lr"], "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        out_dir.mkdir(parents=True, exist_ok=True)
        with results_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}\nSUMMARY (val best_mae)\n{'=' * 60}")
    print(f'{"net":<13} {"arm":<9} {"seed":<6} {"log_para":<10} '
          f'{"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r["net"]:<13} {r["arm"]:<9} {r["seed"]:<6} ERROR: {r["error"]}')
        else:
            print(f'{r["net"]:<13} {r["arm"]:<9} {r["seed"]:<6} '
                  f'{r["log_para"]:<10.2f} {r["best_mae"]:<12.4f} '
                  f'{r["best_mse"]:<12.4f} {r["duration_sec"]:>6.0f}s')


if __name__ == "__main__":
    main()
