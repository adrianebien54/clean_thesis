#!/usr/bin/env python3
"""Paper-augmentation comparison at 386x260 under the ORIGINAL (untuned) hyperparameters.

Three augmentation arms, both nets (6 runs, 1500 epochs each, seed 3035):

  noaug    AUG='raw'  on 386x260          — full image every epoch, zero augmentation
  mc80     AUG='none' on 386x260          — MobileCount-paper protocol: online random
                                            80%-sized crop per image per epoch
  csrnet9  AUG='raw'  on 386x260_csrnet9  — CSRNet-paper protocol: offline fixed dataset
                                            of 18 patches/image (4 quarters + 5 random,
                                            all mirrored), built by
                                            scripts/make_csrnet9_patches_386x260.py

Original HPs for both nets: Adam, weight_decay 1e-4, batch size 6, step LR gamma=0.995
per epoch; LR 1e-5 (CSRNet) / 1e-4 (MobileCount). LOG_PARA stays on the 772x519-anchored
scheme, so per net it is identical across arms: CSRNet 25.05, MobileCount 638.73. The
csrnet9 arm anchors at ANCHOR/4 because its train images are the 193x130 patches (1/4 the
area of 386x260) while cropping leaves per-pixel density unchanged.

Resumable: set SWEEP_DIR to relaunch into the same experiment dir — completed (net, arm)
cells in aug_paper_schemes_results.json are skipped, and a cell with a latest_state.pth
but no JSON record resumes from it (metrics already logged are recovered from tfevents).
ONLY_NET / ONLY_ARM env vars filter the run matrix; MAX_EPOCH overrides the budget
(smoke test: MAX_EPOCH=2).

Writes aug_paper_schemes_results.json under $SWEEP_DIR (default
./exp/aug_paper_schemes_386x260_<timestamp>/) incrementally.
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
config.cfg.EXP_PATH = os.environ.get(
    "SWEEP_DIR", f"./exp/aug_paper_schemes_386x260_{SWEEP_STAMP}")

SEED = 3035
VARIANT = "386x260"
ANCHOR = 772 * 519       # LOG_PARA_BASE_AREA anchor (re-anchored scheme)
FULL_AREA = 386 * 260

# original (untuned) hyperparameters; LOG_PARA bases as in run_aug_schemes_360p.py
NETS = {
    "CSRNet":      {"lr": 1e-5, "lp_base": 100.0},
    "MobileCount": {"lr": 1e-4, "lp_base": 2550.0},
}
# per net, identical across all three arms: CSRNet 25.05, MobileCount 638.73
EXPECTED_LOG_PARA = {n: p["lp_base"] * FULL_AREA / ANCHOR for n, p in NETS.items()}

# csrnet9 anchors at ANCHOR/4: its train images are 193x130 patches (1/4 the area of
# 386x260) but per-pixel density is unchanged by cropping, so LOG_PARA must equal the
# parent resolution's value.
ARMS = {
    "noaug":   {"data_path": f"./exp/data/Tenebrio/{VARIANT}",          "aug": "raw",  "base_area": ANCHOR},
    "mc80":    {"data_path": f"./exp/data/Tenebrio/{VARIANT}",          "aug": "none", "base_area": ANCHOR},
    "csrnet9": {"data_path": f"./exp/data/Tenebrio/{VARIANT}_csrnet9",  "aug": "raw",  "base_area": ANCHOR / 4},
}
ARM_ORDER = ["noaug", "mc80", "csrnet9"]

RESULTS_NAME = "aug_paper_schemes_results.json"

# cheap -> expensive: MobileCount first, csrnet9 (18x dataset) last per net
RUNS = [
    {"net": net, "arm": arm, **NETS[net], **ARMS[arm]}
    for net in ("MobileCount", "CSRNet")
    for arm in ARM_ORDER
    if net == os.environ.get("ONLY_NET", net)
    and arm == os.environ.get("ONLY_ARM", arm)
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

    config.cfg.NET = run["net"]
    config.cfg.LR = run["lr"]
    config.cfg.OPTIMIZER = "Adam"
    config.cfg.SEED = SEED
    config.cfg.EXP_NAME = (f"{VARIANT}_{run['net']}_Adam"
                           f"_lr{run['lr']:.1e}_seed{SEED}_bs6_{run['arm']}")

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

    seed_everything(SEED)

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

    print(f"  [{run['net']}/{run['arm']}] optimizer = {type(trainer.optimizer).__name__}  "
          f"lr={config.cfg.LR}  batch_size={cfg_data.TRAIN_BATCH_SIZE}  "
          f"log_para={cfg_data.LOG_PARA:.2f}  aug={cfg_data.AUG}  "
          f"weight_decay={config.cfg.WEIGHT_DECAY}  seed={SEED}  "
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
        "seed": SEED,
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
    checks = [
        (ROOT / f"exp/data/Tenebrio/{VARIANT}/train/img", 840),
        (ROOT / f"exp/data/Tenebrio/{VARIANT}_csrnet9/train/img", 15120),
    ]
    for img_dir, expected in checks:
        n = len(list(img_dir.glob("*.png"))) if img_dir.is_dir() else 0
        if n != expected:
            sys.exit(f"preflight: {img_dir} has {n} PNGs, expected {expected} "
                     f"(run scripts/make_csrnet9_patches_386x260.py first?)")
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
    done = {(r["net"], r["arm"]) for r in results if "error" not in r}

    print(f"=== Paper aug schemes (noaug/mc80/csrnet9) @ {VARIANT}, original HPs "
          f"(Adam, bs6, wd1e-4): {len(RUNS)} runs x {config.cfg.MAX_EPOCH} epochs, "
          f"seed {SEED} ===")
    print(f"=== LOG_PARA (772x519-anchored): "
          + "  ".join(f"{n} {lp:.2f}" for n, lp in EXPECTED_LOG_PARA.items()) + " ===")
    print(f"=== Output under {config.cfg.EXP_PATH} ===")
    if done:
        print(f"=== Already completed (skipped): {sorted(done)} ===")
    print()

    for run in RUNS:
        label = f"{run['net']}/{run['arm']}/lr{run['lr']:.1e}"
        if (run["net"], run["arm"]) in done:
            print(f"SKIP (done): {label}")
            continue
        print(f"\n{'=' * 60}\nRUN: {label}\n{'=' * 60}")
        try:
            results.append(train_one(run))
        except Exception as e:
            print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
            results.append({"net": run["net"], "arm": run["arm"], "lr": run["lr"],
                            "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        out_dir.mkdir(parents=True, exist_ok=True)
        with results_path.open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}\nSUMMARY (val best_mae)\n{'=' * 60}")
    print(f'{"net":<13} {"arm":<9} {"lr":<10} {"log_para":<10} '
          f'{"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r["net"]:<13} {r["arm"]:<9} {r["lr"]:<10.1e} ERROR: {r["error"]}')
        else:
            print(f'{r["net"]:<13} {r["arm"]:<9} {r["lr"]:<10.1e} '
                  f'{r["log_para"]:<10.2f} {r["best_mae"]:<12.4f} '
                  f'{r["best_mse"]:<12.4f} {r["duration_sec"]:>6.0f}s')


if __name__ == "__main__":
    main()
