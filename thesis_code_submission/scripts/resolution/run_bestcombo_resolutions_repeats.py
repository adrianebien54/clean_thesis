#!/usr/bin/env python3
"""5x same-seed repeats of the best-combo cross-resolution comparison (800 epochs).

Repeats the final resolution study (run_best_combo_resolutions.py) 5 times under
IDENTICAL configuration -- same tuned HPs, same seed 3035 every repeat -- so the
thesis table can report mean +- std per (net, resolution) cell. Fixing the seed
does not make the runs reproducible (cuDNN kernels are nondeterministic and the
DataLoader workers are unseeded), so the across-repeat spread measures framework
noise, which is exactly the statistic the supervisor asked for.

Differences from the original sweep:
  - all 6 resolutions in one protocol (386x260 is now swept too, instead of being
    folded in from the tuning-era sweeps),
  - 800 epochs instead of 1500 (every best-val epoch in the original sweep was
    <= 424; the per-epoch step LR decay is unaffected by MAX_EPOCH),
  - repeats are distinguished by a rep index in EXP_NAME, not by seed,
  - per-run code copies are DISABLED (copy_cur_env would put the 15 GB .venv in
    every run dir; provenance is the git commit instead),
  - TB images are suppressed (vis_results wrote one per epoch -> GBs of tfevents),
  - after each cell, checkpoints are pruned to the single best-val-MAE epoch.

  CSRNet      AdamW lr=4e-5 bs1  lp_base=100   (tuned at 386x260)
  MobileCount AdamW lr=1e-3 bs6  lp_base=2550  (tuned at 386x260)
  LOG_PARA anchored at 772x519 (auto-scales per resolution in loading_data()).

60 runs (2 nets x 6 resolutions x 5 reps), ordered rep-major (a complete pass of
all 12 cells per rep, smallest->largest resolution within a pass) so an
interrupted sweep still yields whole passes -> usable n at any stopping point.

Env overrides:
  REPS      comma list (default "1,2,3,4,5")
  RES       comma list of variants (default all six)
  ONLY_NET  restrict to one net ("CSRNet" or "MobileCount")
  MAX_EPOCH epoch budget (default 800; smoke test: MAX_EPOCH=2)
  SWEEP_DIR output dir (default ./exp/bestcombo_resolutions_repeats_800ep);
            completed (variant, net, rep) cells in the results JSON are skipped,
            half-finished cells resume from latest_state.pth -- relaunch-safe.

Writes bestcombo_repeats_results.json under $SWEEP_DIR incrementally.
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
import misc.utils

# copy_cur_env copies the whole work dir (incl. the 15 GB .venv) into every run
# dir; 60 runs would be ~900 GB. The sweep is committed to git instead. The
# patch survives the per-cell module reset because misc.utils is never evicted.
misc.utils.copy_cur_env = lambda *a, **k: None

config.cfg.DATASET = "Tenebrio"
config.cfg.GPU_ID = [0]
config.cfg.MAX_EPOCH = int(os.environ.get("MAX_EPOCH", "800"))
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

config.cfg.EXP_PATH = os.environ.get(
    "SWEEP_DIR", "./exp/bestcombo_resolutions_repeats_800ep")

SEED = 3035              # same seed for every repeat -- the spread IS the point
ANCHOR = 772 * 519       # LOG_PARA_BASE_AREA for both nets (re-anchored scheme)

# each net's best cell from the new-LOG_PARA grids (tuned at 386x260)
BEST = {
    "CSRNet":      {"optimizer": "AdamW", "lr": 4e-5, "bs": 1, "lp_base": 100.0},
    "MobileCount": {"optimizer": "AdamW", "lr": 1e-3, "bs": 6, "lp_base": 2550.0},
}

# (w, h) per variant, for the LOG_PARA anchoring assertion
RES_DIMS = {
    "49x33": (49, 33), "97x65": (97, 65), "193x130": (193, 130),
    "386x260": (386, 260), "772x519": (772, 519), "1544x1038": (1544, 1038),
}
ALL_RES = list(RES_DIMS)  # smallest -> largest

REPS = [int(r) for r in os.environ.get("REPS", "1,2,3,4,5").split(",")]
_env_res = [v.strip() for v in os.environ.get("RES", ",".join(ALL_RES)).split(",")]
assert all(v in RES_DIMS for v in _env_res), f"unknown variant in RES={_env_res}"
RUN_RES = [v for v in ALL_RES if v in set(_env_res)]  # canonical order
ONLY_NET = os.environ.get("ONLY_NET", "")
RUN_NETS = [n for n in ("CSRNet", "MobileCount") if not ONLY_NET or n == ONLY_NET]
assert RUN_NETS, f"ONLY_NET={ONLY_NET!r} matches no net"

RESULTS_NAME = "bestcombo_repeats_results.json"

# rep-major: every completed pass is a full grid, so n grows one pass at a time
RUNS = [
    {"net": net, "variant": v, "rep": rep, **BEST[net]}
    for rep in REPS
    for v in RUN_RES
    for net in RUN_NETS
]


class RecordingWriter:
    def __init__(self, real_writer):
        self._real = real_writer
        self.records: list[dict] = []

    def add_scalar(self, tag, value, step):
        self._real.add_scalar(tag, value, step)
        # flush immediately: tensorboardX buffers ~120 s, so a crash would lose
        # the tail of the scalar history that crash-recovery reads back from
        # tfevents (at ~2 s/epoch on the small variants that is dozens of epochs)
        self._real.flush()
        try:
            self.records.append({"tag": tag, "value": float(value), "step": int(step)})
        except Exception:
            self.records.append({"tag": tag, "value": float(value), "step": str(step)})

    def add_image(self, *args, **kwargs):
        # vis_results logs a val image every epoch (5+ GB of tfevents at 1544x1038);
        # crash recovery only needs scalars, so drop images entirely
        pass

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


def prune_checkpoints(run_dir: Path, metrics: list[dict]) -> dict:
    """Keep only the best-val-MAE epoch's checkpoint; drop the rest + latest_state.

    update_model saves on MAE *or* MSE improvement, so best_model_name can point
    at an MSE-only epoch -- the keeper is recomputed from the recorded val mae
    (TB step == the 1-based epoch in the filename). The epoch-prefix glob is
    unambiguous because "_mae_" delimits the epoch number.
    """
    mae_pts = [(int(m["step"]), float(m["value"])) for m in metrics if m["tag"] == "mae"]
    if not mae_pts:
        print(f"  WARNING: no mae metrics for {run_dir}; skip prune")
        return {"pruned": False}
    epoch, _ = min(mae_pts, key=lambda t: t[1])
    keep = sorted(run_dir.glob(f"all_ep_{epoch}_mae_*.pth"))
    if not keep:
        print(f"  WARNING: keeper all_ep_{epoch}_mae_*.pth not found in {run_dir}; skip prune")
        return {"pruned": False}
    removed = 0
    for p in run_dir.glob("all_ep_*.pth"):
        if p != keep[0]:
            p.unlink()
            removed += 1
    (run_dir / "latest_state.pth").unlink(missing_ok=True)
    return {"pruned": True, "kept_checkpoint": keep[0].name, "pruned_count": removed}


def train_one(run: dict) -> dict:
    from datasets.Tenebrio.setting import cfg_data
    cfg_data.DATA_PATH = f"./exp/data/Tenebrio/{run['variant']}"
    cfg_data.TRAIN_BATCH_SIZE = run["bs"]
    cfg_data.VAL_BATCH_SIZE = 1
    cfg_data.LOG_PARA_BASE = run["lp_base"]   # 100 (CSRNet) / 2550 (MobileCount)
    cfg_data.LOG_PARA_BASE_AREA = ANCHOR      # 772*519, read by _compute_log_para()
    cfg_data.AUG = "none"                     # RandomCrop only, as in the original sweep

    rep = run["rep"]
    config.cfg.NET = run["net"]
    config.cfg.LR = run["lr"]
    config.cfg.OPTIMIZER = run["optimizer"]
    config.cfg.SEED = SEED
    config.cfg.EXP_NAME = (f"{run['variant']}_{run['net']}_{run['optimizer']}"
                           f"_lr{run['lr']:.1e}_seed{SEED}_rep{rep}_bs{run['bs']}")

    # resume a half-finished cell from its latest_state.pth; RESUME/RESUME_PATH
    # are sticky module state, so both must be reset explicitly for fresh runs
    run_dir = Path(config.cfg.EXP_PATH) / config.cfg.EXP_NAME
    latest_state = run_dir / "latest_state.pth"
    resumed = latest_state.exists()
    old_records: list[dict] = []
    resumed_from_epoch = None
    if resumed:
        old_records = recover_tfevents_records(run_dir)
        resumed_from_epoch = max(
            (int(r["step"]) for r in old_records if r["tag"] == "mae"), default=0)
        config.cfg.RESUME = True
        config.cfg.RESUME_PATH = str(latest_state)
        print(f"  resuming from {latest_state} after epoch {resumed_from_epoch} "
              f"({len(old_records)} pre-existing metric records recovered)")
    else:
        config.cfg.RESUME = False
        config.cfg.RESUME_PATH = ""

    seed_everything(SEED)

    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))

    w, h = RES_DIMS[run["variant"]]
    expected_lp = run["lp_base"] * (w * h) / ANCHOR
    assert abs(cfg_data.LOG_PARA - expected_lp) < 0.01, (
        f"LOG_PARA anchoring wrong: got {cfg_data.LOG_PARA:.4f}, "
        f"expected {expected_lp:.4f} for {run['net']}/{run['variant']}")

    print(f"  [{run['net']}/{run['variant']}/rep{rep}] "
          f"optimizer = {type(trainer.optimizer).__name__}  lr={config.cfg.LR}  "
          f"batch_size={cfg_data.TRAIN_BATCH_SIZE}  log_para={cfg_data.LOG_PARA:.2f}  "
          f"aug={cfg_data.AUG}  weight_decay={config.cfg.WEIGHT_DECAY}  seed={SEED}  "
          f"train_iters/epoch={len(trainer.train_loader)}")

    rec_writer = RecordingWriter(trainer.writer)
    trainer.writer = rec_writer

    t0 = time.time()
    trainer.forward()
    duration = time.time() - t0

    result = {
        "net": run["net"],
        "optimizer": run["optimizer"],
        "seed": SEED,
        "rep": rep,
        "variant": run["variant"],
        "aug": "none",
        "batch_size": run["bs"],
        "lr": run["lr"],
        "log_para_base": run["lp_base"],
        "log_para_base_area": ANCHOR,
        "log_para": float(cfg_data.LOG_PARA),
        "weight_decay": config.cfg.WEIGHT_DECAY,
        "max_epoch": config.cfg.MAX_EPOCH,
        "exp_name": config.cfg.EXP_NAME,
        "resumed": resumed,
        "resumed_from_epoch": resumed_from_epoch,
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
    expected = {"train": 840, "val": 168, "test": 112}
    for variant in RUN_RES:
        for split, n_expected in expected.items():
            img_dir = ROOT / f"exp/data/Tenebrio/{variant}/{split}/img"
            n = len(list(img_dir.glob("*.png"))) if img_dir.is_dir() else 0
            if n != n_expected:
                sys.exit(f"preflight: {img_dir} has {n} PNGs, expected {n_expected}")


def write_results(results_path: Path, results: list[dict]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)


def main() -> None:
    preflight()

    out_dir = Path(config.cfg.EXP_PATH)
    results_path = out_dir / RESULTS_NAME
    results: list[dict] = []
    if results_path.exists():
        results = json.load(results_path.open())
    done = {(r["variant"], r["net"], r["rep"]) for r in results if "error" not in r}

    # idempotent prune pass: cover cells that finished but died before pruning
    pruned_any = False
    for r in results:
        if "error" in r or r.get("pruned"):
            continue
        info = prune_checkpoints(out_dir / r["exp_name"], r["metrics"])
        r.update(info)
        pruned_any = True
        print(f"startup prune: {r['exp_name']} -> {info}")
    if pruned_any:
        write_results(results_path, results)

    print(f"=== Best-combo resolution repeats: {len(RUNS)} runs x "
          f"{config.cfg.MAX_EPOCH} epochs, seed {SEED} every rep ===")
    print(f"=== nets={RUN_NETS}  res={RUN_RES}  reps={REPS} ===")
    print(f"=== Output under {config.cfg.EXP_PATH} ===")
    if done:
        print(f"=== Already completed (skipped): {len(done)} cells ===")
    print()

    for run in RUNS:
        label = f"{run['net']}/{run['variant']}/rep{run['rep']}"
        if (run["variant"], run["net"], run["rep"]) in done:
            print(f"SKIP (done): {label}")
            continue
        print(f"\n{'=' * 60}\nRUN: {label}\n{'=' * 60}")
        try:
            result = train_one(run)
        except Exception as e:
            print(f"  [{label}] FAILED: {type(e).__name__}: {e}")
            results.append({"net": run["net"], "variant": run["variant"],
                            "rep": run["rep"], "lr": run["lr"],
                            "error": f"{type(e).__name__}: {e}"})
            write_results(results_path, results)
            torch.cuda.empty_cache()
            gc.collect()
            continue
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        # flush the row before pruning so a crash mid-prune can't lose the cell;
        # the startup prune pass finishes the job on relaunch
        results.append(result)
        write_results(results_path, results)
        result.update(prune_checkpoints(out_dir / result["exp_name"], result["metrics"]))
        write_results(results_path, results)

    print(f"\n{'=' * 60}\nSUMMARY (val best_mae)\n{'=' * 60}")
    print(f'{"net":<13} {"variant":<11} {"rep":<4} {"log_para":<10} '
          f'{"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r["net"]:<13} {r["variant"]:<11} {r["rep"]:<4} ERROR: {r["error"]}')
        else:
            print(f'{r["net"]:<13} {r["variant"]:<11} {r["rep"]:<4} '
                  f'{r["log_para"]:<10.2f} {r["best_mae"]:<12.4f} '
                  f'{r["best_mse"]:<12.4f} {r["duration_sec"]:>6.0f}s')


if __name__ == "__main__":
    main()
