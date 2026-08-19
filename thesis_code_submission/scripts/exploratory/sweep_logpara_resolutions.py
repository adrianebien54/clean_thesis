#!/usr/bin/env python3
"""Train MobileCount on each Tenebrio resolution variant to probe whether
LOG_PARA=100 produces problematic dynamics across the resolution range.

Runs in-process; one variant at a time; 30 epochs each. Catches CUDA OOM
on the 1544x1038 variant and retries at batch=2. Writes a single
sweep_results.json under exp/sweep_logpara_<timestamp>/ containing
per-epoch metrics for plotting.
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

# ---- sweep-wide cfg overrides ----
config.cfg.NET = "MobileCount"
config.cfg.DATASET = "Tenebrio"
config.cfg.GPU_ID = [0]
config.cfg.MAX_EPOCH = 30
config.cfg.VAL_FREQ = 1
config.cfg.VAL_DENSE_START = -1  # validate every epoch
config.cfg.PRINT_FREQ = 50
config.cfg.LR = 1e-5
config.cfg.PRE_GCC = False
config.cfg.RESUME = False

SWEEP_STAMP = time.strftime("%m-%d_%H-%M", time.localtime())
config.cfg.EXP_PATH = f"./exp/sweep_logpara_{SWEEP_STAMP}"

VARIANTS = ["1544x1038", "772x519", "386x260", "193x130", "97x65"]


class RecordingWriter:
    """Wraps the TB SummaryWriter and tees add_scalar calls into a list."""

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


def measure_output_range(net, loader) -> dict:
    net.CCN.eval()
    img, gt = next(iter(loader))
    img = img.cuda()
    gt = gt.cuda()
    with torch.no_grad():
        pred = net.CCN(img)
    return {
        "pred_min": float(pred.min()), "pred_max": float(pred.max()),
        "pred_mean": float(pred.mean()), "pred_std": float(pred.std()),
        "gt_min": float(gt.min()), "gt_max": float(gt.max()),
        "gt_mean": float(gt.mean()), "gt_std": float(gt.std()),
        "pred_shape": list(pred.shape), "gt_shape": list(gt.shape),
    }


def train_one_variant(variant: str, batch_size: int) -> dict:
    from datasets.Tenebrio.setting import cfg_data
    cfg_data.DATA_PATH = f"./exp/data/Tenebrio/{variant}"
    cfg_data.TRAIN_BATCH_SIZE = batch_size
    cfg_data.VAL_BATCH_SIZE = 1

    config.cfg.EXP_NAME = f"{variant}_bs{batch_size}"

    # Reload the trainer module in case any module-level state was set
    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))

    print(f"  [{variant}] pre-training output range probe:")
    pre_range = measure_output_range(trainer.net, trainer.val_loader)
    print(f"    pred range=[{pre_range['pred_min']:.4f}, {pre_range['pred_max']:.4f}] "
          f"mean={pre_range['pred_mean']:.4f}  std={pre_range['pred_std']:.4f}")
    print(f"    gt   range=[{pre_range['gt_min']:.4f}, {pre_range['gt_max']:.4f}] "
          f"mean={pre_range['gt_mean']:.4f}  std={pre_range['gt_std']:.4f}")

    rec_writer = RecordingWriter(trainer.writer)
    trainer.writer = rec_writer

    t0 = time.time()
    trainer.forward()
    duration = time.time() - t0
    print(f"  [{variant}] training took {duration:.0f}s ({duration / 60:.1f} min)")

    print(f"  [{variant}] post-training output range probe:")
    post_range = measure_output_range(trainer.net, trainer.val_loader)
    print(f"    pred range=[{post_range['pred_min']:.4f}, {post_range['pred_max']:.4f}] "
          f"mean={post_range['pred_mean']:.4f}  std={post_range['pred_std']:.4f}")

    result = {
        "variant": variant,
        "batch_size": batch_size,
        "duration_sec": duration,
        "pre_range": pre_range,
        "post_range": post_range,
        "metrics": rec_writer.records,
        "best_mae": float(trainer.train_record["best_mae"]),
        "best_mse": float(trainer.train_record["best_mse"]),
        "best_model_name": trainer.train_record["best_model_name"],
    }

    # cleanup
    try:
        trainer.writer._real.close()
    except Exception:
        pass
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    print(f"=== LOG_PARA sweep — MobileCount × {len(VARIANTS)} variants × {config.cfg.MAX_EPOCH} epochs ===")
    print(f"=== Output under {config.cfg.EXP_PATH} ===\n")

    results: list[dict] = []
    sweep_t0 = time.time()
    for variant in VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"VARIANT: {variant}")
        print(f"{'=' * 60}")

        batch = 6
        try:
            results.append(train_one_variant(variant, batch))
        except torch.cuda.OutOfMemoryError:
            print(f"  [{variant}] OOM at batch=6, retrying with batch=2")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                results.append(train_one_variant(variant, 2))
            except torch.cuda.OutOfMemoryError:
                print(f"  [{variant}] OOM at batch=2 too — skipping")
                results.append({"variant": variant, "error": "OOM at batch=2"})
        except Exception as e:
            print(f"  [{variant}] FAILED: {type(e).__name__}: {e}")
            results.append({"variant": variant, "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        # incremental save so we have data if a later variant blows up
        out_dir = Path(config.cfg.EXP_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "sweep_results.json").open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n=== Sweep total time: {(time.time() - sweep_t0) / 60:.1f} min ===")
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f'{"variant":<12} {"batch":<6} {"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r["variant"]:<12} ERROR: {r["error"]}')
        else:
            print(f'{r["variant"]:<12} {r["batch_size"]:<6} '
                  f'{r["best_mae"]:<12.4f} {r["best_mse"]:<12.4f} '
                  f'{r["duration_sec"]:>6.0f}s')


if __name__ == "__main__":
    main()
