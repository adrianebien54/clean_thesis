#!/usr/bin/env python3
"""Same as sweep_logpara_resolutions.py but with per-variant LOG_PARA chosen
to make the post-LOG_PARA target peak comparable across variants.

Hypothesis from the fixed-LOG_PARA sweep: the U-shape in MAE-vs-resolution
(stalls at 1544x1038 + 97x65, sweet spot at 386x260 / 772x519) is partly
caused by target dynamic range varying ~155x across variants while
LOG_PARA is held constant. This sweep tests that by scaling LOG_PARA so
that LOG_PARA * raw_density is roughly constant across variants.

Derivation:
  raw_density_max(v) ∝ 1 / area(v)   (count preserved, so density per
                                       pixel shrinks with more pixels)
  To keep LOG_PARA(v) * raw_density_max(v) ≈ const, set
     LOG_PARA(v) = LOG_PARA_base * area(v) / area_base

With area_base = 386 * 260 and LOG_PARA_base = 100:
   1544x1038 -> 1597
   772x519   ->  399
   386x260   ->  100
   193x130   ->   25
   97x65     ->    6
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
config.cfg.MAX_EPOCH = 30
config.cfg.VAL_FREQ = 1
config.cfg.VAL_DENSE_START = -1
config.cfg.PRINT_FREQ = 50
config.cfg.LR = 1e-5
config.cfg.PRE_GCC = False
config.cfg.RESUME = False

SWEEP_STAMP = time.strftime("%m-%d_%H-%M", time.localtime())
config.cfg.EXP_PATH = f"./exp/sweep_logpara_var_{SWEEP_STAMP}"

# Per-variant (resolution, suggested LOG_PARA) such that LOG_PARA * raw_max ≈ 0.78
VARIANTS: list[tuple[str, float]] = [
    ("1544x1038", 1597.0),
    ("772x519",    399.0),
    ("386x260",    100.0),
    ("193x130",     25.0),
    ("97x65",        6.0),
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


def train_one_variant(variant: str, log_para: float, batch_size: int) -> dict:
    from datasets.Tenebrio.setting import cfg_data
    cfg_data.DATA_PATH = f"./exp/data/Tenebrio/{variant}"
    cfg_data.TRAIN_BATCH_SIZE = batch_size
    cfg_data.VAL_BATCH_SIZE = 1
    cfg_data.LOG_PARA = log_para

    config.cfg.EXP_NAME = f"{variant}_lp{log_para:g}_bs{batch_size}"

    for mod_name in list(sys.modules):
        if mod_name in ("trainer", "datasets.Tenebrio.loading_data"):
            del sys.modules[mod_name]

    from trainer import Trainer
    from datasets.Tenebrio.loading_data import loading_data

    trainer = Trainer(loading_data, cfg_data, str(ROOT))

    pre_range = measure_output_range(trainer.net, trainer.val_loader)
    print(f"  [{variant}] LOG_PARA={log_para:g}  pre-training:")
    print(f"    pred [{pre_range['pred_min']:.4f}, {pre_range['pred_max']:.4f}]  "
          f"gt [{pre_range['gt_min']:.4f}, {pre_range['gt_max']:.4f}]  "
          f"gt_mean={pre_range['gt_mean']:.6f}")

    rec_writer = RecordingWriter(trainer.writer)
    trainer.writer = rec_writer

    t0 = time.time()
    trainer.forward()
    duration = time.time() - t0

    post_range = measure_output_range(trainer.net, trainer.val_loader)
    print(f"  [{variant}] took {duration:.0f}s.  post-training:")
    print(f"    pred [{post_range['pred_min']:.4f}, {post_range['pred_max']:.4f}]  "
          f"ratio pred_max/gt_max = {post_range['pred_max']/max(post_range['gt_max'],1e-9):.3f}")

    result = {
        "variant": variant,
        "log_para": log_para,
        "batch_size": batch_size,
        "duration_sec": duration,
        "pre_range": pre_range,
        "post_range": post_range,
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
    print(f"=== Variable-LOG_PARA sweep — MobileCount × {len(VARIANTS)} variants × {config.cfg.MAX_EPOCH} epochs ===")
    print(f"=== Output under {config.cfg.EXP_PATH} ===\n")

    results: list[dict] = []
    sweep_t0 = time.time()
    for variant, log_para in VARIANTS:
        print(f"\n{'=' * 60}\nVARIANT: {variant}  (LOG_PARA={log_para:g})\n{'=' * 60}")

        batch = 6
        try:
            results.append(train_one_variant(variant, log_para, batch))
        except torch.cuda.OutOfMemoryError:
            print(f"  [{variant}] OOM at batch=6, retrying with batch=2")
            torch.cuda.empty_cache()
            gc.collect()
            try:
                results.append(train_one_variant(variant, log_para, 2))
            except torch.cuda.OutOfMemoryError:
                print(f"  [{variant}] OOM at batch=2 too — skipping")
                results.append({"variant": variant, "error": "OOM at batch=2"})
        except Exception as e:
            print(f"  [{variant}] FAILED: {type(e).__name__}: {e}")
            results.append({"variant": variant, "error": f"{type(e).__name__}: {e}"})
        finally:
            torch.cuda.empty_cache()
            gc.collect()

        out_dir = Path(config.cfg.EXP_PATH)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "sweep_results.json").open("w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n=== Sweep total time: {(time.time() - sweep_t0) / 60:.1f} min ===")
    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f'{"variant":<12} {"log_para":<10} {"batch":<6} {"best_mae":<12} {"best_mse":<12} {"duration":<10}')
    for r in results:
        if "error" in r:
            print(f'{r["variant"]:<12} ERROR: {r["error"]}')
        else:
            print(f'{r["variant"]:<12} {r["log_para"]:<10.4g} {r["batch_size"]:<6} '
                  f'{r["best_mae"]:<12.4f} {r["best_mse"]:<12.4f} '
                  f'{r["duration_sec"]:>6.0f}s')


if __name__ == "__main__":
    main()
