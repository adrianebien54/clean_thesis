#!/usr/bin/env python3
"""Measure peak GPU memory of single-image inference for every net x resolution.

For each (net, resolution) row in exp/combined_resolution/testset_results.json this
loads the exact best-val-MAE checkpoint used for the test-split evaluation and runs a
batch-1 forward pass (net.test_forward, no_grad) on an input of that resolution,
recording via the CUDA caching-allocator stats:

  * weights_mb  -- memory_allocated after model construction + checkpoint load
                   (parameters + buffers resident on the GPU),
  * peak_mb     -- max_memory_allocated during the forward pass, measured after a
                   warm-up run so cuDNN autotune workspaces don't inflate it.
                   Includes resident weights + input + activation high-water mark,
  * params_m    -- parameter count in millions.

Allocator state is reset (empty_cache + baseline subtraction) between cells so the
12 measurements are independent. Memory is content-independent, but loading the real
checkpoints keeps the numbers tied to the reported test MAEs.

Writes exp/combined_resolution/inference_memory.json and inference_memory_table.md.
Run with the project venv: .venv/bin/python scripts/measure_inference_memory.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path("/home/umrobotics/clean_thesis/crowd_counting")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import config
config.cfg.GPU_ID = [0]

from models.CC import CrowdCounter

RESULTS_JSON = ROOT / "exp/combined_resolution/testset_results.json"
OUT_JSON = ROOT / "exp/combined_resolution/inference_memory.json"
OUT_TABLE = ROOT / "exp/combined_resolution/inference_memory_table.md"

MB = 1024 ** 2


def measure(net_name: str, ckpt: Path, width: int, height: int) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()

    net = CrowdCounter(config.cfg.GPU_ID, net_name)
    state = torch.load(str(ckpt), map_location="cuda", weights_only=True)
    net.load_state_dict(state)
    net.eval()
    # Free the checkpoint dict before measuring: load_state_dict copies the values
    # into the module's own tensors, so keeping `state` alive would double-count
    # every weight in both weights_mb and peak_mb.
    del state
    gc.collect()
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated() - base
    params = sum(p.numel() for p in net.CCN.parameters())

    x = torch.randn(1, 3, height, width, device="cuda")
    with torch.no_grad():
        for _ in range(2):  # warm-up: cuDNN algo search allocates throwaway workspaces
            net.test_forward(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        net.test_forward(x)
        torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base

    del net, x
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "weights_mb": weights / MB,
        "peak_mb": peak / MB,
        "params_m": params / 1e6,
    }


def main() -> None:
    rows = json.load(RESULTS_JSON.open())
    results = []
    print(f"{'net':<12} {'variant':<11} {'params_M':>9} {'weights_MB':>11} {'peak_MB':>9}")
    print("-" * 56)
    for r in rows:
        w, h = (int(v) for v in r["variant"].split("x"))
        m = measure(r["net"], ROOT / r["checkpoint"], w, h)
        results.append({
            "variant": r["variant"], "net": r["net"], "pixels": r["pixels"],
            "test_mae": r["test_mae"], "test_mse": r["test_mse"],
            **m,
            "checkpoint": r["checkpoint"],
        })
        print(f"{r['net']:<12} {r['variant']:<11} {m['params_m']:>9.2f} "
              f"{m['weights_mb']:>11.1f} {m['peak_mb']:>9.1f}")

    with OUT_JSON.open("w") as f:
        json.dump(results, f, indent=2)
    write_table(results)
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_TABLE}")


def write_table(results: list[dict]) -> None:
    by = {(r["variant"], r["net"]): r for r in results}
    variants = list(dict.fromkeys(r["variant"] for r in results))
    lines = [
        "# Peak GPU memory of batch-1 inference (Tenebrio best-val checkpoints)",
        "",
        "max_memory_allocated during one test_forward after warm-up (weights + input +",
        "activation peak), measured on the checkpoints behind testset_table.md.",
        "",
        "| Resolution | CSRNet peak MB | CSRNet test MAE | MobileCount peak MB | MobileCount test MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for v in variants:
        c, m = by[(v, "CSRNet")], by[(v, "MobileCount")]
        lines.append(f"| {v} | {c['peak_mb']:.1f} | {c['test_mae']:.4f} "
                     f"| {m['peak_mb']:.1f} | {m['test_mae']:.4f} |")
    c, m = by[(variants[0], "CSRNet")], by[(variants[0], "MobileCount")]
    lines += ["",
              f"Model weights: CSRNet {c['params_m']:.2f} M params ({c['weights_mb']:.1f} MB), "
              f"MobileCount {m['params_m']:.2f} M params ({m['weights_mb']:.1f} MB)."]
    with OUT_TABLE.open("w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
