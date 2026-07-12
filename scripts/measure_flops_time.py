#!/usr/bin/env python3
"""Measure FLOPs and batch-1 inference time for every net x resolution.

For each (net, resolution) row in exp/combined_resolution/testset_results.json:

  * gmac     -- multiply-accumulate count of one forward pass at that input size,
                via the vendored models/ptflops counter (as in test_flops.py). The
                model is run on CPU because ptflops builds a CPU input tensor.
                FLOPs are weight-independent, so no checkpoint is needed here.
                Note ptflops only counts hooked module types (conv/linear/bn/act/
                pool); functional ops such as F.interpolate are not counted.
  * ms_mean  -- wall time of one batch-1 test_forward on the GPU, mean over
                N_ITERS synchronized runs after N_WARMUP warm-up runs with
                cudnn.benchmark autotune (as in test_time.py). The best-val-MAE
                checkpoint behind testset_table.md is loaded so the numbers stay
                tied to the reported test MAEs (values don't affect timing).

Fresh models are constructed per measurement so cells stay independent.
Writes exp/combined_resolution/flops_time.json and flops_time_table.md.
Run with the project venv: .venv/bin/python scripts/measure_flops_time.py
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

ROOT = Path("/home/umrobotics/clean_thesis/crowd_counting")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import config
config.cfg.GPU_ID = [0]

from models.CC import CrowdCounter
from models.ptflops import get_model_complexity_info

RESULTS_JSON = ROOT / "exp/combined_resolution/testset_results.json"
OUT_JSON = ROOT / "exp/combined_resolution/flops_time.json"
OUT_TABLE = ROOT / "exp/combined_resolution/flops_time_table.md"

N_WARMUP = 10
N_ITERS = 100


def measure_flops(net_name: str, width: int, height: int) -> dict:
    net = CrowdCounter(config.cfg.GPU_ID, net_name).CCN.cpu()
    flops, params = get_model_complexity_info(
        net, (height, width), as_strings=False, print_per_layer_stat=False
    )
    del net
    gc.collect()
    torch.cuda.empty_cache()
    return {"gmac": flops / 1e9, "params_m": params / 1e6}


def measure_time(net_name: str, ckpt: Path, width: int, height: int) -> dict:
    net = CrowdCounter(config.cfg.GPU_ID, net_name)
    state = torch.load(str(ckpt), map_location="cuda", weights_only=True)
    net.load_state_dict(state)
    net.eval()

    x = torch.randn(1, 3, height, width, device="cuda")
    times = []
    with torch.no_grad():
        for _ in range(N_WARMUP):
            net.test_forward(x)
        torch.cuda.synchronize()
        for _ in range(N_ITERS):
            t0 = time.perf_counter()
            net.test_forward(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)

    del net, state, x
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "ms_mean": statistics.mean(times),
        "ms_std": statistics.stdev(times),
        "fps": 1e3 / statistics.mean(times),
    }


def main() -> None:
    cudnn.benchmark = True
    rows = json.load(RESULTS_JSON.open())
    results = []
    print(f"{'net':<12} {'variant':<11} {'GMac':>10} {'ms/img':>8} {'fps':>8}")
    print("-" * 52)
    for r in rows:
        w, h = (int(v) for v in r["variant"].split("x"))
        f = measure_flops(r["net"], w, h)
        t = measure_time(r["net"], ROOT / r["checkpoint"], w, h)
        results.append({
            "variant": r["variant"], "net": r["net"], "pixels": r["pixels"],
            "test_mae": r["test_mae"], "test_mse": r["test_mse"],
            **f, **t,
            "n_warmup": N_WARMUP, "n_iters": N_ITERS,
            "checkpoint": r["checkpoint"],
        })
        print(f"{r['net']:<12} {r['variant']:<11} {f['gmac']:>10.3f} "
              f"{t['ms_mean']:>8.2f} {t['fps']:>8.1f}")

    with OUT_JSON.open("w") as f:
        json.dump(results, f, indent=2)
    write_table(results)
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_TABLE}")


def write_table(results: list[dict]) -> None:
    by = {(r["variant"], r["net"]): r for r in results}
    variants = list(dict.fromkeys(r["variant"] for r in results))
    lines = [
        "# FLOPs and batch-1 inference time (Tenebrio best-val checkpoints)",
        "",
        f"GMac via vendored ptflops (CPU forward, hooked modules only); time is the",
        f"mean of {N_ITERS} synchronized GPU test_forward runs after {N_WARMUP} warm-up",
        "runs with cudnn.benchmark, on the checkpoints behind testset_table.md.",
        "",
        "| Resolution | CSRNet GMac | CSRNet ms/img | MobileCount GMac | MobileCount ms/img |",
        "|---|---:|---:|---:|---:|",
    ]
    for v in variants:
        c, m = by[(v, "CSRNet")], by[(v, "MobileCount")]
        lines.append(
            f"| {v} | {c['gmac']:.3f} | {c['ms_mean']:.2f} ± {c['ms_std']:.2f} "
            f"| {m['gmac']:.3f} | {m['ms_mean']:.2f} ± {m['ms_std']:.2f} |")
    c, m = by[(variants[0], "CSRNet")], by[(variants[0], "MobileCount")]
    lines += ["",
              f"Params: CSRNet {c['params_m']:.2f} M, MobileCount {m['params_m']:.2f} M."]
    with OUT_TABLE.open("w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
