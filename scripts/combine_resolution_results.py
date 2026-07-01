#!/usr/bin/env python3
"""Combine the bestcombo resolution sweep with the 386x260 (360p) AUG=none point.

The bestcombo_resolutions sweep (scripts/run_best_combo_resolutions.py) trained each
net's best combo (CSRNet AdamW lr4e-5 bs1, MobileCount AdamW lr1e-3 bs6, AUG=none,
1500 ep, seed 3035) at 5 resolutions but deliberately EXCLUDED 386x260 (the tuning
resolution). This folds the comparable 386x260 AUG=none best-combo point back in --
read from its original sweeps -- so the resolution curve spans all 6 resolutions.

Sources (all AUG=none, same best combo, LOG_PARA anchored at 772x519):
  49x33..1544x1038  -> exp/bestcombo_resolutions_06-24_11-49/bestcombo_resolutions_results.json
  386x260 CSRNet    -> exp/batchsize_sweep_logpara_06-21_22-41/batchsize_logpara_results.json (bs1)
  386x260 MobileCount -> exp/aa_final/mobilecount_logpara2550_anchor772x519_06-17_14-26/
                         mobilecount_logpara2550_results.json (bs6 lr1e-3; best of the runs)

Writes next to exp/:
  exp/combined_resolution_table.md
  exp/combined_resolution_mae.png
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/umrobotics/clean_thesis/crowd_counting"
os.chdir(ROOT)

# (label, width, height) smallest -> largest
RESOLUTIONS = [
    ("49x33", 49, 33),
    ("97x65", 97, 65),
    ("193x130", 193, 130),
    ("386x260", 386, 260),
    ("772x519", 772, 519),
    ("1544x1038", 1544, 1038),
]
NETS = ["CSRNet", "MobileCount"]

BESTCOMBO = "exp/bestcombo_resolutions_06-24_11-49/bestcombo_resolutions_results.json"
CSRNET_386 = "exp/batchsize_sweep_logpara_06-21_22-41/batchsize_logpara_results.json"
MOBILE_386 = ("exp/aa_final/mobilecount_logpara2550_anchor772x519_06-17_14-26/"
              "mobilecount_logpara2550_results.json")


def _rows(path):
    d = json.load(open(path))
    rows = d if isinstance(d, list) else d.get("runs", [])
    return [r for r in rows if isinstance(r, dict) and "error" not in r]


def best_of(rows):
    """Pick the row with the lowest best_mae (ties -> lowest best_mse)."""
    return min(rows, key=lambda r: (float(r["best_mae"]), float(r["best_mse"])))


def collect() -> dict:
    """data[variant][net] = {'mae':.., 'mse':.., 'src':..}"""
    data = {v: {} for v, _, _ in RESOLUTIONS}

    # 5 resolutions x 2 nets from the bestcombo sweep
    for r in _rows(BESTCOMBO):
        v, n = str(r.get("variant")), r.get("net")
        if v in data and n in NETS:
            data[v][n] = {"mae": float(r["best_mae"]), "mse": float(r["best_mse"]),
                          "src": "bestcombo_resolutions"}

    # 386x260 CSRNet (bs1 lr4e-5, AUG=none)
    cs = [r for r in _rows(CSRNET_386)
          if r.get("net") == "CSRNet" and str(r.get("variant")) == "386x260"
          and int(r.get("batch_size", 0)) == 1]
    b = best_of(cs)
    data["386x260"]["CSRNet"] = {"mae": float(b["best_mae"]), "mse": float(b["best_mse"]),
                                 "src": "batchsize_logpara"}

    # 386x260 MobileCount (bs6 lr1e-3 lp638.73, AUG=none; best of runs)
    ms = [r for r in _rows(MOBILE_386)
          if r.get("net") == "MobileCount" and str(r.get("variant")) == "386x260"
          and int(r.get("batch_size", 0)) == 6 and abs(float(r.get("lr", 0)) - 1e-3) < 1e-9]
    b = best_of(ms)
    data["386x260"]["MobileCount"] = {"mae": float(b["best_mae"]), "mse": float(b["best_mse"]),
                                      "src": "aa_final/mobilecount_logpara2550"}
    return data


def write_table(data: dict, path: str) -> None:
    lines = [
        "# Best-combo val MAE across input resolutions (Tenebrio, seed 3035, AUG=none, 1500 ep)",
        "",
        "CSRNet AdamW lr4e-5 bs1 | MobileCount AdamW lr1e-3 bs6 | LOG_PARA anchored at 772x519.",
        "386x260 (the tuning resolution) is folded in from its original sweeps; all others",
        "from bestcombo_resolutions_06-24_11-49.",
        "",
        "| Resolution | Pixels | CSRNet MAE | CSRNet MSE | MobileCount MAE | MobileCount MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v, w, h in RESOLUTIONS:
        c, m = data[v]["CSRNet"], data[v]["MobileCount"]
        star = " *" if v == "386x260" else ""
        lines.append(f"| {v}{star} | {w*h:,} | {c['mae']:.4f} | {c['mse']:.4f} "
                     f"| {m['mae']:.4f} | {m['mse']:.4f} |")
    lines += [
        "",
        "\\* 386x260 point: CSRNet from batchsize_logpara sweep, MobileCount from "
        "aa_final/mobilecount_logpara2550 (same best combo & AUG=none).",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot(data: dict, path: str) -> None:
    areas = [w * h for _, w, h in RESOLUTIONS]
    labels = [v for v, _, _ in RESOLUTIONS]
    fig, ax = plt.subplots(figsize=(8, 5))
    for net, color, marker in [("CSRNet", "tab:blue", "o"),
                               ("MobileCount", "tab:orange", "s")]:
        y = [data[v][net]["mae"] for v, _, _ in RESOLUTIONS]
        ax.plot(areas, y, marker=marker, color=color, label=net, linewidth=1.8, markersize=7)
        for a, yy in zip(areas, y):
            ax.annotate(f"{yy:.3f}", (a, yy), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(areas)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.get_xaxis().set_minor_locator(plt.NullLocator())
    ax.set_xlabel("input resolution (pixels, log scale)")
    ax.set_ylabel("best validation MAE (log scale)")
    ax.set_title("Best-combo val MAE vs input resolution\n(Tenebrio, seed 3035, AUG=none, 1500 ep)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


OUTDIR = "exp/combined_resolution"


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    data = collect()
    write_table(data, f"{OUTDIR}/resolution_table.md")
    print(f"wrote {OUTDIR}/resolution_table.md")
    # echo the table to stdout
    for v, w, h in RESOLUTIONS:
        c, m = data[v]["CSRNet"], data[v]["MobileCount"]
        print(f"  {v:<11} px={w*h:>9,}  CSRNet {c['mae']:.4f}  MobileCount {m['mae']:.4f}")
    plot(data, f"{OUTDIR}/mae_vs_resolution.png")


if __name__ == "__main__":
    main()
