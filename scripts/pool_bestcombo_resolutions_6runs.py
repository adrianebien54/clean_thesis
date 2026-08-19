#!/usr/bin/env python3
"""Pool the 5 same-seed 800-epoch repeats with the original 1500-epoch sweep (n=6).

The cross-resolution curve was first trained once per cell at a 1500-epoch budget
(exp/bestcombo_resolutions_06-24_11-49 + the two 386x260 host sweeps, folded in by
scripts/combine_resolution_results.py) and later repeated 5x per cell at 800 epochs
(exp/bestcombo_resolutions_repeats_800ep).  Config, seed (3035), AUG=none and the
772x519-anchored LOG_PARA are identical across all six runs of every cell, and every
cell's best-validation epoch in the 1500-epoch run falls at or below 445, so the
longer budget never produced the reported checkpoint: the first run is a legitimate
sixth sample of the same 800-epoch trajectory distribution.

Reads:
  exp/bestcombo_resolutions_repeats_800ep/testset_repeats_results.json   (5 reps/cell)
  exp/combined_resolution/testset_results.json                          (1500-ep run)

Writes:
  exp/bestcombo_resolutions_repeats_800ep/testset_6runs_table.md
  exp/bestcombo_resolutions_repeats_800ep/testset_6runs.json
  exp/bestcombo_resolutions_repeats_800ep/testset_6runs_table.tex
"""
from __future__ import annotations

import json
import os
import statistics as st

ROOT = "/home/umrobotics/clean_thesis/crowd_counting"
os.chdir(ROOT)

REPEATS = "exp/bestcombo_resolutions_repeats_800ep/testset_repeats_results.json"
ORIGINAL = "exp/combined_resolution/testset_results.json"
OUTDIR = "exp/bestcombo_resolutions_repeats_800ep"

RESOLUTIONS = [
    ("49x33", r"$49\times33$", 1617),
    ("97x65", r"$97\times65$", 6305),
    ("193x130", r"$193\times130$", 25090),
    ("386x260", r"$386\times260^*$", 100360),
    ("772x519", r"$772\times519$", 400668),
    ("1544x1038", r"$1544\times1038$", 1602672),
]
NETS = ["CSRNet", "MobileCount"]


def _mean_std(xs):
    return st.mean(xs), (st.stdev(xs) if len(xs) > 1 else float("nan"))


def collect():
    reps = json.load(open(REPEATS))
    orig = {(r["variant"], r["net"]): r for r in json.load(open(ORIGINAL))}
    out = {}
    for variant, _, pixels in RESOLUTIONS:
        for net in NETS:
            rows = sorted(
                (r for r in reps if r["variant"] == variant and r["net"] == net),
                key=lambda r: r["rep"],
            )
            g = orig[(variant, net)]
            # sanity: the sixth run must share the cell's LOG_PARA and stay inside
            # the 800-epoch window the repeats had available.
            assert abs(rows[0]["log_para"] - g["log_para"]) < 1e-9, (variant, net)
            assert g["best_epoch"] <= 800, (variant, net, g["best_epoch"])
            runs = [
                {"src": f"rep{r['rep']} (800 ep)", "epoch": r["best_epoch"],
                 "val_mae": r["val_mae"], "test_mae": r["test_mae"],
                 "test_mse": r["test_mse"]}
                for r in rows
            ] + [
                {"src": "run0 (1500 ep)", "epoch": g["best_epoch"],
                 "val_mae": g["val_mae"], "test_mae": g["test_mae"],
                 "test_mse": g["test_mse"]}
            ]
            cell = {"variant": variant, "net": net, "pixels": pixels,
                    "n": len(runs), "runs": runs,
                    "epoch_min": min(r["epoch"] for r in runs),
                    "epoch_max": max(r["epoch"] for r in runs)}
            for key in ("val_mae", "test_mae", "test_mse"):
                m, s = _mean_std([r[key] for r in runs])
                cell[f"{key}_mean"], cell[f"{key}_std"] = m, s
            cell["test_mae_cv"] = cell["test_mae_std"] / cell["test_mae_mean"]
            out[(variant, net)] = cell
    return out


def epoch_cell(c):
    lo, hi = c["epoch_min"], c["epoch_max"]
    return f"${lo}$" if lo == hi else f"${lo}$--${hi}$"


def sig(mean, std):
    """Round mean/std to the same decimals: 3 below 1, else 2."""
    d = 3 if mean < 1 else 2
    return f"{mean:.{d}f}", f"{std:.{d}f}"


def write_markdown(data):
    lines = [
        "# Cross-resolution: TEST-split error, mean +- std over 6 same-seed runs",
        "",
        "Five 800-epoch repeats (exp/bestcombo_resolutions_repeats_800ep) pooled with",
        "the original 1500-epoch run (exp/combined_resolution/testset_results.json).",
        "Identical config and seed 3035 in all six; spread = framework nondeterminism.",
        "Every 1500-epoch best-validation epoch is <= 445, inside the 800-epoch window,",
        "so the longer budget never produced the reported checkpoint.",
        "std is the sample std (ddof=1). Test split: 112 images.",
        "",
        "| Resolution | Pixels | n | Epoch range | CSRNet val MAE | CSRNet test MAE | "
        "CSRNet test MSE | Epoch range | MobileCount val MAE | MobileCount test MAE | "
        "MobileCount test MSE |",
        "|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for variant, _, pixels in RESOLUTIONS:
        cs, mc = data[(variant, "CSRNet")], data[(variant, "MobileCount")]
        cells = [variant, f"{pixels:,}", str(cs["n"])]
        for c in (cs, mc):
            cells.append(f"{c['epoch_min']}-{c['epoch_max']}")
            for key in ("val_mae", "test_mae", "test_mse"):
                cells.append(f"{c[key+'_mean']:.4f} +- {c[key+'_std']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Coefficient of variation of test MAE (std/mean)", ""]
    for variant, _, _ in RESOLUTIONS:
        cs, mc = data[(variant, "CSRNet")], data[(variant, "MobileCount")]
        lines.append(f"- {variant}: CSRNet {cs['test_mae_cv']*100:.1f}%, "
                     f"MobileCount {mc['test_mae_cv']*100:.1f}%")
    path = os.path.join(OUTDIR, "testset_6runs_table.md")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def write_latex(data):
    best = {net: min(RESOLUTIONS, key=lambda r: data[(r[0], net)]["test_mae_mean"])[0]
            for net in NETS}
    rows = []
    for variant, label, _ in RESOLUTIONS:
        cells = [f"{label:<17}"]
        for net in NETS:
            c = data[(variant, net)]
            cells.append(f"{epoch_cell(c):<12}")
            for key in ("test_mae", "test_mse"):
                m, s = sig(c[key + "_mean"], c[key + "_std"])
                body = f"${m} \\pm {s}$"
                if variant == best[net]:
                    body = f"$\\mathbf{{{m} \\pm {s}}}$"
                cells.append(f"{body:<26}")
        rows.append("        " + " & ".join(cells) + r" \\")
    path = os.path.join(OUTDIR, "testset_6runs_table.tex")
    open(path, "w").write("\n".join(rows) + "\n")
    return path


def main():
    data = collect()
    md = write_markdown(data)
    tex = write_latex(data)
    js = os.path.join(OUTDIR, "testset_6runs.json")
    json.dump([data[k] for k in data], open(js, "w"), indent=1)
    print(f"wrote {md}\nwrote {tex}\nwrote {js}\n")
    print(open(tex).read())


if __name__ == "__main__":
    main()
