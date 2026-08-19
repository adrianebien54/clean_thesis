#!/usr/bin/env python3
"""Rank TEST-split images by how much CSRNet and MobileCount disagree.

Companion to viz_density_examples.py: that script renders density maps for a chosen
image, this one picks which image is worth rendering. It scores every test image with
both nets at the three resolutions the figures use (49x33, 386x260, 1544x1038), using
the same best-val-MAE checkpoints as exp/combined_resolution/testset_results.json and
the same inference path as trainer.validate_V1.

Three different notions of "biggest difference" are reported, because they answer
different questions:
  * count gap   |pred_CSRNet - pred_MobileCount|  -- the two nets say different numbers
  * error gap    |err_CSRNet| - |err_MobileCount| -- one net is right and the other is
                 not (signed: positive means MobileCount is the better one here)
  * map distance -- how differently the two density maps are *shaped*. Each map is
                 normalised to sum 1 and compared by total-variation distance
                 (0.5 * L1), so it measures where the mass sits, independently of how
                 much total mass each net predicted. This is the one that predicts how
                 different the two rows of a figure will look. CSRNet's map is bilinearly
                 upsampled from a stride-8 grid while MobileCount's is not, so at low
                 resolution the two live on very different effective grids; the maps are
                 compared at full input resolution, as plotted.
A large count gap with both nets wrong in the same direction is not evidence that one
architecture wins; the error gap is. A large map distance with a small count gap means
the nets put the same total in visibly different places.

Writes exp/combined_resolution/density_examples/net_disagreement.json (per-image rows)
and net_disagreement_table.md (the ranked extracts printed below).
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision.transforms as standard_transforms
from PIL import Image, ImageOps

ROOT = Path(os.environ.get("C3_ROOT") or Path(__file__).resolve().parents[2])
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import config
config.cfg.GPU_ID = [0]

from datasets.Tenebrio.setting import cfg_data
from models.CC import CrowdCounter

TESTSET_JSON = ROOT / "exp/combined_resolution/testset_results.json"
OUT_DIR = ROOT / "exp/combined_resolution/density_examples"
SPLIT = "test"
VARIANTS = ["49x33", "386x260", "1544x1038"]
NETS = ["CSRNet", "MobileCount"]
TOP_N = 8


def load_specs() -> dict:
    rows = json.load(open(TESTSET_JSON))
    return {(r["net"], r["variant"]): r for r in rows}


def stems(variant: str) -> list[str]:
    d = ROOT / f"exp/data/Tenebrio/{variant}/{SPLIT}/img"
    return sorted(p.stem for p in d.glob("*.png"))


def load_input(variant: str, stem: str):
    img = Image.open(ROOT / f"exp/data/Tenebrio/{variant}/{SPLIT}/img/{stem}.png")
    if img.mode == "L":
        img = img.convert("RGB")
    w, h = img.size
    pad_w, pad_h = (8 - w % 8) % 8, (8 - h % 8) % 8
    if pad_w or pad_h:
        img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)
    return img


def gt_count(variant: str, stem: str) -> float:
    with h5py.File(ROOT / f"exp/data/Tenebrio/{variant}/{SPLIT}/den/{stem}.h5", "r") as f:
        return float(f["density"][:].sum())


def load_net(net_name: str, ckpt: Path) -> CrowdCounter:
    net = CrowdCounter(config.cfg.GPU_ID, net_name)
    net.load_state_dict(torch.load(str(ckpt), map_location="cuda", weights_only=True))
    net.cuda()
    net.eval()
    return net


@torch.no_grad()
def score_variant(variant: str, specs: dict, names: list[str]) -> dict[str, dict]:
    """Per-image counts for both nets plus the shape distance between their maps.

    Both nets are resident at once so each image is compared without holding the whole
    split's maps in memory (a 1544x1038 map is 6.4 MB).
    """
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD),
    ])
    nets, log_paras = {}, {}
    for net_name in NETS:
        spec = specs[(net_name, variant)]
        nets[net_name] = load_net(net_name, ROOT / spec["checkpoint"])
        log_paras[net_name] = float(spec["log_para"])

    out = {}
    for stem in names:
        x = img_transform(load_input(variant, stem)).unsqueeze(0).cuda()
        maps, counts = {}, {}
        for net_name in NETS:
            m = nets[net_name].test_forward(x).squeeze()
            counts[net_name] = float(m.sum().item()) / log_paras[net_name]
            maps[net_name] = m
        # shape-only comparison: clamp negatives, normalise each map to sum 1
        a, b = (m.clamp(min=0) for m in (maps["CSRNet"], maps["MobileCount"]))
        a, b = a / a.sum().clamp(min=1e-12), b / b.sum().clamp(min=1e-12)
        out[stem] = {"counts": counts,
                     "map_distance": float(0.5 * (a - b).abs().sum().item())}
        del x, maps, a, b

    del nets
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    specs = load_specs()
    rows = []

    for variant in VARIANTS:
        names = stems(variant)
        gts = {s: gt_count(variant, s) for s in names}
        scored = score_variant(variant, specs, names)
        print(f"scored both nets  {variant:<10} ({len(names)} images)")

        for s in names:
            pc = scored[s]["counts"]["CSRNet"]
            pm = scored[s]["counts"]["MobileCount"]
            gt = gts[s]
            rows.append({
                "variant": variant, "stem": s, "gt": gt,
                "pred_csrnet": pc, "pred_mobilecount": pm,
                "err_csrnet": pc - gt, "err_mobilecount": pm - gt,
                "count_gap": abs(pc - pm),
                "error_gap": abs(pc - gt) - abs(pm - gt),
                "map_distance": scored[s]["map_distance"],
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "net_disagreement.json").open("w") as f:
        json.dump(rows, f, indent=2)

    lines = ["# CSRNet vs MobileCount disagreement on the test split", ""]
    for variant in VARIANTS:
        sub = [r for r in rows if r["variant"] == variant]
        lines += [f"## {variant}", ""]
        for key, title in [("map_distance", "maps most differently shaped (TV distance)"),
                           ("count_gap", "largest count gap |pred_C - pred_M|"),
                           ("error_gap", "MobileCount most ahead (error_gap > 0)"),
                           ("-error_gap", "CSRNet most ahead (error_gap < 0)")]:
            rev = not key.startswith("-")
            k = key.lstrip("-")
            top = sorted(sub, key=lambda r: r[k], reverse=rev)[:TOP_N]
            lines += [f"**{title}**", "",
                      "| image | GT | CSRNet | MobileCount | err_C | err_M | count gap | error gap | map dist |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for r in top:
                lines.append(
                    f"| {r['stem']} | {r['gt']:.0f} | {r['pred_csrnet']:.2f} | "
                    f"{r['pred_mobilecount']:.2f} | {r['err_csrnet']:+.2f} | "
                    f"{r['err_mobilecount']:+.2f} | {r['count_gap']:.2f} | "
                    f"{r['error_gap']:+.2f} | {r['map_distance']:.3f} |")
            lines.append("")

    # images that disagree consistently, averaged over the three resolutions
    by_stem: dict[str, list[dict]] = {}
    for r in rows:
        by_stem.setdefault(r["stem"], []).append(r)
    agg = sorted(
        ({"stem": s,
          "mean_count_gap": float(np.mean([x["count_gap"] for x in v])),
          "mean_error_gap": float(np.mean([x["error_gap"] for x in v])),
          "gt": v[0]["gt"]} for s, v in by_stem.items()),
        key=lambda d: d["mean_count_gap"], reverse=True)
    lines += ["## averaged over the three resolutions", "",
              "| image | GT | mean count gap | mean error gap |", "|---|---|---|---|"]
    for d in agg[:TOP_N]:
        lines.append(f"| {d['stem']} | {d['gt']:.0f} | {d['mean_count_gap']:.2f} | "
                     f"{d['mean_error_gap']:+.2f} |")

    (OUT_DIR / "net_disagreement_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT_DIR/'net_disagreement.json'}")
    print(f"wrote {OUT_DIR/'net_disagreement_table.md'}")


if __name__ == "__main__":
    main()
