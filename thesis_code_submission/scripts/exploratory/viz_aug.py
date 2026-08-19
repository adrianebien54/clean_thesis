"""Visualize the Tenebrio 'extended' augmentation set on real training images.

Produces two figures per resolution variant:
  decomposition.png : one image, stage by stage (geometric -> radiometric -> noise),
                      with amplified diffs so the subtle ±1% radiometric and the sparse
                      noise are clearly visible.
  variety.png       : a grid of original vs. real random 'extended' draws.

Augmentation parameters are read from datasets/Tenebrio/setting.py so the figures always
match what training uses. Geometric ops are applied jointly to image + density map.

Usage:
  python scripts/viz_aug.py --all                       # every variant under DATA_ROOT
  python scripts/viz_aug.py --variant 386x260
  python scripts/viz_aug.py --data-path exp/data/Tenebrio/386x260 --out exp/aug_samples/386x260
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageOps

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as TT

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import misc.transforms as T  # noqa: E402
from datasets.Tenebrio.loading_data import _compute_train_size  # noqa: E402
from datasets.Tenebrio.setting import cfg_data  # noqa: E402

DATA_ROOT = ROOT / 'exp/data/Tenebrio'

# Augmentation params straight from the Tenebrio config (single source of truth).
ROT = getattr(cfg_data, 'AUG_ROTATION', 10.0)
RAD = getattr(cfg_data, 'AUG_RADIOMETRIC', 0.01)
FRAC = getattr(cfg_data, 'AUG_NOISE_FRACTION', 0.08)
STD = getattr(cfg_data, 'AUG_NOISE_STD', 0.1)
RAD_AMP = 20  # amplification for the radiometric-effect diff panel (±1% is near-invisible)

to_tensor = TT.ToTensor()
radiometric = T.RandomRadiometric(RAD, RAD)  # per-channel, on [0,1] tensor
noise = T.AddSparseGaussianNoise(FRAC, STD)


def load(imgdir, dendir, fname):
    img = Image.open(os.path.join(imgdir, fname)).convert('RGB')
    w, h = img.size
    with h5py.File(os.path.join(dendir, os.path.splitext(fname)[0] + '.h5'), 'r') as f:
        den = f['density'][:].astype(np.float32)
    pw, ph = (8 - w % 8) % 8, (8 - h % 8) % 8
    if pw or ph:
        img = ImageOps.expand(img, border=(0, 0, pw, ph), fill=0)
        if den.shape == (h, w):
            den = np.pad(den, ((0, ph), (0, pw)))
    return img, Image.fromarray(den)


def chw_to_hwc(t):
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


def center_crop(img, th, tw):
    w, h = img.size
    x, y = max(0, (w - tw) // 2), max(0, (h - th) // 2)
    return img.crop((x, y, x + tw, y + th))


def pick_sources(imgdir, n=4):
    """n files spread across distinct numeric prefixes (density variety)."""
    files = sorted(f for f in os.listdir(imgdir) if f.lower().endswith('.png'))
    by_prefix = {}
    for f in files:
        by_prefix.setdefault(f.split('_')[0], []).append(f)
    prefixes = sorted(by_prefix, key=lambda x: int(x) if x.isdigit() else 1 << 30)
    if len(prefixes) >= n:
        idx = [round(i * (len(prefixes) - 1) / (n - 1)) for i in range(n)]
        return [by_prefix[prefixes[i]][0] for i in idx]
    return files[:n]


def extended_draw(img, den, train_size):
    i, d = T.RandomHorizontallyFlip()(img, den)
    i, d = T.RandomVerticallyFlip()(i, d)
    i, d = T.RandomRotationJoint(ROT)(i, d)
    i, d = T.RandomCrop(train_size)(i, d)
    return noise(radiometric(to_tensor(i)))     # per-channel radiometric, then noise


def make_figures(data_path, out_dir, variant_name):
    imgdir = os.path.join(data_path, 'train/img')
    dendir = os.path.join(data_path, 'train/den')
    train_size = _compute_train_size(data_path)
    th, tw = train_size
    os.makedirs(out_dir, exist_ok=True)
    sources = pick_sources(imgdir, 4)
    interp = 'nearest' if min(th, tw) < 120 else 'antialiased'

    # ---- Figure 1: decomposition on one (mid-density) image ----
    random.seed(7); torch.manual_seed(7)
    fn0 = sources[len(sources) // 2]
    img, den = load(imgdir, dendir, fn0)
    g_img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(8, resample=Image.BILINEAR, expand=False)
    g_den = den.transpose(Image.FLIP_LEFT_RIGHT).rotate(8, resample=Image.NEAREST, expand=False)
    g_img, _ = T.RandomCrop(train_size)(g_img, g_den)
    base = center_crop(img, th, tw)
    t_geo = to_tensor(g_img)
    t_rad = radiometric(t_geo.clone())              # + per-channel radiometric
    t_noise = noise(t_rad.clone())                  # + sparse noise (final)
    rad_diff = ((t_rad - t_geo) * RAD_AMP + 0.5).clamp(0, 1)
    noise_diff = ((t_noise - t_rad) * 3 + 0.5).clamp(0, 1)
    panels = [
        (np.asarray(base) / 255.0, 'Original\n(center crop)'),
        (chw_to_hwc(t_geo), 'Geometric\n(flip + rotate + crop)'),
        (chw_to_hwc(t_rad), f'+ per-channel radiometric\n(±{RAD:.0%}, jimaging)'),
        (chw_to_hwc(t_noise), '+ sparse noise\n(= final extended)'),
        (chw_to_hwc(rad_diff), f'radiometric effect\n(diff x{RAD_AMP})'),
        (chw_to_hwc(noise_diff), 'noise effect\n(diff x3)'),
    ]
    fig, ax = plt.subplots(1, 6, figsize=(20, 4))
    for a, (im, t) in zip(ax, panels):
        a.imshow(im, interpolation=interp); a.set_title(t, fontsize=11); a.axis('off')
    fig.suptitle(f'Extended augmentation — stage by stage  |  {variant_name}  '
                 f'(train crop {th}x{tw})  '
                 f'per-channel radiometric ±{RAD:.0%} (see amplified diff)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p1 = os.path.join(out_dir, 'decomposition.png')
    fig.savefig(p1, dpi=110, bbox_inches='tight'); plt.close(fig)

    # ---- Figure 2: variety ----
    random.seed(3); torch.manual_seed(3)
    ncol = 4
    fig, ax = plt.subplots(len(sources), ncol, figsize=(4 * ncol, 3 * len(sources)))
    for r, fn in enumerate(sources):
        img, den = load(imgdir, dendir, fn)
        ax[r, 0].imshow(np.asarray(center_crop(img, th, tw)) / 255.0, interpolation=interp)
        ax[r, 0].set_title(f'{fn}\noriginal' if r == 0 else fn, fontsize=10)
        ax[r, 0].axis('off')
        for c in range(1, ncol):
            ax[r, c].imshow(chw_to_hwc(extended_draw(img, den, train_size)), interpolation=interp)
            ax[r, c].set_title('extended draw' if r == 0 else '', fontsize=10)
            ax[r, c].axis('off')
    fig.suptitle(f'Original vs. random "extended" draws  |  {variant_name}  '
                 f'(train crop {th}x{tw})', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p2 = os.path.join(out_dir, 'variety.png')
    fig.savefig(p2, dpi=110, bbox_inches='tight'); plt.close(fig)
    return p1, p2


def discover_variants():
    return sorted(
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and (d / 'train/img').is_dir()
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', help='resolution variant under exp/data/Tenebrio, e.g. 386x260')
    ap.add_argument('--data-path', help='explicit data root (overrides --variant)')
    ap.add_argument('--out', help='output dir (default exp/aug_samples/<variant>)')
    ap.add_argument('--all', action='store_true', help='process every resolution variant')
    args = ap.parse_args()

    if args.all:
        variants = discover_variants()
    elif args.data_path:
        variants = [None]
    elif args.variant:
        variants = [args.variant]
    else:
        ap.error('pass --all, --variant, or --data-path')

    print(f'AUG params: rotation=±{ROT}°  per-channel radiometric ±{RAD}  '
          f'noise: {FRAC:.0%} px @ std={STD}')
    for v in variants:
        if v is None:
            data_path = args.data_path
            name = Path(data_path).name
        else:
            data_path = str(DATA_ROOT / v)
            name = v
        out_dir = args.out or str(ROOT / 'exp/aug_samples' / name)
        p1, p2 = make_figures(data_path, out_dir, name)
        print(f'  [{name}] -> {p1}\n           {p2}')


if __name__ == '__main__':
    main()
