"""Standalone visualization / sanity reference for the Tenebrio augmentation sets.

The augmentation actually used in training lives in the dataset pipeline, selected by
`cfg_data.AUG` ('none' | 'basic' | 'extended') in datasets/Tenebrio/setting.py and
assembled in datasets/Tenebrio/loading_data.py. The photometric transforms below are
IMPORTED from misc.transforms so this file can never drift from what training uses.

Two caveats about the pipelines defined here:
  * They are IMAGE-ONLY (for eyeballing what an augmented frame looks like). In training
    the geometric ops — flips and rotation — are applied JOINTLY to the image and the
    density map (misc.transforms.RandomHorizontallyFlip / RandomVerticallyFlip /
    RandomRotationJoint) so the count is preserved. Do not use these image-only
    pipelines to train.
  * Zoom (in the source's extended set) is omitted on purpose: it rescales the larvae
    and would confound object scale, the variable the resolution study isolates.

Augmentation paradigms (Papadopoulos et al. 2024, TenebrioVision):
  basic    = horizontal + vertical flips (p=0.5 each)
  extended = basic + random rotation (±10°, jimaging7020021)
                   + per-channel radiometric offset & gain (±1%, jimaging7020021)
                   + sparse Gaussian noise on 8% of pixels (std=0.1, torchvision default)
"""
import sys
from pathlib import Path

import torch
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from misc.transforms import RandomRadiometric, AddSparseGaussianNoise  # noqa: E402

# Defaults mirror datasets/Tenebrio/setting.py (kept in sync via the shared transforms).
ROTATION_DEG = 10.0
RADIOMETRIC = 0.01
NOISE_FRACTION = 0.08
NOISE_STD = 0.1

basic = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
])

extended = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=ROTATION_DEG),
    transforms.ToTensor(),                                   # -> [0,1] CxHxW
    RandomRadiometric(gain=RADIOMETRIC, offset=RADIOMETRIC),  # per-channel offset & gain
    AddSparseGaussianNoise(fraction=NOISE_FRACTION, std=NOISE_STD),
])


if __name__ == "__main__":
    from PIL import Image
    torch.manual_seed(0)
    img = Image.fromarray((torch.rand(128, 128, 3) * 255).byte().numpy())
    for name, pipe in [("basic", basic), ("extended", extended)]:
        out = pipe(img)
        print(f"{name:8s} -> {tuple(out.shape)}  dtype={out.dtype}  "
              f"range=[{out.min():.3f}, {out.max():.3f}]")
