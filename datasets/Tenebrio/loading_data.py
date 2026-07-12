import math
import os

import torchvision.transforms as standard_transforms
from PIL import Image
from torch.utils.data import DataLoader

import misc.transforms as own_transforms
from .Tenebrio import Tenebrio
from .setting import cfg_data


def _compute_train_size(data_path):
    """80% of image (H, W), rounded UP to the next multiple of 8."""
    img_dir = os.path.join(data_path, 'train', 'img')
    sample = next(f for f in os.listdir(img_dir) if f.lower().endswith('.png'))
    with Image.open(os.path.join(img_dir, sample)) as im:
        w, h = im.size
    ceil8 = lambda x: max(8, math.ceil(x * 0.8 / 8) * 8)
    return (ceil8(h), ceil8(w))


def _compute_log_para(data_path, base_log_para, base_area):
    """LOG_PARA scaled by image area, so LOG_PARA*raw_density (the target magnitude the
    network regresses) stays ~constant across resolution variants. Anchored on
    (base_log_para, base_area). See scripts/sweep_logpara_variable.py."""
    img_dir = os.path.join(data_path, 'train', 'img')
    sample = next(f for f in os.listdir(img_dir) if f.lower().endswith('.png'))
    with Image.open(os.path.join(img_dir, sample)) as im:
        w, h = im.size
    return base_log_para * (w * h) / base_area


def loading_data():
    mean_std = cfg_data.MEAN_STD

    cfg_data.LOG_PARA = _compute_log_para(
        cfg_data.DATA_PATH, cfg_data.LOG_PARA_BASE, cfg_data.LOG_PARA_BASE_AREA)
    log_para = cfg_data.LOG_PARA

    aug = getattr(cfg_data, 'AUG', 'none')
    if aug not in ('raw', 'none', 'basic', 'extended'):
        raise ValueError(f"cfg_data.AUG={aug!r}; expected 'raw', 'none', 'basic' or 'extended'")

    train_size = None if aug == 'raw' else _compute_train_size(cfg_data.DATA_PATH)
    size_info = 'full-image' if aug == 'raw' else f'{train_size} (H,W)'
    print(f'[Tenebrio] DATA_PATH={cfg_data.DATA_PATH}  TRAIN_SIZE={size_info}'
          f'  LOG_PARA={log_para:.2f}  AUG={aug}')

    # Geometric augmentations act jointly on image + density map (so the count is
    # preserved) and run only on the train split, before the train-size crop.
    # 'raw' skips the geometric stage entirely: train sees the full (padded) image
    # every epoch, exactly like val.
    if aug == 'raw':
        train_main_transform = None
    else:
        geo_transforms = []
        if aug in ('basic', 'extended'):
            geo_transforms += [
                own_transforms.RandomHorizontallyFlip(),
                own_transforms.RandomVerticallyFlip(),
            ]
        if aug == 'extended':
            geo_transforms.append(own_transforms.RandomRotationJoint(cfg_data.AUG_ROTATION))
        geo_transforms.append(own_transforms.RandomCrop(train_size))
        train_main_transform = own_transforms.Compose(geo_transforms)
    val_main_transform = None

    # Photometric augmentations touch the image only (density map unchanged), train only;
    # they run in [0,1] space between ToTensor and Normalize.
    photometric = []
    if aug == 'extended':
        photometric = [
            own_transforms.RandomRadiometric(cfg_data.AUG_RADIOMETRIC, cfg_data.AUG_RADIOMETRIC),
            own_transforms.AddSparseGaussianNoise(cfg_data.AUG_NOISE_FRACTION, cfg_data.AUG_NOISE_STD),
        ]
    train_img_transform = standard_transforms.Compose(
        [standard_transforms.ToTensor()]
        + photometric
        + [standard_transforms.Normalize(*mean_std)]
    )
    val_img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std),
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.LabelNormalize(log_para),
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage(),
    ])

    train_set = Tenebrio(
        cfg_data.DATA_PATH + '/train', 'train',
        main_transform=train_main_transform,
        img_transform=train_img_transform,
        gt_transform=gt_transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg_data.TRAIN_BATCH_SIZE,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )

    val_set = Tenebrio(
        cfg_data.DATA_PATH + '/val', 'val',
        main_transform=val_main_transform,
        img_transform=val_img_transform,
        gt_transform=gt_transform,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg_data.VAL_BATCH_SIZE,
        num_workers=8,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, restore_transform
