import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from misc.transforms import (
    Compose, RandomHorizontallyFlip, RandomVerticallyFlip,
    RandomRotationJoint, RandomTranslationJoint,
    RandomContrast, RandomBrightness, AddGaussianNoise,
)
from .Tenebrio import Tenebrio
from .setting import cfg_data
from config import cfg


def loading_data():
    mean_std = cfg_data.MEAN_STD
    log_para  = cfg_data.LOG_PARA
    aug_set   = getattr(cfg, 'AUG_SET', 0)

    # --- Joint spatial augmentations (applied to both image and density map) ---
    if aug_set == 1:
        train_main_transform = Compose([
            RandomHorizontallyFlip(),
            RandomVerticallyFlip(),
            RandomRotationJoint(degrees=10),
            RandomTranslationJoint(translate=0.02),
        ])
    elif aug_set == 2:
        train_main_transform = Compose([
            RandomHorizontallyFlip(),
            RandomVerticallyFlip(),
        ])
    else:
        train_main_transform = None

    # --- Image-only transforms (radiometric augmentations go here) ---
    img_t = []
    if aug_set == 1:
        img_t += [RandomContrast(0.99, 1.01), RandomBrightness(2.5)]
    img_t += [
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std),
    ]
    if aug_set == 1:
        img_t.append(AddGaussianNoise(std=0.02))
    img_transform = standard_transforms.Compose(img_t)

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
        img_transform=img_transform,
        gt_transform=gt_transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg_data.TRAIN_BATCH_SIZE,
        num_workers=4,
        shuffle=True,
        drop_last=False,
    )

    val_set = Tenebrio(
        cfg_data.DATA_PATH + '/val', 'val',
        main_transform=None,
        img_transform=standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std),
        ]),
        gt_transform=gt_transform,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg_data.VAL_BATCH_SIZE,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, restore_transform
