from easydict import EasyDict as edict

__C_Tenebrio = edict()
cfg_data = __C_Tenebrio

# Path to the prepared Tenebrio data root, relative to where train.py is run
# (the project root). Must contain `train/img`, `train/den`, `val/img`, `val/den`,
# with density maps stored as .csv. Built by running scripts/split_tenebrio.py
# (creates the {train,val,test}/{img,den}/ tree) followed by
# scripts/resize_splits_to_resolutions.py (creates resolution-variant subdirs).
# Switch resolution variants by repointing this, e.g. './exp/data/Tenebrio/772x519'.
__C_Tenebrio.DATA_PATH = './exp/data/Tenebrio/386x260'

# ImageNet mean/std — correct for CSRNet's pretrained VGG16 frontend
__C_Tenebrio.MEAN_STD = (
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225],
)

# Scale density values so MSE loss is large enough for effective gradient flow.
# LOG_PARA is auto-scaled per resolution in loading_data() so that LOG_PARA * raw_density
# (and hence the target magnitude the network regresses) stays ~constant across resolution
# variants — see scripts/sweep_logpara_variable.py, which fixed the high-res target-collapse
# that pinned 1544x1038 MAE at ~11. LOG_PARA cancels out of the count (trainer divides it
# back out), so this only affects gradient/loss scale, not the reported MAE/MSE.
__C_Tenebrio.LOG_PARA_BASE = 100.            # value calibrated for the 386x260 variant
__C_Tenebrio.LOG_PARA_BASE_AREA = 386 * 260  # area at which LOG_PARA == LOG_PARA_BASE
__C_Tenebrio.LOG_PARA = 100.                 # runtime value; overwritten per-resolution by loading_data()

__C_Tenebrio.RESUME_MODEL = ''

__C_Tenebrio.TRAIN_BATCH_SIZE = 6
__C_Tenebrio.VAL_BATCH_SIZE = 1

# ---- Data augmentation (Papadopoulos et al. 2024, TenebrioVision) -------------------
# Two paradigms from the source, plus an off switch. Built in loading_data():
#   'none'     : RandomCrop only (train-size crop; no other augmentation)
#   'basic'    : + horizontal & vertical flips (p=0.5 each)
#   'extended' : basic + random rotation + per-channel radiometric offset/gain
#                + sparse Gaussian noise
# Zoom (in the source's extended set) is INTENTIONALLY OMITTED: it rescales the larvae,
# injecting object-scale jitter — and object scale is the variable the resolution study
# isolates. Magnitudes the source leaves unspecified are taken from jimaging7020021
# (rotation, per-channel radiometric ±1%) or the torchvision default (noise std); the 8%
# noise fraction is from the source. Geometric augs act jointly on image + density map.
__C_Tenebrio.AUG = 'none'
__C_Tenebrio.AUG_ROTATION = 10.0        # ± degrees (jimaging7020021)
__C_Tenebrio.AUG_RADIOMETRIC = 0.01     # per-channel offset & gain, ±1% (jimaging7020021)
__C_Tenebrio.AUG_NOISE_FRACTION = 0.08  # fraction of pixels noised (TenebrioVision)
__C_Tenebrio.AUG_NOISE_STD = 0.1        # torchvision GaussianNoise default (source silent)
