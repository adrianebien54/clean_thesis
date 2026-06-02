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

# Scale density values by 100 so MSE loss is large enough for effective gradient flow.
# Density tensors will sum to count*100; MAE computation divides back by LOG_PARA.
__C_Tenebrio.LOG_PARA = 100.

__C_Tenebrio.RESUME_MODEL = ''

__C_Tenebrio.TRAIN_BATCH_SIZE = 6
__C_Tenebrio.VAL_BATCH_SIZE = 1
