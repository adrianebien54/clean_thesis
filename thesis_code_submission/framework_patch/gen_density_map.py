import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as standard_transforms
from matplotlib import pyplot as plt
from PIL import Image

import misc.transforms as own_transforms
from config import cfg
from models.CC import CrowdCounter


torch.cuda.set_device(cfg.GPU_ID[0])
torch.backends.cudnn.benchmark = True

exp_name = './density_map_outputs'
if not os.path.exists(exp_name):
    os.mkdir(exp_name)

mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],
            [0.23242045939, 0.224925786257, 0.221840232611])

img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std),
])
restore = standard_transforms.Compose([
    own_transforms.DeNormalize(*mean_std),
    standard_transforms.ToPILImage(),
])

# Set these to point at a trained checkpoint and a test set with img/ and den/ subdirs.
dataRoot = './exp/data/UCF-QNRF-1024x1024-mod16/test'
model_path = './exp/04-23_00-08_QNRF_MobileCount_0.0001/all_ep_448_mae_131.1_mse_222.6.pth'


def main():
    file_list = [filename for root, dirs, filename in os.walk(dataRoot + '/img/')]
    test(file_list[0], model_path)


def test(file_list, model_path):
    net = CrowdCounter(cfg.GPU_ID, cfg.NET)
    net.load_state_dict(torch.load(model_path))
    net.cuda()
    net.eval()

    for filename in file_list:
        print(filename)
        imgname = dataRoot + '/img/' + filename
        filename_no_ext = filename.split('.')[0]
        denname = dataRoot + '/den/' + filename_no_ext + '.csv'

        den = pd.read_csv(denname, sep=',', header=None).values.astype(np.float32, copy=False)

        img = Image.open(imgname)
        if img.mode == 'L':
            img = img.convert('RGB')

        wd_1, ht_1 = img.size
        img = img_transform(img)

        with torch.no_grad():
            img = img[None, :, :, :].cuda()
            pred_map = net.test_forward(img)
        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]

        gt_count = np.sum(den)
        pred_cnt = np.sum(pred_map) / 2550.0
        print('gt={:.2f}, pred={:.2f}'.format(gt_count, pred_cnt))

        den = den / np.max(den + 1e-20)
        den = den[0:ht_1, 0:wd_1]
        plt.figure('gt-den' + filename)
        plt.imshow(den)
        plt.show()

        pred_map = pred_map / np.max(pred_map + 1e-20)
        pred_map = pred_map[0:ht_1, 0:wd_1]
        plt.figure('pre-den' + filename)
        plt.imshow(pred_map)
        plt.show()


if __name__ == '__main__':
    main()
