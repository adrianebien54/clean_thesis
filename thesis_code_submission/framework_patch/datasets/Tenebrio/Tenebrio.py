import os

import h5py
import numpy as np
from PIL import Image, ImageOps
from torch.utils import data


class Tenebrio(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'
        self.data_files = [f for f in os.listdir(self.img_path)
                           if os.path.isfile(os.path.join(self.img_path, f))]
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform

    def __getitem__(self, index):
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)
        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            img = img.convert('RGB')
        img_w, img_h = img.size

        den_path = os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.h5')
        with h5py.File(den_path, 'r') as f:
            den = f['density'][:].astype(np.float32, copy=False)
        den_h, den_w = den.shape

        # Pad image to next multiple of 8 so encoder strides divide evenly.
        # If GT is at full image scale, zero-pad GT by the same amount so loss shapes match.
        # If GT is at a downsampled scale (e.g. 1/8) it already lines up with the model's
        # stride-8 output and needs no padding.
        pad_w = (8 - img_w % 8) % 8
        pad_h = (8 - img_h % 8) % 8
        if pad_w or pad_h:
            img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)
            if den_w == img_w and den_h == img_h:
                den = np.pad(den, ((0, pad_h), (0, pad_w)), mode='constant')

        den = Image.fromarray(den)
        return img, den

    def get_num_samples(self):
        return self.num_samples
