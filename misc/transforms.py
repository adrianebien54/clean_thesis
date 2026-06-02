import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from config import cfg
import torch
# ===============================img tranforms============================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, bbx=None):
        if bbx is None:
            for t in self.transforms:
                img, mask = t(img, mask)
            return img, mask
        for t in self.transforms:
            img, mask, bbx = t(img, mask, bbx)
        return img, mask, bbx

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, bbx=None):
        if random.random() < 0.5:
            if bbx is None:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
            w, h = img.size
            xmin = w - bbx[:,3]
            xmax = w - bbx[:,1]
            bbx[:,1] = xmin
            bbx[:,3] = xmax
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), bbx
        if bbx is None:
            return img, mask
        return img, mask, bbx

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, dst_size=None):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        if dst_size is None:
            th, tw = self.size
        else:
            th, tw = dst_size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))



class FreeScale(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, img, mask):
        return img.resize((self.size[1], self.size[0]), Image.BILINEAR), mask.resize((self.size[1], self.size[0]), Image.NEAREST)


class ScaleDown(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, mask):
        return  mask.resize((self.size[1]/cfg.TRAIN.DOWNRATE, self.size[0]/cfg.TRAIN.DOWNRATE), Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        if img.size != mask.size:
            print( img.size )
            print( mask.size )          
        assert img.size == mask.size
        w, h = img.size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img, mask
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


# ===============================label tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        # tensor = 1./(tensor+self.para).log()
        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor*self.para
        return tensor

class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, img):
        w, h = img.size
        if self.factor==1:
            return img
        tmp = np.array(img.resize((w//self.factor, h//self.factor), Image.BICUBIC))*self.factor*self.factor
        img = Image.fromarray(tmp)
        return img


# ===============================augmentation transforms============================

class RandomVerticallyFlip(object):
    """Vertically flip both image and density map with probability 0.5."""
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        return img, mask


class RandomRotationJoint(object):
    """Rotate both image and density map by the same random angle in [-degrees, +degrees]."""
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        img  = img.rotate(angle,  resample=Image.BILINEAR, expand=False)
        mask = mask.rotate(angle, resample=Image.NEAREST,  expand=False)
        return img, mask


class RandomTranslationJoint(object):
    """Translate both image and density map by the same random pixel offset."""
    def __init__(self, translate):
        self.translate = translate  # e.g. 0.02 for ±2%

    def __call__(self, img, mask):
        w, h = img.size
        dx = random.uniform(-self.translate, self.translate) * w
        dy = random.uniform(-self.translate, self.translate) * h
        data = (1, 0, -dx, 0, 1, -dy)
        img  = img.transform(img.size,   Image.AFFINE, data, resample=Image.BILINEAR)
        mask = mask.transform(mask.size, Image.AFFINE, data, resample=Image.NEAREST)
        return img, mask


class RandomContrast(object):
    """Multiply all channel intensities by a uniform random gain (PIL image, before ToTensor)."""
    def __init__(self, lo=0.99, hi=1.01):
        self.lo, self.hi = lo, hi

    def __call__(self, img):
        factor = random.uniform(self.lo, self.hi)
        arr = np.clip(np.array(img, dtype=np.float32) * factor, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


class RandomBrightness(object):
    """Add an independent per-channel brightness offset in [-max_offset, +max_offset] pixel units."""
    def __init__(self, max_offset=2.5):
        self.max_offset = max_offset

    def __call__(self, img):
        arr = np.array(img, dtype=np.float32)
        for c in range(arr.shape[2]):
            arr[:, :, c] = np.clip(
                arr[:, :, c] + random.uniform(-self.max_offset, self.max_offset), 0, 255
            )
        return Image.fromarray(arr.astype(np.uint8))


class AddGaussianNoise(object):
    """Add zero-mean Gaussian noise to a normalised tensor (after ToTensor + Normalize)."""
    def __init__(self, std=0.02):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std
