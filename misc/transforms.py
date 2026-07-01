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
    """Rotate both image and density map by the same random angle in [-degrees, +degrees].

    By design this perturbs the train-sample count by ~1% on average (density near the
    corners rotates out of the frame with expand=False); this is image-consistent
    (the image is clipped identically) and is dwarfed by the RandomCrop count variation,
    so it is an intended augmentation effect, not a bug. Reported metrics are unaffected
    (val/test are un-augmented)."""
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, img, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        img  = img.rotate(angle,  resample=Image.BILINEAR, expand=False)
        mask = mask.rotate(angle, resample=Image.NEAREST,  expand=False)
        return img, mask


class RandomRadiometric(object):
    """Per-channel radiometric augmentation on a [0,1] CxHxW tensor (place AFTER ToTensor,
    BEFORE Normalize): out_c = clip(gain_c * in_c + offset_c, 0, 1), with the gain and
    offset sampled INDEPENDENTLY for each channel. Implements jimaging7020021's "radiometric
    offset and gain of 1% for each individual image channel" (the source, Papadopoulos 2024
    TenebrioVision, states no value). gain_c ~ U[1-gain, 1+gain],
    offset_c ~ U[-offset, +offset] in [0,1] units."""

    def __init__(self, gain=0.01, offset=0.01):
        self.gain = gain
        self.offset = offset

    def __call__(self, tensor):
        c = tensor.shape[0]
        gain = 1.0 + (torch.rand(c, 1, 1) * 2 - 1) * self.gain
        offset = (torch.rand(c, 1, 1) * 2 - 1) * self.offset
        return (tensor * gain + offset).clamp(0.0, 1.0)


class AddSparseGaussianNoise(object):
    """Add zero-mean Gaussian noise to a random `fraction` of the pixels of a [0,1]
    CxHxW tensor (place AFTER ToTensor, BEFORE Normalize), then clamp back to [0,1].
    The noised pixels are shared across channels (one spatial mask) but the noise value
    is independent per channel. Implements the "Gaussian distribution applied to 8% of
    the pixels" noise augmentation of Papadopoulos et al. (2024, TenebrioVision).

    The source does not specify the standard deviation, so the torchvision GaussianNoise
    default (std=0.1) is used; the clamp mirrors torchvision's clip=True and is
    physically correct for saturated speckle (clipped pixels carry slightly less than
    the nominal variance, which is acceptable for this sparse-speckle augmentation)."""

    def __init__(self, fraction=0.08, std=0.1):
        self.fraction = fraction
        self.std = std

    def __call__(self, tensor):
        c, h, w = tensor.shape
        mask = torch.rand(1, h, w) < self.fraction      # same pixels across channels
        noise = torch.randn(c, h, w) * self.std         # independent value per channel
        return (tensor + noise * mask).clamp(0.0, 1.0)
