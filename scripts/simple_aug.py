import torch
from torchvision import transforms


class AddRandomNoise:
    """Add zero-mean Gaussian noise to a random fraction of pixels (default 8%).
    Expects a CxHxW float tensor in [0, 1]; place after ToTensor()."""

    def __init__(self, fraction=0.08, std=0.1):
        self.fraction = fraction
        self.std = std

    def __call__(self, img):
        c, h, w = img.shape
        mask = torch.rand(1, h, w) < self.fraction      # same pixels across channels
        noise = torch.randn(c, h, w) * self.std
        return (img + noise * mask).clamp(0.0, 1.0)


basic = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
])

extended = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),                 # [-10, +10] degrees
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),  # up to 2% shift
    transforms.ColorJitter(brightness=0.01, contrast=0.01),
    transforms.ToTensor(),
    AddRandomNoise(fraction=0.08, std=0.1),                # std NOT given by source; tune
])


if __name__ == "__main__":
    from PIL import Image
    torch.manual_seed(0)
    img = Image.fromarray((torch.rand(128, 128, 3) * 255).byte().numpy())
    for name, pipe in [("basic", basic), ("extended", extended)]:
        out = pipe(img)
        print(f"{name:8s} -> {tuple(out.shape)}  dtype={out.dtype}  "
              f"range=[{out.min():.3f}, {out.max():.3f}]")
