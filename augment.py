#!/usr/bin/env python3

"""
Training augmentations that will be applied to training samples (main.py --aug).

* spatial: random in-plane rotation (+-5 deg), scaling (+-15%)
* intensity: brightness, contrast, gamma, Gaussian noise.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from utils import class2one_hot


AUGMENTATIONS: dict[str, tuple[str, ...]] = {
    'none': (),
    'spatial': ('spatial',),
    'intensity': ('intensity',),
    'all': ('spatial', 'intensity'),
}

# -------------------- CONFIG --------------------

P_SPATIAL: float = 0.3
MAX_ROTATION: float = 5  # degrees
SCALING: tuple[float, float] = (0.85, 1.15)

P_INTENSITY: float = 0.15  # for each indep.
BRIGHTNESS: tuple[float, float] = (0.9, 1.1)
CONTRAST: tuple[float, float] = (0.9, 1.1)
GAMMA: tuple[float, float] = (0.8, 1.25)
MAX_NOISE: float = 0.03  # std of the Gaussian noise, images are in [0, 1]

# --------------------------------------------------

def uniform(low: float, high: float) -> float:
    return low + (high - low) * torch.rand(1).item()


def spatial(img: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
    assert img.shape[1] == img.shape[2], img.shape
    angle: float = math.radians(uniform(-MAX_ROTATION, MAX_ROTATION))
    scale: float = uniform(*SCALING)

    # The 2D planes to transform, as a batch: (N, C, W, H)
    volume: bool = img.ndim == 4
    img_planes: Tensor = img.permute(3, 0, 1, 2) if volume else img[None]
    classes: Tensor = gt.argmax(dim=0).float()  # (W, H) or (W, H, D)
    gt_planes: Tensor = classes.permute(2, 0, 1)[:, None] if volume else classes[None, None]

    # Output to input coordinates: dividing by the scale makes the content larger for scale > 1
    cos, sin = math.cos(angle) / scale, math.sin(angle) / scale
    theta: Tensor = torch.tensor([[cos, -sin, 0.], [sin, cos, 0.]], dtype=torch.float32)
    grid: Tensor = F.affine_grid(theta.expand(img_planes.shape[0], 2, 3), list(img_planes.shape),
                                 align_corners=False)

    # Outside of scan: 0 for the image (the bottom of the HU window), background for the labels
    img_planes = F.grid_sample(img_planes, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    gt_planes = F.grid_sample(gt_planes, grid, mode='nearest', padding_mode='zeros', align_corners=False)

    new_img: Tensor = img_planes.permute(1, 2, 3, 0) if volume else img_planes[0]
    new_classes: Tensor = (gt_planes[:, 0].permute(1, 2, 0) if volume else gt_planes[0, 0]).round().long()
    new_gt: Tensor = class2one_hot(new_classes[None], K=gt.shape[0])[0].type(gt.dtype)

    return new_img.clamp(0, 1), new_gt


def intensity(img: Tensor) -> Tensor:
    if torch.rand(1).item() < P_INTENSITY:
        img = img * uniform(*BRIGHTNESS)
    if torch.rand(1).item() < P_INTENSITY:
        mean: Tensor = img.mean()
        img = (img - mean) * uniform(*CONTRAST) + mean
    img = img.clamp(0, 1)
    if torch.rand(1).item() < P_INTENSITY:
        img = img ** uniform(*GAMMA)
    if torch.rand(1).item() < P_INTENSITY:
        img = img + torch.randn_like(img) * uniform(0, MAX_NOISE)

    return img.clamp(0, 1)


def augment(img: Tensor, gt: Tensor, kinds: tuple[str, ...]) -> tuple[Tensor, Tensor]:
    if 'spatial' in kinds and torch.rand(1).item() < P_SPATIAL:
        img, gt = spatial(img, gt)
    if 'intensity' in kinds:
        img = intensity(img)

    return img, gt
