#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union

import torch
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset

from utils import class2one_hot


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=False, equalize=False, debug=False,
                 context: int = 0):
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: bool = augment
        self.equalize: bool = equalize
        # 2.5D: the `context` slices before and after are stacked as channels around the slice
        self.context: int = context

        self.test_mode: bool = subset == 'test'

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def neighbour(self, index: int, offset: int) -> Path:
        """
        The slice `offset` further in z, from the same scan. The files are sorted, so that is
        `offset` further in the list; past the first or last slice of the scan, that slice is
        repeated instead of reading into the next patient.
        """
        scan: str = self.files[index][0].stem.rsplit('_', 1)[0]

        step: int = 1 if offset > 0 else -1
        for _ in range(abs(offset)):
            if not 0 <= index + step < len(self.files):
                break
            if self.files[index + step][0].stem.rsplit('_', 1)[0] != scan:
                break
            index += step

        return self.files[index][0]

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        img: Tensor = torch.cat([self.img_transform(Image.open(self.neighbour(index, o)))
                                 for o in range(-self.context, self.context + 1)])

        data_dict = {"images": img,
                     "stems": img_path.stem}

        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = img.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            data_dict["gts"] = gt

        return data_dict


class VolumeDataset(Dataset):
    """
    The 3D counterpart of SliceDataset, on the same sliced data: the slices of a scan are
    stacked back into its volume, so that a 3D network gets the same split and the same
    preprocessing as the 2D ones.

    * train: random patches, `samples_per_volume` of them per scan in an epoch. Half of them
             are centred on an organ voxel: the organs are a few percents of a scan, and
             patches drawn uniformly would mostly be background.
    * val:   the whole volumes. They differ in depth, so use a batch size of 1.
    """
    def __init__(self, subset, root_dir, img_transform, gt_transform,
                 patch: tuple[int, int, int], samples_per_volume: int = 32, debug=False):
        assert subset in ['train', 'val']
        self.train_mode: bool = subset == 'train'
        self.patch: Tensor = torch.tensor(patch)
        self.samples_per_volume: int = samples_per_volume

        scans: dict[str, list[tuple[Path, Path]]] = {}
        for img_path, gt_path in make_dataset(root_dir, subset):
            scans.setdefault(img_path.stem.rsplit('_', 1)[0], []).append((img_path, gt_path))
        if debug:
            scans = dict(list(scans.items())[:2])
            self.samples_per_volume = 2

        self.stems: list[str] = []
        self.imgs: list[Tensor] = []  # (1, W, H, D), float
        self.gts: list[Tensor] = []  # (W, H, D), the classes: 5 times smaller in memory than one-hot
        self.foregrounds: list[Tensor] = []  # (N, 3), the coordinates of the organ voxels
        for stem, files in scans.items():
            # No missing slice, and in order: z is then the index in the stack
            assert [int(i.stem.rsplit('_', 1)[1]) for i, _ in files] == list(range(len(files))), stem
            assert all(i.stem == g.stem for i, g in files), stem

            gt: Tensor = torch.stack([gt_transform(Image.open(g)) for _, g in files], dim=-1)
            self.K: int = gt.shape[0]
            self.stems.append(stem)
            self.imgs.append(torch.stack([img_transform(Image.open(i)) for i, _ in files], dim=-1))
            self.gts.append(gt.argmax(dim=0).type(torch.uint8))
            self.foregrounds.append(torch.nonzero(self.gts[-1]))

            assert self.imgs[-1].shape[1:] == self.gts[-1].shape
            assert (torch.tensor(self.gts[-1].shape) >= self.patch).all(), (stem, self.gts[-1].shape, patch)

        print(f">> Created {subset} dataset with {len(self.stems)} volumes"
              + (f", {len(self)} patches per epoch..." if self.train_mode else "..."))

    def __len__(self):
        return len(self.stems) * self.samples_per_volume if self.train_mode else len(self.stems)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        scan: int = index % len(self.stems)
        img: Tensor = self.imgs[scan]
        gt: Tensor = self.gts[scan]

        if self.train_mode:
            # The highest corner a patch can have and still be inside the volume
            max_corner: Tensor = torch.tensor(gt.shape) - self.patch
            fg: Tensor = self.foregrounds[scan]

            corner: Tensor
            if len(fg) and torch.rand(1).item() < 0.5:
                centre: Tensor = fg[torch.randint(len(fg), (1,)).item()]
                corner = torch.minimum((centre - self.patch // 2).clamp(min=0), max_corner)
            else:
                corner = (torch.rand(3) * (max_corner + 1)).long()

            x, y, z = corner.tolist()
            w, h, d = self.patch.tolist()
            img = img[:, x:x + w, y:y + h, z:z + d]
            gt = gt[x:x + w, y:y + h, z:z + d]

        return {"images": img,
                "gts": class2one_hot(gt[None, ...].long(), K=self.K)[0],
                "stems": self.stems[scan]}
