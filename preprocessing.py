#!/usr/bin/env python3

# MIT License

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

"""
Volume-level preprocessing for the SegTHOR CT scans, shared by slice_segthor.py
(forward) and stitch.py (inverse).

* HU windowing: clip the Hounsfield units to a fixed [hu_min, hu_max] window and
  map it linearly to [0, 255]. Unlike the per-volume min-max of norm_arr, the
  same HU value then always gets the same intensity, across all patients.
* Resampling: bring every volume to the same voxel spacing (in mm), so one
  pixel covers the same physical size for all patients. The in-plane size is
  then centre cropped/padded to the network input shape, instead of resized
  (which would undo the resampling).
"""

from typing import Sequence

import numpy as np
from skimage.transform import resize


def window_hu(ct: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    assert hu_min < hu_max, (hu_min, hu_max)

    clipped = np.clip(ct.astype(np.float32), hu_min, hu_max)
    res = 255 * (clipped - hu_min) / (hu_max - hu_min)

    assert 0 <= res.min() and res.max() <= 255, (res.min(), res.max())

    return np.round(res).astype(np.uint8)


def complete_spacing(target: Sequence[float], spacing: Sequence[float]) -> tuple[float, ...]:
    """
    A target given for (x, y) only keeps the original slice thickness, so the
    number of slices (and thus the 2D slices themselves) are left untouched in z.
    """
    assert len(target) in [2, 3], target
    assert len(spacing) == 3, spacing

    if len(target) == 2:
        return (*map(float, target), float(spacing[2]))
    return tuple(map(float, target))


def resampled_shape(shape: Sequence[int], spacing: Sequence[float],
                    target: Sequence[float]) -> tuple[int, ...]:
    assert len(shape) == len(spacing) == len(target)

    return tuple(max(1, int(round(n * s / t))) for n, s, t in zip(shape, spacing, target))


def resample(arr: np.ndarray, spacing: Sequence[float], target: Sequence[float],
             is_label: bool) -> np.ndarray:
    """
    Linear interpolation for the CT (with anti-aliasing when downsampling),
    nearest neighbour for the labels so no new class values are created.
    """
    new_shape = resampled_shape(arr.shape, spacing, target)
    if new_shape == arr.shape:
        return arr

    if is_label:
        res = resize(arr, new_shape, order=0, mode="edge",
                     preserve_range=True, anti_aliasing=False)
        return res.astype(arr.dtype)

    res = resize(arr.astype(np.float32), new_shape, order=1, mode="edge",
                 preserve_range=True, anti_aliasing=True)
    return res.astype(np.float32)


def crop_or_pad(arr: np.ndarray, shape: Sequence[int], pad_value: float = 0) -> np.ndarray:
    """
    Centre crop/pad the leading len(shape) axes of arr to shape. Since the
    offsets only depend on the two shapes, crop_or_pad(crop_or_pad(a, s), a.shape)
    puts the kept content back exactly where it came from, which stitch.py uses
    to invert it.
    """
    assert len(shape) <= arr.ndim

    pad_width: list[tuple[int, int]] = []
    slices: list[slice] = []
    for current, wanted in zip(arr.shape, shape):
        diff = wanted - current
        if diff >= 0:
            pad_width.append((diff // 2, diff - diff // 2))
            slices.append(slice(None))
        else:
            start = (-diff) // 2
            pad_width.append((0, 0))
            slices.append(slice(start, start + wanted))

    pad_width += [(0, 0)] * (arr.ndim - len(shape))

    res = np.pad(arr[tuple(slices)], pad_width, mode="constant", constant_values=pad_value)
    assert res.shape[:len(shape)] == tuple(shape), (res.shape, shape)

    return res
