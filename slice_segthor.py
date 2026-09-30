#!/usr/bin/env python3.7

# MIT License

# Copyright (c) 2024 Hoel Kervadec

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

import json
import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable

import numpy as np
import nibabel as nib
from skimage.io import imsave
from skimage.transform import resize

from utils import map_, tqdm_
from preprocessing import complete_spacing, crop_or_pad, resample, window_hu


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = 255 * norm

    assert 0 == res.min(), res.min()
    assert res.max() == 255, res.max()

    return res.astype(np.uint8)


def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx  # Rounding error
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True


def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype

    # Do the test on 3d: assume all organs are present..
    # assert set(np.unique(gt)) == set(range(5))

    return True


resize_: Callable = partial(resize, mode="constant", preserve_range=True, anti_aliasing=False)


def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int],
                  test_mode: bool = False,
                  hu_window: tuple[float, float] | None = None,
                  spacing: tuple[float, ...] | None = None) -> tuple[tuple[float, float, float], dict[str, Any]]:
    id_path: Path = source_path / ("train" if not test_mode else "test") / id_

    ct_path: Path = (id_path / f"{id_}.nii.gz") if not test_mode else (source_path / "test" / f"{id_}.nii.gz")
    nib_obj = nib.load(str(ct_path))
    ct: np.ndarray = np.asarray(nib_obj.dataobj)
    # dx, dy, dz = nib_obj.header.get_zooms()
    x, y, z = ct.shape
    dx, dy, dz = nib_obj.header.get_zooms()

    assert sanity_ct(ct, *ct.shape, *nib_obj.header.get_zooms())

    gt: np.ndarray
    if not test_mode:
        gt_path: Path = id_path / "GT.nii.gz"
        gt_nib = nib.load(str(gt_path))
        # print(nib_obj.affine, gt_nib.affine)
        gt = np.asarray(gt_nib.dataobj)
        assert sanity_gt(gt, ct)
    else:
        gt = np.zeros_like(ct, dtype=np.uint8)

    # Resampling is done on the raw HU, so the windowing below also clips any
    # interpolation artefact
    to_resample_ct: np.ndarray = ct
    if spacing is not None:
        target: tuple[float, ...] = complete_spacing(spacing, (dx, dy, dz))
        to_resample_ct = resample(ct, (dx, dy, dz), target, is_label=False)
        gt = resample(gt, (dx, dy, dz), target, is_label=True)
        assert to_resample_ct.shape == gt.shape

        # The crop should only remove body periphery, never the organs
        lost: int = np.count_nonzero(gt) - np.count_nonzero(crop_or_pad(gt, shape))
        if lost > 0:
            print(f"[slice] Warning: {id_} loses {lost} foreground voxels when cropping to {shape}")

    norm_ct: np.ndarray = window_hu(to_resample_ct, *hu_window) if hu_window is not None \
        else norm_arr(to_resample_ct)

    to_slice_ct = norm_ct
    to_slice_gt = gt

    for idz in range(to_slice_ct.shape[2]):
        if spacing is not None:
            # A resize would undo the resampling, so crop/pad to the network size instead
            img_slice = crop_or_pad(to_slice_ct[:, :, idz], shape)
            gt_slice = crop_or_pad(to_slice_gt[:, :, idz], shape)
        else:
            img_slice = resize_(to_slice_ct[:, :, idz], shape).astype(np.uint8)
            gt_slice = resize_(to_slice_gt[:, :, idz], shape, order=0).astype(np.uint8)
        assert img_slice.shape == gt_slice.shape
        gt_slice *= 63
        assert gt_slice.dtype == np.uint8, gt_slice.dtype
        # assert set(np.unique(gt_slice)) <= set(range(5))
        assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)

        arrays: list[np.ndarray] = [img_slice, gt_slice]

        subfolders: list[str] = ["img", "gt"]
        assert len(arrays) == len(subfolders)
        for save_subfolder, data in zip(subfolders,
                                        arrays):
            filename = f"{id_}_{idz:04d}.png"

            save_path: Path = Path(dest_path, save_subfolder)
            save_path.mkdir(parents=True, exist_ok=True)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                imsave(str(save_path / filename), data)

    # Everything stitch.py needs to map the slices back to the original scan
    meta: dict[str, Any] = {"orig_shape": [x, y, z],
                            "orig_spacing": [float(dx), float(dy), float(dz)],
                            "resampled_shape": list(to_slice_ct.shape)}

    return (dx, dy, dz), meta


def get_splits(src_path: Path, retains: int, fold: int) -> tuple[list[str], list[str], list[str]]:
    ids: list[str] = sorted(map_(lambda p: p.name, (src_path / 'train').glob('*')))
    print(f"Founds {len(ids)} in the id list")
    print(ids[:10])
    assert len(ids) > retains

    random.shuffle(ids)  # Shuffle before to avoid any problem if the patients are sorted in any way
    validation_slice = slice(fold * retains, (fold + 1) * retains)
    validation_ids: list[str] = ids[validation_slice]
    assert len(validation_ids) == retains

    training_ids: list[str] = [e for e in ids if e not in validation_ids]
    assert (len(training_ids) + len(validation_ids)) == len(ids)

    test_ids: list[str] = sorted(map_(lambda p: Path(p.stem).stem, (src_path / 'test').glob('*')))
    print(f"Founds {len(test_ids)} test ids")
    print(test_ids[:10])

    return training_ids, validation_ids, test_ids


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the clean up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    training_ids, validation_ids, test_ids = get_splits(src_path, args.retains, args.fold)

    resolution_dict: dict[str, tuple[float, float, float]] = {}
    preproc_dict: dict[str, Any] = {"hu_window": args.hu_window,
                                    "spacing": args.spacing,
                                    "shape": args.shape,
                                    "patients": {}}

    split_ids: list[str]
    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape),
                                 test_mode=mode == 'test',
                                 hu_window=args.hu_window,
                                 spacing=args.spacing)
        results: list[tuple[tuple[float, float, float], dict[str, Any]]]
        iterator = tqdm_(split_ids)
        match args.process:
            case 1:
                results = list(map(pfun, iterator))
            case -1:
                results = Pool().map(pfun, iterator)
            case _ as p:
                results = Pool(p).map(pfun, iterator)

        for key, (val, meta) in zip(split_ids, results):
            resolution_dict[key] = val
            preproc_dict["patients"][key] = meta

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")

    with open(dest_path / "preprocessing.json", 'w') as f:
        json.dump(preproc_dict, f, indent=2)
        print(f"Saved preprocessing parameters to {f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')
    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)

    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=25, help="Number of retained patient for the validation data")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--process', '-p', type=int, default=1,
                        help="The number of cores to use for processing")

    parser.add_argument('--hu_window', type=float, nargs=2, metavar=('HU_MIN', 'HU_MAX'), default=None,
                        help="Clip the CT to a fixed [HU_MIN, HU_MAX] window before scaling to [0, 255]. "
                             "Default: per-volume min-max normalization.")
    parser.add_argument('--spacing', type=float, nargs="+", metavar='MM', default=None,
                        help="Resample the volumes to this voxel spacing (in mm), then center crop/pad the "
                             "slices to --shape. Give 2 values (x y) to keep the original slices in z, or 3. "
                             "Default: resize each slice to --shape, whatever its spacing.")
    args = parser.parse_args()

    if args.spacing is not None and len(args.spacing) not in [2, 3]:
        parser.error("--spacing takes 2 (x y) or 3 (x y z) values")

    random.seed(args.seed)

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
