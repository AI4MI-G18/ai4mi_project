#!/usr/bin/env python3

"""
Post-processing for predictions. Keeps only the largest connected component for every organ.
"""

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

from utils import tqdm_


def largest_components(seg: np.ndarray, num_classes: int) -> tuple[np.ndarray, dict[int, int]]:
    res: np.ndarray = seg.copy()
    removed: dict[int, int] = {}
    for k in range(1, num_classes):
        mask: np.ndarray = seg == k
        components, n = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
        if n <= 1:
            continue
        sizes: np.ndarray = np.bincount(components.ravel())[1:]  # without the background
        largest: int = int(sizes.argmax()) + 1
        islands: np.ndarray = mask & (components != largest)
        res[islands] = 0
        removed[k] = int(islands.sum())

    return res, removed


def main(args: argparse.Namespace) -> None:
    preds: list[Path] = sorted(args.pred_folder.glob("*.nii.gz"))
    assert preds, f"No .nii.gz found in {args.pred_folder}"
    args.dest_folder.mkdir(parents=True, exist_ok=True)

    for path in tqdm_(preds):
        nib_obj = nib.load(str(path))
        seg: np.ndarray = np.asarray(nib_obj.dataobj)
        res, removed = largest_components(seg, args.num_classes)
        if removed:
            print(f"[postprocess] {path.name}: removed " + ", ".join(f"{v} voxels of class {k}"
                                                                    for k, v in removed.items()))
        nib.save(nib.nifti1.Nifti1Image(res, affine=nib_obj.affine, header=nib_obj.header),
                 args.dest_folder / path.name)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep the largest connected component of every class")
    parser.add_argument('--pred_folder', type=Path, required=True, help="Folder with the stitched Patient_XX.nii.gz")
    parser.add_argument('--dest_folder', type=Path, required=True)
    parser.add_argument('--num_classes', type=int, default=5)

    args = parser.parse_args()
    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
