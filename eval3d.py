#!/usr/bin/env python3

"""
3D Dice and HD95 between stitched predictions and the original ground truth,
computed on the original voxel grid. This is the only place where runs using
different preprocessings can be compared: the 2D DSC logged by main.py is
computed against the preprocessed (e.g. resampled and cropped) slices.

HD95 is in mm, using the voxel spacing of the ground truth (the voxels are not
isotropic, ~1 x 1 x 2.5 mm). It is the max of the two directed 95th percentiles
of the surface distances (prediction -> ground truth, ground truth -> prediction).
When a class is in only one of the two, it is undefined: it is then set to the
diagonal of the volume (the largest possible distance), so missing an organ is
penalized instead of ignored. It is 0 when the class is in neither, and NaN for
the background, which has no meaningful surface.

Each metric is saved as a .npz mapping each patient id to a K array, as
expected for the submission.
"""

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, distance_transform_edt

from utils import tqdm_


def dice_3d(pred: np.ndarray, gt: np.ndarray, K: int) -> np.ndarray:
    assert pred.shape == gt.shape, (pred.shape, gt.shape)

    res: np.ndarray = np.zeros(K, dtype=np.float64)
    for k in range(K):
        p, g = pred == k, gt == k
        denom: int = p.sum() + g.sum()
        # Class absent from both: perfect agreement
        res[k] = 2 * (p & g).sum() / denom if denom > 0 else 1.

    return res


def surface(mask: np.ndarray) -> np.ndarray:
    # Voxels of the mask with at least one (6-connected) neighbour outside of it
    return mask & ~binary_erosion(mask, border_value=0)


def hd95_binary(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float]) -> float:
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    assert pred.any() and gt.any()

    # Both surfaces are within the bounding box of the union, and so is the
    # closest surface point of any voxel of it: cropping to it (with a margin
    # for the erosion) gives the same distances, much faster
    union: np.ndarray = pred | gt
    crop: tuple[slice, ...] = tuple(slice(max(idx.min() - 1, 0), idx.max() + 2)
                                    for idx in np.nonzero(union))
    pred_surf: np.ndarray = surface(pred[crop])
    gt_surf: np.ndarray = surface(gt[crop])

    # Distance (mm) of every voxel to the closest surface voxel of the other mask
    dist_to_gt: np.ndarray = distance_transform_edt(~gt_surf, sampling=spacing)
    dist_to_pred: np.ndarray = distance_transform_edt(~pred_surf, sampling=spacing)

    return float(max(np.percentile(dist_to_gt[pred_surf], 95),
                     np.percentile(dist_to_pred[gt_surf], 95)))


def hd95_3d(pred: np.ndarray, gt: np.ndarray, K: int, spacing: Sequence[float],
            id_: str = "", class_names: Sequence[str] | None = None) -> np.ndarray:
    assert pred.shape == gt.shape, (pred.shape, gt.shape)

    diagonal: float = float(np.linalg.norm(np.array(pred.shape) * np.array(spacing)))

    res: np.ndarray = np.full(K, np.nan, dtype=np.float64)
    for k in range(1, K):
        p, g = pred == k, gt == k
        if p.any() and g.any():
            res[k] = hd95_binary(p, g, spacing)
        elif not p.any() and not g.any():
            res[k] = 0.
        else:
            name: str = class_names[k] if class_names else str(k)
            print(f"[eval3d] Warning: {id_} {name} is only in the {'prediction' if p.any() else 'ground truth'}, "
                  f"HD95 set to the volume diagonal ({diagonal:.1f} mm)")
            res[k] = diagonal

    return res


def main(args: argparse.Namespace) -> None:
    preds: list[Path] = sorted(args.pred_folder.glob("*.nii.gz"))
    assert preds, f"No .nii.gz found in {args.pred_folder}"

    hd95_dest: Path = args.hd95_dest or args.dest.with_name("hd95_val.npz")

    dices: dict[str, np.ndarray] = {}
    hd95s: dict[str, np.ndarray] = {}
    for pred_path in tqdm_(preds):
        id_: str = pred_path.name.removesuffix(".nii.gz")
        pred: np.ndarray = np.asarray(nib.load(str(pred_path)).dataobj)
        gt_nib = nib.load(args.gt_pattern.format(id_=id_))
        gt: np.ndarray = np.asarray(gt_nib.dataobj)
        spacing: tuple[float, ...] = tuple(map(float, gt_nib.header.get_zooms()))

        dices[id_] = dice_3d(pred, gt, args.num_classes)
        hd95s[id_] = hd95_3d(pred, gt, args.num_classes, spacing, id_, args.class_names)

    for dest, metrics in [(args.dest, dices), (hd95_dest, hd95s)]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dest, **metrics)

    for title, dest, metrics, fmt in [("3D DSC", args.dest, dices, ".4f"),
                                      ("HD95 (mm)", hd95_dest, hd95s, ".2f")]:
        stacked: np.ndarray = np.stack(list(metrics.values()))
        print(f"{title} over {len(metrics)} patients, saved to {dest}")
        for k in range(1, args.num_classes):
            name: str = args.class_names[k] if args.class_names else str(k)
            print(f"  {name:>10}: {stacked[:, k].mean():{fmt}} ± {stacked[:, k].std():{fmt}}")
        print(f"  {'mean fg':>10}: {stacked[:, 1:].mean():{fmt}}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='3D Dice and HD95 on the stitched volumes')
    parser.add_argument('--pred_folder', type=Path, required=True,
                        help="Folder with the stitched Patient_XX.nii.gz predictions")
    parser.add_argument('--gt_pattern', type=str, required=True,
                        help="Pattern to the ground truth, e.g. data/train/{id_}/GT.nii.gz")
    parser.add_argument('--dest', type=Path, required=True, help="The .npz file to save the 3D DSC to")
    parser.add_argument('--hd95_dest', type=Path, default=None,
                        help="The .npz file to save the HD95 to. Default: hd95_val.npz next to --dest")
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--class_names', type=str, nargs='*',
                        default=["background", "esophagus", "heart", "trachea", "aorta"])

    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    main(get_args())
