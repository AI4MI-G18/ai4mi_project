#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage, stats


def load(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    img = nib.load(str(path))
    return np.asarray(img.dataobj), tuple(map(float, img.header.get_zooms()))


def crop(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    box = ndimage.find_objects((gt | pred).astype(np.uint8))[0]
    box = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in box)
    return np.pad(gt[box], 1), np.pad(pred[box], 1)


def error_stats(gt: np.ndarray, pred: np.ndarray, spacing, rim: float) -> dict[str, float]:
    gt, pred = crop(gt, pred)
    fn = gt & ~pred
    fp = pred & ~gt
    dist_to_surface = np.where(gt, ndimage.distance_transform_edt(gt, sampling=spacing),
                               ndimage.distance_transform_edt(~gt, sampling=spacing))
    errors = fn | fp
    empty_slices = ~pred.any(axis=(0, 1))
    return {"volume ratio": pred.sum() / gt.sum(),
            "fn share": fn.sum() / errors.sum(),
            "rim share": (errors & (dist_to_surface <= rim)).sum() / errors.sum(),
            "missed slice share": fn[:, :, empty_slices].sum() / fn.sum()}


def main(args: argparse.Namespace) -> None:
    per_patient = {}
    for run in args.runs:
        for path in sorted((Path(args.volumes_root) / run).glob("*.nii.gz")):
            id_ = path.name.removesuffix(".nii.gz")
            pred, _ = load(path)
            gt, spacing = load(Path(args.train_dir) / id_ / "GT.nii.gz")
            per_patient.setdefault(id_, []).append(error_stats(gt == args.organ, pred == args.organ, spacing, args.rim))

    ids = sorted(per_patient)
    names = list(per_patient[ids[0]][0])
    mean = {i: {n: np.mean([r[n] for r in per_patient[i]]) for n in names} for i in ids}

    print(f"organ {args.organ}, {len(ids)} patients, each averaged over the runs it appears in, "
          f"rim = errors within {args.rim} mm of the ground-truth surface")
    print(f"\n{'patient':<12}" + "".join(f"{n:>20}" for n in names))
    for i in ids:
        print(f"{i:<12}" + "".join(f"{mean[i][n]:>20.2f}" for n in names))
    print(f"{'median':<12}" + "".join(f"{np.median([mean[i][n] for i in ids]):>20.2f}" for n in names))

    print(f"\nTwo-sided Wilcoxon signed-rank over {len(ids)} patients")
    for name, null in [("volume ratio", 1.0), ("fn share", 0.5), ("rim share", 0.5)]:
        x = np.array([mean[i][name] for i in ids])
        res = stats.wilcoxon(x - null)
        print(f"{name:<14} vs {null}: median {np.median(x):.2f}, above {null} in {(x > null).sum()}/{len(x)}, p={res.pvalue:.4f}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Where are the errors: missing or extra voxels, on the rim or deep, at the ends?")
    parser.add_argument("--runs", nargs="+", default=[f"UNet3DSmall_DS-DiceFocal-aug_all-sched_poly-s{s}-lcc" for s in range(3)])
    parser.add_argument("--organ", type=int, default=1, help="1 esophagus, 2 heart, 3 trachea, 4 aorta")
    parser.add_argument("--rim", type=float, default=3.0, help="mm")
    parser.add_argument("--volumes_root", default="volumes/preproc/SEGTHOR_PREPROC")
    parser.add_argument("--train_dir", default="data/train")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
