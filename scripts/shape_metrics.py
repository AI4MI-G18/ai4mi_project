#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage, stats


def load(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    img = nib.load(str(path))
    return np.asarray(img.dataobj), tuple(map(float, img.header.get_zooms()))


def volume_mm3(mask: np.ndarray, spacing) -> float:
    return mask.sum() * np.prod(spacing)


def n_components(mask: np.ndarray) -> int:
    return ndimage.label(mask, structure=np.ones((3, 3, 3)))[1]


def max_area_jump(mask: np.ndarray, spacing) -> float:
    areas = mask.sum(axis=(0, 1)) * spacing[0] * spacing[1]
    present = np.nonzero(areas)[0]
    if len(present) < 2:
        return 0.0
    areas = areas[present[0]:present[-1] + 1]
    return np.abs(np.diff(areas)).max() / np.median(areas[areas > 0])


def descriptors(mask: np.ndarray, spacing, normal_volume: float) -> dict[str, float]:
    return {"volume vs normal": abs(volume_mm3(mask, spacing) / normal_volume - 1),
            "components": n_components(mask),
            "max area jump": max_area_jump(mask, spacing)}


def normal_volume(train_dir: Path, exclude: set[str], organ: int) -> float:
    train_ids = [p.name for p in sorted(train_dir.glob("Patient_*")) if p.name not in exclude]
    return np.mean([volume_mm3(gt == organ, sp) for gt, sp in (load(train_dir / i / "GT.nii.gz") for i in train_ids)])


def main(args: argparse.Namespace) -> None:
    per_patient = {}
    for run in args.runs:
        preds = sorted((Path(args.volumes_root) / run).glob("*.nii.gz"))
        val_ids = [p.name.removesuffix(".nii.gz") for p in preds]
        normal = normal_volume(Path(args.train_dir), set(val_ids), args.organ)
        print(f"{run}: {len(val_ids)} patients, normal volume of organ {args.organ} {normal / 1000:.1f} ml")
        dice = np.load(Path(args.results_root) / run / "dice3d_val.npz")
        hd95 = np.load(Path(args.results_root) / run / "hd95_val.npz")
        for path, id_ in zip(preds, val_ids):
            pred, spacing = load(path)
            d = descriptors(pred == args.organ, spacing, normal)
            per_patient.setdefault(id_, []).append({**d, "dice": dice[id_][args.organ], "hd95": hd95[id_][args.organ]})

    ids = sorted(per_patient)
    names = ["volume vs normal", "components", "max area jump", "dice", "hd95"]
    mean = {i: {n: np.mean([r[n] for r in per_patient[i]]) for n in names} for i in ids}

    print(f"\n{len(ids)} patients, each averaged over the runs it appears in")
    print(f"{'patient':<12}{'volume vs normal':>18}{'components':>12}{'max area jump':>15}{'dice':>8}{'hd95':>8}")
    for i in ids:
        m = mean[i]
        print(f"{i:<12}{m['volume vs normal']:>18.2f}{m['components']:>12.1f}{m['max area jump']:>15.2f}{m['dice']:>8.3f}{m['hd95']:>8.1f}")

    print(f"\nSpearman correlation over {len(ids)} patients (rho, p)")
    for name in names[:3]:
        x = [mean[i][name] for i in ids]
        rd = stats.spearmanr(x, [mean[i]["dice"] for i in ids])
        rh = stats.spearmanr(x, [mean[i]["hd95"] for i in ids])
        print(f"{name:<18} vs dice: {rd.statistic:+.2f} (p={rd.pvalue:.3f})   vs hd95: {rh.statistic:+.2f} (p={rh.pvalue:.3f})")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Do shape descriptors of a prediction flag the patients it gets wrong?")
    parser.add_argument("--runs", nargs="+", default=[f"UNet3DSmall_DS-DiceFocal-aug_all-sched_poly-s{s}" for s in range(3)])
    parser.add_argument("--organ", type=int, default=1, help="1 esophagus, 2 heart, 3 trachea, 4 aorta")
    parser.add_argument("--results_root", default="results/preproc/SEGTHOR_PREPROC")
    parser.add_argument("--volumes_root", default="volumes/preproc/SEGTHOR_PREPROC")
    parser.add_argument("--train_dir", default="data/train")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())