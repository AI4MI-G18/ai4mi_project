#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

ORGANS = ["esophagus", "heart", "trachea", "aorta"]


def load_folds(results_root: str, dataset: str, run: str) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    folds = []
    for d in [dataset] + [f"{dataset}_F{f}" for f in (1, 2, 3)]:
        dice = np.load(Path(results_root) / d / run / "dice3d_val.npz")
        hd95 = np.load(Path(results_root) / d / run / "hd95_val.npz")
        folds.append({i: (dice[i][1:], hd95[i][1:]) for i in dice.files})
    return folds


def report(name: str, folds: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    patients = {i: v for fold in folds for i, v in fold.items()}
    assert len(patients) == sum(len(f) for f in folds), "a patient is in multiple validation folds"

    print(f"\n{name}: {len(folds)} folds, {len(patients)} patients")
    print(f"{'':<8}" + "".join(f"{o:>11}" for o in ORGANS) + f"{'mean':>11}{'fold std':>11}")
    for m, metric in enumerate(["dice", "hd95"]):
        per_patient = np.array([v[m] for v in patients.values()])
        per_fold = [np.mean([v[m].mean() for v in f.values()]) for f in folds]
        print(f"{metric:<8}" + "".join(f"{x:>11.3f}" for x in per_patient.mean(axis=0))
              + f"{per_patient.mean():>11.3f}{np.std(per_fold):>11.3f}")
    return patients


def main(args: argparse.Namespace) -> None:
    base = report("baseline " + "/".join(args.baseline), load_folds(args.results_root, *args.baseline))
    ours = report("ours " + "/".join(args.ours), load_folds(args.results_root, *args.ours))

    ids = sorted(base)
    assert ids == sorted(ours)
    print(f"\nPaired Wilcoxon signed-rank, ours vs baseline, over {len(ids)} patients")
    for m, metric in enumerate(["dice", "hd95"]):
        b = np.array([base[i][m].mean() for i in ids])
        o = np.array([ours[i][m].mean() for i in ids])
        res = stats.wilcoxon(o, b)
        better = (o > b).sum() if metric == "dice" else (o < b).sum()
        print(f"{metric:<6} median difference {np.median(o - b):+.3f}, ours better in {better}/{len(ids)}, p={res.pvalue:.2e}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validation table and paired test, ours vs baseline")
    parser.add_argument("--baseline", nargs=2, default=["SEGTHOR", "ENet-CE-s0"], metavar=("DATASET", "RUN"))
    parser.add_argument("--ours", nargs=2, metavar=("DATASET", "RUN"),
                        default=["SEGTHOR_PREPROC", "UNet3DSmall_DS-DiceFocal-aug_all-sched_poly-s0-lcc"])
    parser.add_argument("--results_root", default="results/preproc")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
