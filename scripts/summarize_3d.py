#!/usr/bin/env python3

"""
Tables of the 3D DSC and HD95 (mean ± std (median) over the validation patients) of
every run evaluated by eval3d.py, found as <root>/<dataset>/<loss>/dice3d_val.npz
(and hd95_val.npz next to it). The median is less sensitive than the mean to a
single failed patient, which matters most for HD95. "mean fg" averages over all
organs and patients, "median fg" averages the per-organ medians.

    python scripts/summarize_3d.py --root results/preproc
"""

import argparse
from pathlib import Path

import numpy as np


def print_table(title: str, files: list[Path], class_names: list[str], fmt: str) -> None:
    header: list[str] = ["dataset", "loss", "n", *class_names[1:], "mean fg", "median fg"]
    rows: list[list[str]] = []
    for f in files:
        metrics = np.load(f)
        stacked: np.ndarray = np.stack([metrics[k] for k in metrics.files])  # patients x K
        medians: np.ndarray = np.median(stacked, axis=0)
        per_class: list[str] = [f"{stacked[:, k].mean():{fmt}} ± {stacked[:, k].std():{fmt}} ({medians[k]:{fmt}})"
                                for k in range(1, stacked.shape[1])]
        rows.append([f.parent.parent.name, f.parent.name, str(len(stacked)),
                     *per_class, f"{stacked[:, 1:].mean():{fmt}}", f"{medians[1:].mean():{fmt}}"])

    widths: list[int] = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    print(f"{title}, mean ± std (median) over the patients")
    for r in [header, ["-" * w for w in widths], *rows]:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def main(args: argparse.Namespace) -> None:
    for title, name, fmt in [("3D DSC (higher is better)", "dice3d_val.npz", ".4f"),
                             ("HD95 in mm (lower is better)", "hd95_val.npz", ".2f")]:
        files: list[Path] = sorted(args.root.glob(f"*/*/{name}"))
        if not files:
            print(f"No */*/{name} found in {args.root}\n")
            continue
        print_table(title, files, args.class_names, fmt)
        print()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize the 3D DSC and HD95 of several runs')
    parser.add_argument('--root', type=Path, default=Path("results/preproc"))
    parser.add_argument('--class_names', type=str, nargs='*',
                        default=["background", "esophagus", "heart", "trachea", "aorta"])

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
