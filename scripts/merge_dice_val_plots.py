#!/usr/bin/env python3
"""Merge the validation dice of every experiment found under a folder.

Three figures are written in <root>:
    - dice_val_among_experiments.png: one subplot per experiment (same curves
      as plot.py), all subplots sharing the same y range;
    - dice_val_among_experiments_shared_axis.png: one curve per experiment
      ("All classes" mean, as in plot.py), all drawn on the same axis;
    - dice_val_among_experiments_shared_axis_k1.png: same, for the class k=1 only.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def label(path: Path, root: Path) -> tuple[str, str]:
    # (<phase>, <dataset>/<experiment>)
    rel = path.relative_to(root)
    return rel.parts[0], '/'.join(rel.parts[-3:-1])


def merge_grid(args: argparse.Namespace) -> None:
    npys: list[Path] = sorted(args.root.glob("*/results/**/dice_val.npy"))
    assert npys, f"No dice_val.npy found under {args.root}"

    cols: int = args.cols
    rows: int = math.ceil(len(npys) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 5.0 * rows),
                             sharey=True, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    # Same curves as plot.py, but re-drawn so that all subplots share the y range
    for ax, npy in zip(axes.flat, npys):
        metrics: np.ndarray = np.load(npy)  # E x N x K
        E, _, K = metrics.shape
        epcs = np.arange(E)

        ax.axis("on")
        for k in range(1, K):
            ax.plot(epcs, metrics[:, :, k].mean(axis=1), label=f"{k=}", linewidth=1.5)
        ax.plot(epcs, metrics.mean(axis=1).mean(axis=1), label="All classes", linewidth=3)

        phase, exp = label(npy, args.root)
        ax.set_title(f"{phase}\n{exp}", fontsize=14)
        ax.set_ylim(0, 1)
        ax.tick_params(labelleft=True)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")

    fig.tight_layout()
    dest: Path = args.root / "dice_val_among_experiments.png"
    fig.savefig(dest, dpi=100)
    print(f"Merged {len(npys)} plots into {dest}")


def merge_shared_axis(args: argparse.Namespace, k: int | None = None) -> None:
    # k: class to plot; None for the mean over all classes
    npys: list[Path] = sorted(args.root.glob("*/results/**/dice_val.npy"))
    assert npys, f"No dice_val.npy found under {args.root}"

    fig = plt.figure(figsize=(14, 8))
    ax = fig.gca()
    what: str = "mean over all classes" if k is None else f"{k=}"
    ax.set_title(f"Validation dice ({what}) among experiments")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")

    colors = plt.get_cmap("tab20").colors
    phases: list[str] = sorted({label(npy, args.root)[0] for npy in npys})
    styles: list[str] = ["-", "--", ":", "-."]

    for i, npy in enumerate(npys):
        metrics: np.ndarray = np.load(npy)  # E x N x K
        if k is None:
            y = metrics.reshape(metrics.shape[0], -1).mean(axis=1)
        else:
            y = metrics[:, :, k].mean(axis=1)
        phase, exp = label(npy, args.root)
        ax.plot(np.arange(len(y)), y, label=f"{phase} | {exp}", linewidth=2,
                color=colors[i % len(colors)],
                linestyle=styles[phases.index(phase) % len(styles)])

    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)

    fig.tight_layout()
    suffix: str = "" if k is None else f"_k{k}"
    dest: Path = args.root / f"dice_val_among_experiments_shared_axis{suffix}.png"
    fig.savefig(dest, dpi=100)
    print(f"Merged {len(npys)} curves into {dest}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the validation dice of all experiments")
    parser.add_argument("root", type=Path, help="Folder containing the <phase>/results/ folders")
    parser.add_argument("--cols", type=int, default=4,
                        help="Number of columns of the grid figure")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    merge_grid(args)
    merge_shared_axis(args)
    merge_shared_axis(args, k=1)
