#!/usr/bin/env python3
"""Merge the losses of every experiment found under a folder.

Two figures are written in <root>:
    - loss_among_experiments.png: one subplot per experiment (same curves as
      plot_losses.py), all subplots sharing the same y range. That y axis is
      logarithmic: the experiments use different loss functions, whose values
      differ by orders of magnitude;
    - loss_among_experiments_original_axis.png: same, but each subplot keeps
      the y axis of its own loss.png (linear, fitted to its curves).
"""

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


MODES = {
    "tra": "Training",
    "val": "Validation",
}


def label(path: Path, root: Path) -> tuple[str, str]:
    # (<phase>, <dataset>/<experiment>)
    rel = path.relative_to(root)
    return rel.parts[0], '/'.join(rel.parts[-3:-1])


def load(exp: Path, mode: str) -> np.ndarray:
    return np.load(exp / f"loss_{mode}.npy").mean(axis=1)  # E x B -> E


def find_experiments(root: Path) -> list[Path]:
    exps: list[Path] = sorted(p.parent for p in root.glob("*/results/**/loss_val.npy"))
    assert exps, f"No loss_val.npy found under {root}"
    return exps


def merge_grid(args: argparse.Namespace, sharey: bool) -> None:
    # sharey: same (log) y range for all subplots; otherwise each subplot keeps
    # the y axis of its own loss.png (linear, fitted to its curves)
    exps: list[Path] = find_experiments(args.root)

    cols: int = args.cols
    rows: int = math.ceil(len(exps) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 5.0 * rows),
                             sharey=sharey, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, exp in zip(axes.flat, exps):
        ax.axis("on")
        for mode, name in MODES.items():
            y = load(exp, mode)
            ax.plot(np.arange(len(y)), y, label=name, linewidth=2)

        phase, name = label(exp / "loss_val.npy", args.root)
        ax.set_title(f"{phase}\n{name}", fontsize=14)
        if sharey:
            ax.set_yscale("log")
            ax.tick_params(labelleft=True)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

    fig.tight_layout()
    suffix: str = "" if sharey else "_original_axis"
    dest: Path = args.root / f"loss_among_experiments{suffix}.png"
    fig.savefig(dest, dpi=100)
    print(f"Merged {len(exps)} plots into {dest}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the losses of all experiments")
    parser.add_argument("root", type=Path, help="Folder containing the <phase>/results/ folders")
    parser.add_argument("--cols", type=int, default=4,
                        help="Number of columns of the grid figure")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    merge_grid(args, sharey=True)
    merge_grid(args, sharey=False)
