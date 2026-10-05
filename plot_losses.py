#!/usr/bin/env python3

"""Plot the training and validation loss curves of one experiment."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOSS_FILES = {
    "Training": "loss_tra.npy",
    "Validation": "loss_val.npy",
}


def load_losses(results_dir: Path) -> dict[str, np.ndarray]:
    """Load and average the batch losses for every epoch."""
    losses = {}
    missing = []

    for label, filename in LOSS_FILES.items():
        loss_file = results_dir / filename
        if not loss_file.is_file():
            missing.append(str(loss_file))
            continue

        values = np.asarray(np.load(loss_file), dtype=float)
        if values.ndim != 2:
            raise ValueError(f"Expected {loss_file} to have shape (epochs, batches), got {values.shape}")
        losses[label] = values.mean(axis=1)

    if missing:
        missing_files = "\n".join(f"  - {file}" for file in missing)
        raise FileNotFoundError(f"Missing loss files:\n{missing_files}")

    return losses


def plot_losses(losses: dict[str, np.ndarray], title: str, destination: Path | None,
                show: bool = False) -> None:
    figure, axis = plt.subplots()
    for label, values in losses.items():
        axis.plot(np.arange(len(values)), values, linewidth=2, label=label)

    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()

    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination)
    if show:
        plt.show()
    plt.close(figure)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the losses of one experiment.")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Experiment folder, containing loss_tra.npy and loss_val.npy.")
    parser.add_argument("--dest", type=Path, default=None,
                        help="Output image path (default: <results-dir>/loss.png).")
    parser.add_argument("--show", action="store_true", help="Display the plot after saving it.")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    losses = load_losses(args.results_dir)
    destination = args.dest if args.dest else args.results_dir / "loss.png"
    plot_losses(losses, str(args.results_dir), destination, show=args.show)


if __name__ == "__main__":
    main()
