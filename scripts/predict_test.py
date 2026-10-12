#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
from skimage.transform import resize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from UNet import UNet3DSmall_DS
from utils import sliding_window_z
from postprocess import largest_components
from preprocessing import complete_spacing, crop_or_pad, resample, resampled_shape, window_hu

K = 5
RUN = "UNet3DSmall_DS-DiceFocal-aug_all-sched_poly-s0"
FOLDS = [f"results/preproc/{d}/{RUN}" for d in
         ["SEGTHOR_PREPROC", "SEGTHOR_PREPROC_F1", "SEGTHOR_PREPROC_F2", "SEGTHOR_PREPROC_F3"]]


def load_nets(run_dirs: list[str], device: torch.device) -> torch.nn.ModuleList:
    nets = torch.nn.ModuleList()
    for run_dir in run_dirs:
        net = UNet3DSmall_DS(1, K)
        net.load_state_dict(torch.load(Path(run_dir) / "bestweights.pt", map_location=device))
        nets.append(net.to(device).eval())
    return nets


def preprocess(ct: np.ndarray, spacing: tuple[float, ...], args: argparse.Namespace) -> np.ndarray:
    resampled = resample(ct, spacing, complete_spacing(args.spacing, spacing), is_label=False)
    return crop_or_pad(window_hu(resampled, *args.hu_window), args.shape)


def predict(nets: torch.nn.ModuleList, volume: np.ndarray, depth: int, device: torch.device) -> np.ndarray:
    img = torch.tensor(volume / 255, dtype=torch.float32, device=device)[None, None]
    with torch.no_grad():
        probs = sum(sliding_window_z(net, img, depth) for net in nets) / len(nets)
    return probs[0].argmax(dim=0).cpu().numpy().astype(np.uint8)


def unpreprocess(seg: np.ndarray, ct_shape: tuple[int, ...], spacing: tuple[float, ...],
                 args: argparse.Namespace) -> np.ndarray:
    shape = resampled_shape(ct_shape, spacing, complete_spacing(args.spacing, spacing))
    uncropped = crop_or_pad(seg, shape[:2])
    return resize(uncropped, ct_shape, order=0, mode="edge", preserve_range=True,
                  anti_aliasing=False).astype(np.int16)


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    nets = load_nets(args.runs, device)
    print(f"{len(nets)} models from {args.runs}")
    if args.save_models:
        torch.save(nets.cpu(), args.save_models)
        nets.to(device)
        print(f"saved the ensemble to {args.save_models}")

    args.dest.mkdir(parents=True, exist_ok=True)
    scans = sorted(Path().glob(args.scans))
    assert scans, args.scans
    for path in scans:
        id_ = path.name.removesuffix(".nii.gz")
        ct_nib = nib.load(str(path))
        ct = np.asarray(ct_nib.dataobj)
        spacing = tuple(map(float, ct_nib.header.get_zooms()))

        seg = predict(nets, preprocess(ct, spacing, args), args.depth, device)
        seg = unpreprocess(seg, ct.shape, spacing, args)
        if not args.no_lcc:
            seg, _ = largest_components(seg, K)

        missing = set(range(K)) - set(np.unique(seg))
        print(f"{id_}: {ct.shape}" + (f", no voxels of class(es) {missing}" if missing else ""))
        out = nib.nifti1.Nifti1Image(seg.astype(np.uint8), affine=ct_nib.affine)
        out.set_qform(ct_nib.affine, code=int(ct_nib.header["qform_code"]))
        out.set_sform(ct_nib.affine, code=int(ct_nib.header["sform_code"]))
        nib.save(out, args.dest / f"{id_}.nii.gz")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average the softmax of the fold models on whole scans")
    parser.add_argument("--scans", default="data/test/Patient_*.nii.gz", help="glob of the CT scans")
    parser.add_argument("--runs", nargs="+", default=FOLDS, help="run folders with a bestweights.pt")
    parser.add_argument("--dest", type=Path, default=Path("volumes/test/pred"))
    parser.add_argument("--save_models", type=Path, default=None, help="also torch.save the ensemble here")
    parser.add_argument("--no_lcc", action="store_true", help="skip keeping the largest component")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--depth", type=int, default=64, help="slices per window, as in validation")
    parser.add_argument("--hu_window", type=float, nargs=2, default=[-500, 500])
    parser.add_argument("--spacing", type=float, nargs=2, default=[1.5, 1.5])
    parser.add_argument("--shape", type=int, nargs=2, default=[256, 256])
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
