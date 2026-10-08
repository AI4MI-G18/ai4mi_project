#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import os
import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from functools import partial

from dataset import SliceDataset, VolumeDataset
from augment import AUGMENTATIONS
from ShallowNet import shallowCNN
from ENet import ENet
from ENetOurs import (ENetImproved, ENetImproved25D, ENetImproved3D,
                      ENetImproved_DS, ENetImproved25D_DS, ENetImproved3D_DS)
from UNet import (UNet, UNet25D, UNet3D, UNet_DS, UNet25D_DS, UNet3D_DS,
                  UNetSmall_DS, UNet25DSmall_DS, UNet3DSmall_DS)
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images,
                   sliding_window_z)

from losses import (CrossEntropy, TopKCrossEntropy, FocalLoss,
                    SoftDiceLoss, DiceCELoss, DiceTopKLoss, DiceFocalLoss)

losses = {
    'CE': CrossEntropy,
    'TopK': TopKCrossEntropy,
    'Focal': FocalLoss,
    'Dice': SoftDiceLoss,
    'DiceCE': DiceCELoss,
    'DiceTopK': DiceTopKLoss,
    'DiceFocal': DiceFocalLoss
}

# the networks --arch can name
architectures = {
    'ENet': ENet,
    'ENetImproved': ENetImproved,
    'ENetImproved25D': ENetImproved25D,
    'ENetImproved3D': ENetImproved3D,
    'ENetImproved_DS': ENetImproved_DS,
    'ENetImproved25D_DS': ENetImproved25D_DS,
    'ENetImproved3D_DS': ENetImproved3D_DS,
    'shallowCNN': shallowCNN,
    'UNet': UNet,
    'UNet25D': UNet25D,
    'UNet3D': UNet3D,
    'UNet_DS': UNet_DS,
    'UNet25D_DS': UNet25D_DS,
    'UNet3D_DS': UNet3D_DS,
    'UNetSmall_DS': UNetSmall_DS,
    'UNet25DSmall_DS': UNet25DSmall_DS,
    'UNet3DSmall_DS': UNet3DSmall_DS,
}

# Deep supervision: the weights of the auxiliary losses, finest output first (the main loss weighs 1)
DEEP_SUPERVISION_WEIGHTS: list[float] = [0.5, 0.25]

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
# Preprocessing ablation, see the Makefile: HU windowing only, resampling only, both
datasets_params["SEGTHOR_HU"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_RESAMPLE"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_PREPROC"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
# Segthor with resampled Z
datasets_params["SEGTHOR_PREPROC_Z"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["TOY2_OURS"] = {'K': 2, 'net': ENetImproved, 'B': 2, 'kernels': 8, 'factor': 2, 'root': 'TOY'}
datasets_params["SEGTHOR_OURS"] = {'K': 5, 'net': ENetImproved, 'B': 8, 'kernels': 8, 'factor': 2, 'root': 'SEGTHOR'}


def img_transform(img):
    img = img.convert('L')
    img = np.array(img)[np.newaxis, ...]
    img = img / 255  # max <= 1
    img = torch.tensor(img, dtype=torch.float32)
    return img


def gt_transform(K, img):
    img = np.array(img)[...]
    # The idea is that the classes are mapped to {0, 255} for binary cases
    # {0, 85, 170, 255} for 4 classes
    # {0, 51, 102, 153, 204, 255} for 6 classes
    # Very sketchy but that works here and that simplifies visualization
    img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
    img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
    img = class2one_hot(img, K=K)
    return img[0]


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    mps: bool = args.mps and torch.backends.mps.is_available()
    device = torch.device("cuda") if gpu else torch.device('mps') if mps else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    # --arch picks the network; without it the dataset's own default applies
    net_class = architectures[args.arch] if args.arch else datasets_params[args.dataset]['net']
    net = net_class(1, K, **{k: v for k, v in datasets_params[args.dataset].items()
                             if k not in ('K', 'net', 'B', 'root')})
    net.init_weights()
    net.to(device)

    # Slices before and after the one to segment, that the network wants as input channels (2.5D)
    context: int = getattr(net_class, 'context', 0)

    lr = 0.0005 if args.lr is None else args.lr
    betas = (0.9, 0.999) if args.betas is None else tuple(args.betas)
    weight_decay = 1e-4 if args.weight_decay is None else args.weight_decay
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=betas) if args.opt == 'Adam' else torch.optim.AdamW(
        net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    print(
        f">> Using {args.opt} optimizer with lr={lr}, betas={betas}, weight_decay={weight_decay if args.opt == 'AdamW' else 'None'}")

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path(os.environ.get("DATA_ROOT", "data")) / datasets_params[args.dataset].get('root', args.dataset)

    train_set: SliceDataset | VolumeDataset
    val_set: SliceDataset | VolumeDataset
    B_val: int = B
    if getattr(net_class, 'ndim', 2) == 3:
        # Patches of the volumes for training, and the whole volumes for validation:
        # one at a time, as they do not have the same depth
        B, B_val = 2, 1
        if args.patch is None:  # The network's own patch size, if it has one
            args.patch = list(getattr(net_class, 'patch', (128, 128, 64)))
        train_set = VolumeDataset('train',
                                  root_dir,
                                  img_transform=img_transform,
                                  gt_transform=partial(gt_transform, K),
                                  patch=tuple(args.patch),
                                  samples_per_volume=args.samples_per_volume,
                                  debug=args.debug,
                                  augment=AUGMENTATIONS[args.aug])
        val_set = VolumeDataset('val',
                                root_dir,
                                img_transform=img_transform,
                                gt_transform=partial(gt_transform, K),
                                patch=tuple(args.patch),
                                debug=args.debug)
    else:
        train_set = SliceDataset('train',
                                 root_dir,
                                 img_transform=img_transform,
                                 gt_transform=partial(gt_transform, K),
                                 debug=args.debug,
                                 context=context,
                                 augment=AUGMENTATIONS[args.aug])
        val_set = SliceDataset('val',
                               root_dir,
                               img_transform=img_transform,
                               gt_transform=partial(gt_transform, K),
                               debug=args.debug,
                               context=context)

    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              shuffle=True)
    val_loader = DataLoader(val_set,
                            batch_size=B_val,
                            num_workers=5,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def make_scheduler(optimizer, kind: str, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    schedules = {'none': lambda t: 1.0,
                 'cosine': lambda t: 0.5 * (1 + math.cos(math.pi * t / total_steps)),
                 'poly': lambda t: (1 - t / total_steps) ** 0.9}
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedules[kind])


def set_seed(seed: int):
    import random, numpy, torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def runTraining(args):
    set_seed(args.seed)
    print(f">>> Setting up to train on {args.dataset} with {args.mode} and {args.loss}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.mode == "full":
        idk = list(range(K))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        idk = [0, 1, 3, 4]  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    loss_fn = losses[args.loss](idk=idk)
    scheduler = make_scheduler(optimizer, args.scheduler, args.epochs * len(train_loader))

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0

    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B = img.shape[0]
                    volumes: bool = img.ndim == 5  # (B, 1, W, H, D), from the VolumeDataset

                    aux_logits: list[Tensor] = []  # Deep supervision: the coarser outputs, finest first
                    if volumes and m == 'val':
                        # A whole volume, for a network trained on patches of it
                        pred_probs = sliding_window_z(net, img, args.patch[2])
                    else:
                        pred_logits = net(img)
                        if isinstance(pred_logits, tuple):  # Deep supervision, in training only
                            pred_logits, aux_logits = pred_logits
                        pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(gt, pred_seg)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    for weight, aux in zip(DEEP_SUPERVISION_WEIGHTS, aux_logits):
                        # Nearest neighbour keeps the downsampled ground truth one-hot
                        aux_gt: Tensor = F.interpolate(gt.float(), size=aux.shape[2:], mode='nearest').type(gt.dtype)
                        loss = loss + weight * loss_fn(F.softmax(aux, dim=1), aux_gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()
                        scheduler.step()  # Per optimizer step

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            stems: list[str] = data['stems']
                            if volumes:
                                # One image per slice, named as the 2D networks do: stitch.py
                                # then does not have to know where they come from
                                assert B == 1
                                D: int = predicted_class.shape[-1]
                                predicted_class = predicted_class[0].permute(2, 0, 1)
                                stems = [f"{stems[0]}_{z:04d}" for z in range(D)]
                            save_images(predicted_class * mult,
                                        stems,
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        if args.scheduler != 'none':
            print(f">> Learning rate after epoch {e}: {scheduler.get_last_lr()[0]:.2e}")

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        if current_dice > best_dice:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--loss', default='CE', choices=losses.keys())
    parser.add_argument('--arch', default=None, choices=architectures.keys(),
                        help="Network to train. Default: the dataset's own choice.")
    parser.add_argument('--patch', type=int, nargs=3, default=None, metavar=('W', 'H', 'D'),
                        help="3D networks only: the size of the training patches. The validation "
                             "slides a window of D slices over the whole volumes. "
                             "Default: the network's own (ENetImproved3D: 256 256 32), else 128 128 64.")
    parser.add_argument('--samples_per_volume', type=int, default=32,
                        help="3D networks only: the number of patches per scan in an epoch.")
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--mps', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    parser.add_argument('--opt', default='Adam', choices=['Adam', 'AdamW'],
                        help="Optimizer to use. Default: Adam.")
    parser.add_argument('--lr', type=float, help="Learning rate. Default: 0.0005.")
    parser.add_argument('--betas', nargs=2, type=float, help="Beta values for the optimizer. Default: (0.9, 0.999).")
    parser.add_argument('--weight_decay', type=float, help="Weight decay for the optimizer. Default: 1e-4.")
    parser.add_argument('--scheduler', default='none', choices=['none', 'cosine', 'poly'],
                        help="Learning rate schedule, per optimizer step: constant (none), cosine "
                             "annealing to 0, or poly (1 - t/T)^0.9. Default: none.")
    parser.add_argument('--aug', default='none', choices=AUGMENTATIONS.keys(),
                        help="Training augmentations, see augment.py. Default: none.")

    args = parser.parse_args()

    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
