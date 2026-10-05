#!/usr/bin/env python3

# MIT License

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

"""
U-Net (Ronneberger et al., 2015), in the variants main.py can train:

* UNet:    2D, one slice in, its segmentation out.
* UNet25D: the same 2D network, fed the slice and its neighbours in z as input
           channels. It still predicts the centre slice only, so everything
           downstream of main.py (stitch.py, eval3d.py) is left untouched.
* UNet3D:  3D convolutions, trained on patches of the volumes (3D U-Net, Cicek
           et al., 2016).

main.py reads the `context` and `ndim` class attributes to know what data a
network wants.
"""

from typing import Any
from functools import partial

import torch
import torch.nn as nn
from torch import Tensor


# The layers of the 2D and of the 3D network. InstanceNorm in 3D: the batches are too
# small (2 patches) for the statistics of BatchNorm
layers: dict[int, dict[str, Any]] = {
    2: {'conv': nn.Conv2d, 'norm': nn.BatchNorm2d, 'pool': nn.MaxPool2d, 'up': nn.ConvTranspose2d},
    3: {'conv': nn.Conv3d, 'norm': partial(nn.InstanceNorm3d, affine=True), 'pool': nn.MaxPool3d,
        'up': nn.ConvTranspose3d},
}


def double_conv(in_dim: int, out_dim: int, ndim: int) -> nn.Sequential:
    conv, norm = layers[ndim]['conv'], layers[ndim]['norm']

    return nn.Sequential(
        conv(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
        norm(out_dim),
        nn.ReLU(inplace=True),
        conv(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
        norm(out_dim),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    ndim: int = 2  # 2D convolutions on slices, or 3D ones on volumes
    context: int = 0  # Neighbouring slices taken on each side, as extra input channels
    depth: int = 4  # Number of downsamplings
    width: int = 32  # Kernels of the first level, doubled at each downsampling
    deep_supervision: bool = False  # Also outputs at 1/2 and 1/4 resolution, in training

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        # **kwargs discards the keyword arguments of ENet (kernels, factor)
        super().__init__()

        widths: list[int] = [self.width * 2**i for i in range(self.depth + 1)]

        self.encoders = nn.ModuleList()
        prev: int = in_dim * (2 * self.context + 1)
        for w in widths[:-1]:
            self.encoders.append(double_conv(prev, w, self.ndim))
            prev = w
        self.pool = layers[self.ndim]['pool'](2)
        self.bottleneck = double_conv(prev, widths[-1], self.ndim)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = widths[-1]
        for w in reversed(widths[:-1]):
            self.ups.append(layers[self.ndim]['up'](prev, w, kernel_size=2, stride=2))
            self.decoders.append(double_conv(2 * w, w, self.ndim))  # 2 * w: the skip connection is concatenated
            prev = w

        self.final = layers[self.ndim]['conv'](prev, out_dim, kernel_size=1)
        if self.deep_supervision:
            # On the outputs of the decoders at 1/2 and 1/4 resolution
            self.aux_heads = nn.ModuleList([layers[self.ndim]['conv'](w, out_dim, kernel_size=1)
                                            for w in widths[1:3]])

        print(f"Initialized {self.__class__.__name__} succesfully")

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, list[Tensor]]:
        assert all(s % 2**self.depth == 0 for s in input.shape[2:]), (input.shape, self.depth)

        skips: list[Tensor] = []
        x = input
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        decoded: list[Tensor] = []
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = decoder(torch.cat([up(x), skip], dim=1))
            decoded.append(x)

        # The auxiliary outputs only in training: validation and the sliding window get the main one
        if not (self.deep_supervision and self.training):
            return self.final(x)

        # decoded[-2] is at 1/2 resolution, decoded[-3] at 1/4
        return self.final(x), [head(d) for head, d in zip(self.aux_heads, decoded[-2::-1])]

    def init_weights(self, *args, **kwargs):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class UNet25D(UNet):
    context: int = 2  # 5 slices: with 2 to 3.7 mm between slices, that is 1 to 1.5 cm of context in z


class UNet3D(UNet):
    ndim: int = 3
    depth: int = 3  # 64 slices in a patch: 8 are left after 3 downsamplings


class UNet_DS(UNet):
    deep_supervision: bool = True


class UNet25D_DS(UNet25D):
    deep_supervision: bool = True


class UNet3D_DS(UNet3D):
    deep_supervision: bool = True
