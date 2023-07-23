from __future__ import annotations

import functools
from typing import Iterable, Optional

import torch
import torch.nn as nn

from todels import ConvBlock, MLP

class VGG_BLOCK(nn.Module):
    def __init__(self,
                 out_channel: int,
                 num_layers: int,
                 bn: Optional[str]=None,
                 device: Optional[torch.device | str]=None):
        """
        Parameters:
        ----------
            out_channel: int
                The size of out channels for every layer
            num_layers: int,
                The number conv layers
            bn: str
                If want to do batch normalize after each conv layer
                pass the type of bn
            device: Optional[torch.device | str] = None
                Target device which you would run on
        """
        super(self.__class__, self).__init__()
        kernel_sizes = []
        for _ in range(num_layers):
            kernel_sizes.append(ConvBlock(out_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    activation="ReLU",
                                    type_norm=bn,
                                    device=device))
        kernel_sizes.append(nn.MaxPool2d(2, stride=2))
        self.block = nn.Sequential(*kernel_sizes)

    def forward(self, x):
        return self.block(x)


class _VGG(nn.Module):
    """Base VGG class implementation"""
    def __init__(self,
                 num_classes: int,
                 num_layers: Iterable[int],
                 has_bn: bool=False,
                 device: Optional[torch.device | str] = None):
        """
        Parameters:
        ----------
            num_classes: int
                The number class classification
            num_layers: Iterable[int],
                The number conv layers in each block
            bn: bool
                If want to do batch normalize or not
            device: Optional[torch.device | str] = None
                Target device which you would run on
        """
        super(self.__class__, self).__init__()
        assert len(num_layers) == 5
        bn = None
        if has_bn:
            bn = "batch"
        self.block1 = VGG_BLOCK(64, num_layers[0], bn, device=device)
        self.block2 = VGG_BLOCK(128, num_layers[1], bn, device=device)
        self.block3 = VGG_BLOCK(256, num_layers[2], bn, device=device)
        self.block4 = VGG_BLOCK(512, num_layers[3], bn, device=device)
        self.block5 = VGG_BLOCK(512, num_layers[4], bn, device=device)
        self.fc = MLP([4096, 4096, num_classes],
                      layers_activation_function="ReLU",
                      final_activation_function="Softmax",
                      device=device)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.fc(out)
        return out


VGG11 = functools.partial(_VGG, num_layers=[1, 1, 2, 2, 2])
VGG13 = functools.partial(_VGG, num_layers=[2, 2, 2, 2, 2])
VGG16 = functools.partial(_VGG, num_layers=[2, 2, 3, 3, 3])
VGG19 = functools.partial(_VGG, num_layers=[2, 2, 4, 4, 4])

__all__ = (
    "VGG11",
    "VGG13",
    "VGG16",
    "VGG19",
)

# TODO: implement VGG-LRN and VGG type C