from __future__ import annotations

from typing import Iterable, Optional, Union

import torch
import torch.nn as nn

from todels.residual import ResnetShortcut
from todels.residual.resnet import BottleneckBlock


class ResnextBlockA(nn.Module):
    def __init__(self,
                 out_channels_convs: int,
                 C: int,
                 stride: Union[int, Iterable] = 1,
                 downsample: bool = False,
                 activation: Optional[str] = "ReLU",
                 device: Optional[Union[torch.device, str]] = None):
        """
        A Resnext block (type A) for Resnext layer
        summation of some bottleneck block and
        also added with the shortcut path at the end.

        Parameters:
        ----------
            out_channels_convs: Union[Iterable, int]
                the number of block output channels
            stride: Union[int, Iterable] = 1
                 the stride value(s) for middle conv (conv with kernel_size 3)
            downsample: bool = False
                if it is True then input sum with output of block
            C: int
                Cardinality of block (number of parallel block)
            activation: str = "ReLU"
                if it isn't None then do activation after summation of output with input
            device: Optional[Union[torch.device, str]] = None
                target device which you would run on
        """
        super(self.__class__, self).__init__()

        # TODO: control better d
        d = (out_channels_convs) // C
        self.blocks = []
        for _ in range(C):
            self.blocks.append(BottleneckBlock(out_channels_convs = [C*d, C*d, out_channels_convs*4],
                                               stride = stride,
                                               has_identity = False,
                                               downsample = downsample,
                                               activation = None,
                                               device = device))
        self.blocks = nn.ModuleList(self.blocks)
        self.bn_group = nn.BatchNorm2d(out_channels_convs*4, device=device)
        # TODO: check via stride and output channels
        if downsample:
            stride = 2
        else:
            stride = 1
        self.short_way = ResnetShortcut(out_channels_convs*4,
                                        downsample=True,
                                        stride=stride,
                                        activation=None,
                                        device=device)

        if activation is not None:
            if activation.capitalize() == "Relu":
                self.activation = nn.ReLU(True)
            else:
                raise NotImplementedError(f"{activation} is not implemented yet!")
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        out = [block(x) for block in self.blocks]
        out = sum(out)
        out = self.short_way(x) + self.bn_group(out)
        return self.activation(out)


class ResnextBlockB(nn.Module):
    def __init__(self,
                 out_channels_convs: int,
                 C: int,
                 stride: Union[int, Iterable] = 1,
                 downsample: bool = False,
                 activation: Optional[str] = "ReLU",
                 device: Optional[Union[torch.device, str]] = None):
        """
        A Resnext block (type B) for Resnext layer
        concat the output of some conv1x1 and conv3x3 together and then use a conv1x1
        and also added with the shortcut path at the end.

        Parameters:
        ----------
            out_channels_convs: Union[Iterable, int]
                the number of block output channels
            stride: Union[int, Iterable] = 1
                 the stride value(s) for middle conv (conv with kernel_size 3)
            downsample: bool = False
                if it is True then input sum with output of block
            C: int
                Cardinality of block (number of parallel block)
            activation: str = "ReLU"
                if it isn't None then do activation after summation of output with input
            device: Optional[Union[torch.device, str]] = None
                target device which you would run on
        """
        super(self.__class__, self).__init__()

        # TODO: control better d
        d = (out_channels_convs) // C
        self.blocks = BottleneckBlock(out_channels_convs = [C*d, C*d, out_channels_convs*4],
                                               stride = stride,
                                               groups = C,
                                               has_identity = False,
                                               downsample = downsample,
                                               activation = None,
                                               device = device)
        self.bn_group = nn.BatchNorm2d(out_channels_convs*4, device=device)
        # TODO: check via stride and output channels
        if downsample:
            stride = 2
        else:
            stride = 1
        self.short_way = ResnetShortcut(out_channels_convs*4,
                                        downsample=True,
                                        stride=stride,
                                        activation=None,
                                        device=device)

        if activation is not None:
            if activation.capitalize() == "Relu":
                self.activation = nn.ReLU(True)
            else:
                raise NotImplementedError(f"{activation} is not implemented yet!")
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        out = self.blocks(x)
        out = self.short_way(x) + self.bn_group(out)
        return self.activation(out)
