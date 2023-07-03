"""Residual Block. uses in Residual Models that have a shortcut to prevent gradient vanishing
example:

          -------------        ReLU
input -> | Res Block   | -> + ------> output
  |       -------------     |
   ------------>------------
           shortcut
"""


from __future__ import annotations

from typing import Iterable, Optional, Union

import torch
import torch.nn as nn

from todels.residual import ResnetShortcut
from todels import ConvBlock


class SimpleResnetBlock(nn.Module):
    """
    A residual unit for resnet network with layers <= 34

    the main pipes which sum with shortcut

    -> conv3x3 -> bn -> activation -> conv3x3 -> bn
    """
    def __init__(self,
                 out_channels_convs: Iterable,
                 stride: Union[int, Iterable] = 1,
                 has_identity: bool = True,
                 first_block: bool = True,
                 activation: Optional[str] = None,
                 device: Optional[Union[torch.device, str]] = None):
        """
        Parameters:
        ----------
            out_channels_convs: Union[Iterable, int]
                the number of block output channels
            stride: Union[int, Iterable] = 1
                 the stride value(s) for first conv and the short way
            has_identity: bool = True
                if it is True then sum with input at the end otherwise
                this block will be just a normal block
            activation: str = "ReLU"
                if it isn't None than do activation after sum output with input
            device: Optional[Union[torch.device, str]] = None
                target device which you would run on
        """
        super(self.__class__, self).__init__()
        if not isinstance(out_channels_convs, Iterable) and isinstance(out_channels_convs, int):
            out_channels_convs = [out_channels_convs, out_channels_convs]
        self.layer1 = ConvBlock(out_channels_convs[0],
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                type_norm="batch",
                                activation="ReLU",
                                device=device)
        self.layer2 = ConvBlock(out_channels_convs[1],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                type_norm="batch",
                                activation=None,
                                device=device)
        # shortcut
        self.has_identity = has_identity
        if self.has_identity:
            # will be checked downsample via stride
            self.short_way = ResnetShortcut(out_channels_convs[-1],
                                            stride=stride,
                                            device=device)
        # last activation
        # TODO: better handle
        if activation is not None:
            if activation.capitalize() == "Relu":
                self.activation = nn.ReLU(True)
            else:
                raise NotImplementedError(f"{activation} is not implemented yet!")
        else:
            self.activation = nn.Identity()

    def forward(self, inputs):
        out = self.layer1(inputs)
        out = self.layer2(out)
        if self.has_identity:
            out += self.short_way(inputs)
        return self.activation(out)


class BottleneckBlock(nn.Module):
    """
    A residual unit for resnet network with layers > 34

    the main pipe which sums with shortcut

    -> conv1x1 -> bn -> activation -> conv3x3 -> bn -> activation -> conv1x1 -> bn
    """
    def __init__(self,
                 out_channels_convs: Iterable,
                 stride: Union[int, Iterable] = 1,
                 groups: int = 1,
                 #TODO: delete first_block as param and add in_channels for control
                 first_block = False,
                 has_identity: bool = True,
                 activation: Optional[str] = "ReLU",
                 device: Optional[Union[torch.device, str]] = None):
        """
        Parameters:
        ----------
            out_channels_convs: Union[Iterable, int]
                the number of block output channels
            stride: Union[int, Iterable] = 1
                 the stride value(s) for middle conv (conv with kernel_size 3)
                 and the short way
            groups: int = 1
                pass as parameter to first and second conv (uses for ResnextB)
            first_block: bool = False
                determine it is fist block in resnet and other models or not
            has_identity: bool = True
                if it is True then sum with input at the end otherwise this block
                will be just a normal block
            activation: str = "ReLU"
                if it isn't None then do activation after summation of output with input
            device: Optional[Union[torch.device, str]] = None
                target device which you would run on
        """
        super(self.__class__, self).__init__()
        if not isinstance(out_channels_convs, Iterable) and isinstance(out_channels_convs, int):
            out_channels_convs = [out_channels_convs, out_channels_convs, out_channels_convs*4]
        self.layer1 = ConvBlock(out_channels_convs[0],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                groups=groups,
                                type_norm="batch",
                                activation="ReLU",
                                device=device)
        self.layer2 = ConvBlock(out_channels_convs[1],
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                groups=groups,
                                type_norm="batch",
                                activation="ReLU",
                                device=device)
        self.layer3 = ConvBlock(out_channels_convs[2],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                type_norm="batch",
                                activation=None,
                                device=device)
        # shortcut
        self.has_identity = has_identity
        if self.has_identity:
            # will be checked downsample via stride
            do_conv = False
            if first_block:
                do_conv = True
            self.short_way = ResnetShortcut(out_channels_convs[-1],
                                            stride=stride,
                                            do_conv=do_conv,
                                            device=device)

        # last activation
        if activation is not None:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.Identity()

    def forward(self, inputs):
        out = self.layer1(inputs)
        out = self.layer2(out)
        out = self.layer3(out)
        if self.has_identity:
            out += self.short_way(inputs)
        return self.activation(out)
