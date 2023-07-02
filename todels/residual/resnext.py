"""Reference article: https://arxiv.org/abs/1611.05431v2"""

from __future__ import annotations

import functools
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn

from todels import ConvBlock, MLP
from todels.residual import (
    LAYERS_RESIDUAL50,
    LAYERS_RESIDUAL101,
    LAYERS_RESIDUAL152
)
from todels.residual.resnextblock import ResnextBlockA


class _Resnext(nn.Module):
    def __init__(self,
                 num_classes: int,
                 layers: Iterable,
                 C: int = 32,
                 out_channels: Iterable = [64,128,256,512],
                 ResnextBlock: object = ResnextBlockA,
                 device: Optional[Union[torch.device, str]] = None):
        """
        Base Resnet class implementation

        Parameters:
        ----------
            num_classes: int
                the size of model output
            layers: Iterable
                the number of each blocks (layers) (and should has len=4)
            C: int = 32
                the number of parallel residual unit for each layer
            out_channels: Iterable = [64,128,256,512]
                the out channels of each block (layers) and if the model is Bottleneck Resnet then the third conv of each block will be multiplied by four
            ResnextBlock: object
                type of ResnexBlocks as a class (type A and B)
            device: Optional[Union[torch.device, str]] = None)
                target device which you would run on
        """
        # TODO: implement more choice

        super(self.__class__, self).__init__()
        
        # block 1
        self.conv1 = ConvBlock(out_channels[0],
                                kernel_size=7,
                                stride=2,
                                padding=3,
                                type_norm="batch",
                                activation="ReLU",
                                device=device)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # block 2
        block2 = []
        stride = 1 #  doing that from resnext paper
        for i in range(layers[0]):
            block2.append(ResnextBlock(out_channels_convs=out_channels[0],
                                       stride=stride,
                                       C=C,
                                       device=device,
                                       activation="ReLU"))
        self.block2 = nn.Sequential(*block2)

        
        # block 3
        block3 = []
        for i in range(layers[1]):
            if i == 0:
                stride=2
            else:
                stride=1
            block3.append(ResnextBlock(out_channels_convs=out_channels[1],
                                       stride=stride,
                                       C=C,
                                       device=device,
                                       activation="ReLU"))
        self.block3 = nn.Sequential(*block3)

        # block 4
        block4 = []
        for i in range(layers[2]):
            if i == 0:
                stride=2
            else:
                stride=1
            block4.append(ResnextBlock(out_channels_convs=out_channels[2],
                                       stride=stride,
                                       C=C,
                                       device=device,
                                       activation="ReLU"))
        self.block4 = nn.Sequential(*block4)
        
        # block 5
        block5 = []
        for i in range(layers[3]):
            if i == 0:
                stride=2,
            else:
                stride=1,
            block5.append(ResnextBlock(out_channels_convs=out_channels[3],
                                       stride=stride,
                                       C=C,
                                       device=device,
                                       activation="ReLU"))
        self.block5 = nn.Sequential(*block5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()        
        # fc (last block)
        self.fc = MLP([num_classes],
                      final_activation_function="softmax",
                      device=device)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        return self.fc(out)

Resnext50 = functools.partial(_Resnext, layers=LAYERS_RESIDUAL50)
Resnext101 = functools.partial(_Resnext, layers=LAYERS_RESIDUAL101)
Resnext152 = functools.partial(_Resnext, layers=LAYERS_RESIDUAL152)

# TODO: test module with real dataset
