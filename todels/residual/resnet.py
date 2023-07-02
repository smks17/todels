from __future__ import annotations

from typing import Iterable, Optional, Union
import functools

import torch
import torch.nn as nn

from todels import ConvBlock, MLP
from todels.residual import (
    LAYERS_RESIDUAL18,
    LAYERS_RESIDUAL34,
    LAYERS_RESIDUAL50,
    LAYERS_RESIDUAL101,
    LAYERS_RESIDUAL152
)
from todels.residual.resnetblock import SimpleResnetBlock, BottleneckBlock

class _Resnet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 layers: Iterable,
                 out_channels: Iterable = [64,128,256,512],
                 is_bottleneck: Optional[bool] = None,
                 is_light = False,
                 device: Optional[Union[torch.device, str]] = None):
        super(self.__class__, self).__init__()
        """
        Base Resnet class implementation

        Parameters:
        ----------
            num_classes: int
                the size of model output
            layers: Iterable
                the number of each blocks (layers) (and should has len=4)
            out_channels: Iterable = [64,128,256,512]
                the out channels of each block (layers) and if the model is Bottleneck Resnet then the third conv of each block will be multiplied by four
            is_bottleneck: Optional[bool] = None
                use bottleneck block or simple block and if this parameter is None then specified from the number of layers
            device: Optional[Union[torch.device, str]] = None)
                target device which you would run on
        """
        # TODO: implement more choice
        
        # block 1
        if not is_light:
            self.conv1 = ConvBlock(out_channels[0],
                                   kernel_size=7,
                                   stride=2,
                                   padding=3,
                                   type_norm="batch",
                                   activation="ReLU",
                                   device=device)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        else:
            self.conv1 = ConvBlock(out_channels[0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   type_norm="batch",
                                   activation="ReLU",
                                   device=device)
            self.maxpool = None
            
        if is_bottleneck is None:
            if sum(layers) <= 32:
                ResnetBlock = SimpleResnetBlock
            else:
                ResnetBlock = BottleneckBlock
                out_channels.append(out_channels[-1]*4)
        else:
            if is_bottleneck:
                ResnetBlock = BottleneckBlock
                out_channels.append(out_channels[-1]*4)
            else:
                ResnetBlock = SimpleResnetBlock

        # block 2
        block2 = []
        stride = 1 #  doing that from resnet paper
        for i in range(layers[0]):
            block2.append(ResnetBlock(out_channels_convs=out_channels[0],
                                      stride=stride,
                                      has_identity=True,
                                      first_block=True,
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
            block3.append(ResnetBlock(out_channels_convs=out_channels[1],
                                      stride=stride,
                                      has_identity=True,
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
            block4.append(ResnetBlock(out_channels_convs=out_channels[2],
                                      stride=stride,
                                      has_identity=True,
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
            block5.append(ResnetBlock(out_channels_convs = out_channels[3],
                                      stride=stride,
                                      has_identity=True,
                                      device=device,
                                      activation="ReLU"))
        self.block5 = nn.Sequential(*block5)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # fc (last block)
        self.fc = MLP([num_classes],
                      final_activation_function = "softmax",
                      device=device)
    
    def forward(self, x):
        out = self.conv1(x)
        if self.maxpool is not None:
            out = self.maxpool(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        return self.fc(out)


Resnet18 = functools.partial(_Resnet, layers=LAYERS_RESIDUAL18)
Resnet34 = functools.partial(_Resnet, layers=LAYERS_RESIDUAL34)
Resnet50 = functools.partial(_Resnet, layers=LAYERS_RESIDUAL50, is_bottleneck=True)
Resnet101 = functools.partial(_Resnet, layers=LAYERS_RESIDUAL101)
Resnet152 = functools.partial(_Resnet, layers=LAYERS_RESIDUAL152)

# TODO: test module with real dataset
# TODO: implement light resnet (more customize)
