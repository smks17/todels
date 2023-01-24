from __future__ import annotations
import functools

from typing import Iterable, Optional, Union

import torch
import torch.nn as nn

from todels import _create_conv_layer
from todels.residual.resnetblock import SimpleResnetBlock, BottleneckBlock

class _Resnet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 layers: Iterable,
                 out_channels: Iterable = [64,128,256,512],
                 is_bottleneck: Optional[bool] = None,
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
                the out channels of each blocks (layers) and if the model is Bottleneck Resnet then the third conv of each blocks will be multiply by four
            is_bottleneck: Optional[bool] = None
                use bottleneck block or simple block anf if this parameter is None then specified from number of layers
            device: Optional[Union[torch.device, str]] = None)
                target device which you would run on
        """
        # TODO: implement more choice
        
        # block 1
        self.block1 = _create_conv_layer(out_channels[0],
                                        kernel_size=7,
                                        stride=2,
                                        padding=3,
                                        has_bn=True,
                                        activation="ReLU",
                                        device=device)
        self.block1.add_module("MaxPool2d", nn.MaxPool2d(3, stride=2, padding=1))

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
        for i in range(layers[0]):
            if i == 0:
                stride=2
                downsample = True
            else:
                stride=1
                downsample = False

            block2.append(ResnetBlock(out_channels_convs = out_channels[0],
                                      stride=stride,
                                      has_identity = True,
                                      downsample = downsample,
                                      device = device,
                                      activation = "ReLU"))
        self.block2 = nn.Sequential(*block2)

        # block 3
        block3 = []
        for i in range(layers[1]):
            if i == 0:
                downsample = True
                stride=2
            else:
                downsample = False
                stride=1
            block3.append(ResnetBlock(out_channels_convs = out_channels[1],
                                      stride=stride,
                                      has_identity = True,
                                      downsample = downsample,
                                      device = device,
                                      activation = "ReLU"))
        self.block3 = nn.Sequential(*block3)
        
        # block 4
        block4 = []
        for i in range(layers[2]):
            if i == 0:
                stride=2
                downsample = True
            else:
                stride=1
                downsample = False
                
            block4.append(ResnetBlock(out_channels_convs = out_channels[2],
                                      stride=stride,
                                      has_identity = True,
                                      downsample = downsample,
                                      device = device,
                                      activation = "ReLU"))
        self.block4 = nn.Sequential(*block4)
        
        # block 5
        block5 = []
        for i in range(layers[3]):
            if i == 0:
                stride=2,
                downsample = True
            else:
                stride=1,
                downsample = False
                
            block5.append(ResnetBlock(out_channels_convs = out_channels[3],
                                      stride=stride,
                                      has_identity = True,
                                      downsample = downsample,
                                      device = device,
                                      activation = "ReLU"))
        self.block5 = nn.Sequential(*block5)
        
        # fc (last block)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels[-1], num_classes, device=device),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        return self.fc(out)


Resnet18 = functools.partial(_Resnet, layers=[2,2,2,2])
Resnet34 = functools.partial(_Resnet, layers=[3,4,6,3])
Resnet50 = functools.partial(_Resnet, layers=[3,4,6,3], is_bottleneck=True)
Resnet101 = functools.partial(_Resnet, layers=[3,4,23,3])
Resnet152 = functools.partial(_Resnet, layers=[3,8,36,3])