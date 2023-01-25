from __future__ import annotations

import functools
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn

from todels.residual import ResnetShortcut
from todels.residual.resnet import SimpleResnetBlock, BottleneckBlock
from todels import _create_conv_layer


class SimpleResnextBlock(nn.Module):
    def __init__(self, n_groups, out_channels_convs, stride=1, downsample=False, activation="ReLU", device=None):
        super(self.__class__, self).__init__()
        self.blocks = []
        for i in range(n_groups):
            self.blocks.append(
                SimpleResnetBlock(out_channels_convs = out_channels_convs,
                                  stride = stride,
                                  has_identity = False,
                                  downsample = False,
                                  activation = None,
                                  device = device)
            )
        
        self.blocks = nn.ModuleList(self.blocks)
        self.bn_group = nn.BatchNorm2d(out_channels_convs, device=device)
        self.short_way = ResnetShortcut(out_channels_convs,
                                        downsample=downsample,
                                        stride=2,
                                        padding=0,
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


class ResnextBottleneckBlock(nn.Module):
    def __init__(self, n_groups, out_channels_convs, stride=1, downsample=False, activation="ReLU", device=None):
        super(self.__class__, self).__init__()
        self.blocks = []
        for i in range(n_groups):
            self.blocks.append(
                BottleneckBlock(out_channels_convs = out_channels_convs,
                                stride = 2,
                                has_identity = False,
                                downsample = downsample,
                                activation = None,
                                device = device)
            )
        
        self.blocks = nn.ModuleList(self.blocks)
        self.bn_group = nn.BatchNorm2d(out_channels_convs*4, device=device)
        self.short_way = ResnetShortcut(out_channels_convs*4,
                                        downsample=True,
                                        stride=2,
                                        padding=0,
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


class _Resnext(nn.Module):
    def __init__(self,
                 num_classes: int,
                 layers: Iterable,
                 C: int = 32,
                 out_channels: Iterable = [64,128,256,512],
                 is_bottleneck: Optional[bool] = None,
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
                the number of prallel residual unit for each layer
            out_channels: Iterable = [64,128,256,512]
                the out channels of each block (layers) and if the model is Bottleneck Resnet then the third conv of each block will be multiplied by four
            is_bottleneck: Optional[bool] = None
                use bottleneck block or simple block and if this parameter is None then specified from the number of layers
            device: Optional[Union[torch.device, str]] = None)
                target device which you would run on
        """
        # TODO: implement more choice

        super(self.__class__, self).__init__()
        
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
                ResnextBlock = SimpleResnextBlock
            else:
                ResnextBlock = ResnextBottleneckBlock
                out_channels.append(out_channels[-1]*4)
        else:
            if is_bottleneck:
                ResnextBlock = ResnextBottleneckBlock
                out_channels.append(out_channels[-1]*4)
            else:
                ResnextBlock = SimpleResnextBlock


        # block 2
        block2 = []
        for i in range(layers[0]):
            if i == 0:
                stride=2
                downsample = True
            else:
                stride=1
                downsample = False

            block2.append(ResnextBlock(n_groups=C,
                                       out_channels_convs = out_channels[0],
                                       stride = stride,
                                       downsample=downsample,
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
            block3.append(ResnextBlock(n_groups=C,
                                       out_channels_convs = out_channels[1],
                                       stride=stride,
                                       downsample=downsample,
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
                
            block4.append(ResnextBlock(n_groups=C,
                                       out_channels_convs = out_channels[2],
                                       stride=stride,
                                       downsample=downsample,
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
                
                
            block4.append(ResnextBlock(n_groups=C,
                                       out_channels_convs = out_channels[3],
                                       stride=stride,
                                       downsample=downsample,
                                       device = device,
                                       activation = "ReLU"))
        self.block5 = nn.Sequential(*block5)
        
        # fc (last block)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes, device=device),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        return self.fc(out)

Resnext18 = functools.partial(_Resnext, layers=[2,2,2,2])
Resnext34 = functools.partial(_Resnext, layers=[3,4,6,3])
Resnext50 = functools.partial(_Resnext, layers=[3,4,6,3], is_bottleneck=True)
Resnext101 = functools.partial(_Resnext, layers=[3,4,23,3])
Resnext152 = functools.partial(_Resnext, layers=[3,8,36,3])
