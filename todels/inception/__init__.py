from __future__ import annotations

import torch
import torch.nn as nn

from todels import ConvBlock
from todels.inception.blocks import *


class InceptionV1(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.4, device=None):
        super(self.__class__, self).__init__()

        self.conv_1 = ConvBlock(out_channel = 64,
                                kernel_size = 7,
                                stride = 2,
                                padding = 3,
                                type_norm = "local",
                                activation = "ReLU",
                                device = device)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv_2 = ConvBlock(out_channel = 192,
                                kernel_size = 3,
                                stride = 1,
                                padding = 1,
                                type_norm = "local",
                                activation = "ReLU",
                                device = device)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_3a = _InceptionBlockV1([64, 96,128, 16,32, 32])
        self.inception_3b = _InceptionBlockV1([128, 128,192, 32,96, 64])
        self.max_pool_3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.inception_4a = _InceptionBlockV1([192, 96,208, 16,48, 64])
        self.aux_1 = _InceptionAUX(num_classes, dropout_rate=dropout_rate, device=device)
        self.inception_4b = _InceptionBlockV1([160, 112,224, 24,64, 64])
        self.inception_4c = _InceptionBlockV1([128, 128,256, 24,64, 64])
        self.inception_4d = _InceptionBlockV1([112, 144,288, 32,64, 64])
        self.inception_4e = _InceptionBlockV1([256, 160,320, 32,128, 128])
        self.max_pool_4 = nn.MaxPool2d(3, stride=2, padding=1)

        self.aux_2 = _InceptionAUX(num_classes, dropout_rate=dropout_rate, device=device)

        self.inception_5a = _InceptionBlockV1([256, 160,320, 32,128, 128])
        self.inception_5b = _InceptionBlockV1([384, 192,384, 48,128, 128])

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.LazyLinear(num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.max_pool_1(out)
        out = self.conv_2(out)
        out = self.max_pool_2(out)

        out = self.inception_3a(out)
        out = self.inception_3b(out)
        out = self.max_pool_3(out)

        out = self.inception_4a(out)
        out1 = self.aux_1(out)
        out = self.inception_4b(out)
        out = self.inception_4c(out)
        out = self.inception_4d(out)
        out2 = self.aux_2(out)
        out = self.inception_4e(out)
        out = self.max_pool_4(out)

        out = self.inception_5a(out)
        out = self.inception_5b(out)
        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.fc(out)

        return out, out1, out2


# version 2 and 3 are almost same
class InceptionV3(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.4, device=None):
        super(self.__class__, self).__init__()
        self.block_1 = nn.Sequential(ConvBlock(out_channel = 32,
                                               kernel_size = 3,
                                               stride = 2,
                                               padding = 0,
                                               type_norm = "local",
                                               activation = "ReLU",
                                               device = device),
                                     ConvBlock(out_channel = 32,
                                               kernel_size = 3,
                                               stride = 1,
                                               padding = 0,
                                               type_norm = "local",
                                               activation = "ReLU",
                                               device = device),
                                     ConvBlock(out_channel = 64,
                                               kernel_size = 3,
                                               stride = 1,
                                               padding = 1,
                                               type_norm = "local",
                                               activation = "ReLU",
                                               device = device))
        self.max_pool_1 = nn.MaxPool2d(3, stride=2, padding=0)

        self.block_2 = nn.Sequential(ConvBlock(out_channel = 80,
                                               kernel_size = 3,
                                               stride = 1,
                                               padding = 0,
                                               type_norm = "local",
                                               activation = "ReLU",
                                               device = device),
                                     ConvBlock(out_channel = 192,
                                               kernel_size = 3,
                                               stride = 2,
                                               padding = 0,
                                               type_norm = "local",
                                               activation = "ReLU",
                                               device = device),
                                     ConvBlock(out_channel = 288,
                                               kernel_size = 3,
                                               stride = 1,
                                               padding = 1,
                                               type_norm = "local",
                                               activation = "ReLU",
                                               device = device))
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception_3a = _InceptionBlockV2_A([64, 48,64, 64,96,96, 64], device = device)
        self.inception_3b = _InceptionBlockV2_A([64, 48,64, 64,96,96, 64], device = device)
        self.inception_3c = _InceptionBlockV2_A([64, 48,64, 64,96,96, 64], device = device)

        self.reduction_1 = _InceptionReduction([64,178, 64,302], device = device)
        
        self.inception_4a = _InceptionBlockV2_B([192, 128,192, 128,192, 192], device = device)
        self.inception_4b = _InceptionBlockV2_B([192, 160,192, 160,192, 192], device = device)
        self.inception_4c = _InceptionBlockV2_B([192, 160,192, 160,192, 192], device = device)
        self.inception_4d = _InceptionBlockV2_B([192, 160,192, 160,192, 192], device = device)
        self.inception_4e = _InceptionBlockV2_B([192, 192,192, 192,192, 192], device = device)

        self.reduction_2 = _InceptionReduction([192,194, 192,318], device = device)

        self.aux = _InceptionAUX(num_classes, dropout_rate=dropout_rate, device=device)

        self.inception_5a = _InceptionBlockV2_C([320, 384,384, 448, 384,384, 192], device = device)
        self.inception_5b = _InceptionBlockV2_C([320, 384,384, 448, 384,384, 192], device = device)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.LazyLinear(num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        out = self.block_1(x)
        out = self.max_pool_1(out)
        out = self.block_2(out)
        out = self.max_pool_2(out)
        
        out = self.inception_3a(out)
        out = self.inception_3b(out)
        out = self.inception_3c(out)

        out = self.reduction_1(out)

        out = self.inception_4a(out)
        out = self.inception_4b(out)
        out = self.inception_4c(out)
        out = self.inception_4d(out)
        out = self.inception_4e(out)

        aux = self.aux(out)

        out = self.reduction_2(out)
        out = self.inception_5a(out)
        out = self.inception_5b(out)

        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.softmax(out)

        return out, aux


__all__ = ("InceptionV1", "InceptionV3")


# TODO: Add docs
# TODO: Test on a real dataset