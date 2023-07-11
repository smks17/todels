from __future__ import annotations

import torch
import torch.nn as nn

from todels import ConvBlock
from todels.util_layers import CatLayer, BranchAndCatLayer


class _InceptionAUX(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.4, device=None):
        super(self.__class__, self).__init__()
        self.aux = nn.Sequential(nn.AdaptiveAvgPool2d((5,5)),
                                 nn.LazyConv2d(1024, kernel_size=1, stride=1, padding=0),
                                 nn.Flatten(),
                                 nn.LazyLinear(1024),
                                 nn.Dropout(dropout_rate),
                                 nn.ReLU(True),
                                 nn.LazyLinear(num_classes),
                                 nn.Softmax(-1)).to(device)

    def forward(self, x):
        return self.aux(x)

class InceptionBlock(nn.Module):
    def __init__(self, cat_dim: int = -3):
        super(InceptionBlock, self).__init__()
        self.channel_1 = NotImplemented
        self.channel_2 = NotImplemented
        self.channel_3 = NotImplemented
        self.channel_4 = NotImplemented
        self.cat_layer = CatLayer(cat_dim)

    def forward(self, x):
        assert self.channel_1 is not NotImplemented, \
            "First define channel_1 in constructor"
        assert self.channel_2 is not NotImplemented, \
            "First define channel_2 in constructor"
        assert self.channel_3 is not NotImplemented, \
            "First define channel_3 in constructor"
        assert self.channel_4 is not NotImplemented, \
            "First define channel_4 in constructor"
        out1 = self.channel_1(x)
        out2 = self.channel_2(x)
        out3 = self.channel_3(x)
        out4 = self.channel_4(x)
        return self.cat_layer(out1, out2, out3, out4)


#####################################
#              version1
#####################################
class _InceptionBlockV1(InceptionBlock):
    def __init__(self, out_channels, device=None):
        super(self.__class__, self).__init__()
        self.channel_1 = ConvBlock(out_channel = out_channels[0],
                                   kernel_size = 1,
                                   stride = 1,
                                   padding = 0,
                                   type_norm = "batch",
                                   activation = "ReLU",
                                   device = device)
        channel_2 = [ConvBlock(out_channel = out_channels[1],
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     ConvBlock(out_channel = out_channels[2],
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device)]
        channel_3 = [ConvBlock(out_channel = out_channels[3],
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     ConvBlock(out_channel = out_channels[4],
                               kernel_size = 5,
                               stride = 1,
                               padding = 2,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device)]
        channel_4 = [nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
                     ConvBlock(out_channel = out_channels[5],
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device)]
        self.channel_2 = nn.Sequential(*channel_2)
        self.channel_3 = nn.Sequential(*channel_3)
        self.channel_4 = nn.Sequential(*channel_4)


#####################################
#              version2
#####################################

class _InceptionBlockV2_A(InceptionBlock):
    def __init__(self, out_channels, device=None):
        super(self.__class__, self).__init__()
        self.channel_1 = ConvBlock(out_channel = out_channels[0],
                                   kernel_size = 1,
                                   stride = 1,
                                   padding = 0,
                                   type_norm = "batch",
                                   activation = "ReLU",
                                   device = device)
        channel_2 = [ConvBlock(out_channel = out_channels[1],
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     ConvBlock(out_channel = out_channels[2],
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device)]
        channel_3 = [ConvBlock(out_channel = out_channels[3],
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     ConvBlock(out_channel = out_channels[4],
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     ConvBlock(out_channel = out_channels[5],
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device)]
        channel_4 = [nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
                     ConvBlock(out_channel = out_channels[6],
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device)]
        self.channel_2 = nn.Sequential(*channel_2)
        self.channel_3 = nn.Sequential(*channel_3)
        self.channel_4 = nn.Sequential(*channel_4)


class _InceptionBlockV2_B(InceptionBlock):
    def __init__(self, out_channels, n=3, device=None):
        super(self.__class__, self).__init__()
        self.channel_1 = ConvBlock(out_channel = out_channels[0],
                                   kernel_size = 1,
                                   stride = 1,
                                   padding = 0,
                                   type_norm = "batch",
                                   activation = "ReLU",
                                   device = device)
        channel_2 = [ConvBlock(out_channel = out_channels[1],
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     ConvBlock(out_channel = out_channels[1],
                               kernel_size = (1, n),
                               stride = 1,
                               padding = (0, n//2),
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     ConvBlock(out_channel = out_channels[2],
                               kernel_size = (n, 1),
                               stride = 1,
                               padding = (n//2, 0),
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device)]
        channel_3 = [ConvBlock(out_channel = out_channels[3],
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     ConvBlock(out_channel = out_channels[3],
                               kernel_size = (1, n),
                               stride = 1,
                               padding = (0, n//2),
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     ConvBlock(out_channel = out_channels[3],
                               kernel_size = (n, 1),
                               stride = 1,
                               padding = (n//2, 0),
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     ConvBlock(out_channel = out_channels[3],
                               kernel_size = (1, n),
                               stride = 1,
                               padding = (0, n//2),
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     ConvBlock(out_channel = out_channels[4],
                               kernel_size = (n, 1),
                               stride = 1,
                               padding = (n//2,0),
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device)]
        channel_4 = [nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
                     ConvBlock(out_channel = out_channels[5],
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device)]
        self.channel_2 = nn.Sequential(*channel_2)
        self.channel_3 = nn.Sequential(*channel_3)
        self.channel_4 = nn.Sequential(*channel_4)


class _InceptionBlockV2_C(InceptionBlock):
    def __init__(self, out_channels, n=3, device=None):
        super(self.__class__, self).__init__()
        self.channel_1 = ConvBlock(out_channel = out_channels[0],
                                   kernel_size = 1,
                                   stride = 1,
                                   padding = 0,
                                   type_norm = "batch",
                                   activation = "ReLU",
                                   device = device)
        channel_2 = [ConvBlock(out_channel = out_channels[1],
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     BranchAndCatLayer([ConvBlock(out_channel = out_channels[2],
                                                  kernel_size = (1, n),
                                                  stride = 1,
                                                  padding = (0, n//2),
                                                  type_norm = "batch",
                                                  activation = "ReLU",
                                                  device = device),
                                        ConvBlock(out_channel = out_channels[2],
                                                  kernel_size = (n, 1),
                                                  stride = 1,
                                                  padding = (n//2, 0),
                                                  type_norm = "batch",
                                                  activation = "ReLU",
                                                  device = device)])]
        channel_3 = [ConvBlock(out_channel = out_channels[3],
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     ConvBlock(out_channel = out_channels[4],
                               kernel_size = 3,
                               stride = 1,
                               padding = 1,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device),
                     BranchAndCatLayer([ConvBlock(out_channel = out_channels[5],
                                                  kernel_size = (1, n),
                                                  stride = 1,
                                                  padding = (0, n//2),
                                                  type_norm = "batch",
                                                  activation = "ReLU",
                                                  device = device),
                                        ConvBlock(out_channel = out_channels[5],
                                                  kernel_size = (n, 1),
                                                  stride = 1,
                                                  padding = (n//2, 0),
                                                  type_norm = "batch",
                                                  activation = "ReLU",
                                                  device = device)])]
        channel_4 = [nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
                     ConvBlock(out_channel = out_channels[6],
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               type_norm = "batch",
                               activation = "ReLU",
                               device = device)]
        self.channel_2 = nn.Sequential(*channel_2)
        self.channel_3 = nn.Sequential(*channel_3)
        self.channel_4 = nn.Sequential(*channel_4)


class _InceptionReduction(nn.Module):
    def __init__(self, out_channels, device = None):
        super(_InceptionReduction, self).__init__()
        branch_1 = nn.Sequential(ConvBlock(out_channels[0],
                                           kernel_size = 1,
                                           stride = 1,
                                           padding = 0,
                                           type_norm = "batch",
                                           activation = "ReLU",
                                           device = device),
                                 ConvBlock(out_channels[1],
                                           kernel_size = 3,
                                           stride = 1,
                                           padding = 1,
                                           type_norm = "batch",
                                           activation = "ReLU",
                                           device = device),
                                 ConvBlock(out_channels[1],
                                           kernel_size = 3,
                                           stride = 2,
                                           padding = 0,
                                           type_norm = "batch",
                                           activation = "ReLU",
                                           device = device))
        branch_2 = nn.Sequential(ConvBlock(out_channels[2],
                                           kernel_size = 1,
                                           stride = 1,
                                           padding = 0,
                                           type_norm = "batch",
                                           activation = "ReLU",
                                           device = device),
                                 ConvBlock(out_channels[3],
                                           kernel_size = 3,
                                           stride = 2,
                                           padding = 0,
                                           type_norm = "batch",
                                           activation = "ReLU",
                                           device = device))
        max_pool = nn.MaxPool2d(3, stride = 2, padding = 0)
        self.branch_and_cat = BranchAndCatLayer([branch_1, branch_2, max_pool])

    def forward(self, x):
        return self.branch_and_cat(x)


__all__ = ("_InceptionAUX",
           "_InceptionBlockV1",
           "_InceptionBlockV2_A",
           "_InceptionBlockV2_B",
           "_InceptionBlockV2_C",
           "_InceptionReduction")
