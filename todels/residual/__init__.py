from __future__ import annotations

from typing import Iterable, Optional, Union

import torch
import torch.nn as nn

from todels import ConvBlock


class ResnetShortcut(nn.Module):
    def __init__(self,
                 out_channel: int,
                 stride: Union[int, Iterable] = 2,  # also control downsample via stride
                 activation: Optional[str] = None,
                 do_conv: bool = False,
                 device: Optional[Union[torch.device, str]] = None):
        super(self.__class__, self).__init__()
        if not do_conv and stride == 1:
            self.short_way = nn.Identity()
        else:
            self.short_way = ConvBlock(out_channel,
                                       kernel_size=1,
                                       stride=stride,
                                       padding=0,
                                       type_norm="batch",
                                       activation=activation,
                                       device=device)
    def forward(self, x):
        return self.short_way(x)

LAYERS_RESIDUAL18 = [2,2,2,2]
LAYERS_RESIDUAL34 = [3,4,6,3]
LAYERS_RESIDUAL50 = [3,4,6,3]
LAYERS_RESIDUAL101 = [3,4,23,3]
LAYERS_RESIDUAL152 = [3,8,36,3]
