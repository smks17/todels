from __future__ import annotations

from typing import Iterable, Optional, Union

import torch
import torch.nn as nn

from todels import _create_conv_layer


class ResnetShortcut(nn.Module):
    def __init__(self,
                 out_channel: int,
                 downsample: bool,
                 stride: Union[int, Iterable],
                 padding: Union[int, Iterable],
                 activation: Optional[str] = None,
                 device: Optional[Union[torch.device, str]] = None):
        super(self.__class__, self).__init__()
        if not downsample:
            self.short_way = nn.Identity()
        else:
            self.short_way = _create_conv_layer(out_channel,
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               has_bn=True,
                                               activation=activation,
                                               device=device)
    def forward(self, x):
        return self.short_way(x)
