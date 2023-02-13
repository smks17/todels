from typing import Iterable, Optional, Union

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self,
                 out_channel: int,
                 kernel_size: Union[int, Iterable],
                 stride: Union[int, Iterable] = 1,
                 padding: Union[int, Iterable] = 0,
                 groups: int = 1,
                 type_norm: Optional[str] = None,
                 activation: Optional[str] = "ReLU",
                 device: Optional[Union[torch.device, str]] = None
    ):
        """
        Parameters
        ---------
        out_channel: int
            the number of channels for output
        kernel_size: Optional[int, Iterable]
            size of conv kernel
        stride: Optional[int, Iterable] = 1
            stride of conv
        padding: Optional[int, Iterable] = 0
            size of padding fo conv
        groups: int = 1
            controls the connection input and output
        type_bn: Optional[str] = None,
            has whiich type of normalization or it it is None it means don't use
        activation: Optional[str] = "ReLU"
            if it isn't None then do activation to output
        device: Optional[Union[torch.device, str]] = None
            target device which you would run on
        """
        # TODO: implement for 3d too
        super(self.__class__, self).__init__()
        self.layer = nn.Sequential()
        bias = True
        if type_norm is None:
            bias = False
        self.layer.add_module(
            "Conv2d", nn.LazyConv2d(out_channels=out_channel,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     groups=groups,
                                     bias=bias,
                                     device=device)
        )
        # TODO: able to customize norm
        if type_norm is not None:
            if type_norm.lower() == 'batch':
                self.layer.add_module("BatchNorm2d",
                                       nn.BatchNorm2d(out_channel, device=device))
            elif type_norm.lower() == 'local':
                self.layer.add_module("LocalNorm", nn.LocalResponseNorm(11))
            else:
                raise NotImplementedError(f"normalize {type_norm} is not implemented yet!")

            # TODO: better handle
            if activation is not None:
                if activation.capitalize() == "Relu":
                    self.layer.add_module("ReLU", nn.ReLU(True))
                else:
                    raise NotImplementedError(f"activation {activation} is not implemented yet!")

    def forward(self, x):
        return self.layer(x)

def _create_fc(num_classes: int,
               final_activation: str = "Softmax",
               dropout_rate: float = 0.25,
               device: Optional[Union[torch.device, str]] = None):
    if final_activation.capitalize() == "Softmax":
        activation = nn.Softmax
    elif final_activation.capitalize() == "Logsoftmax":
        activation = nn.LogSoftmax
    else:
        raise NotImplementedError(f"{activation} is not implemented yet!")
    fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.LazyLinear(num_classes, device=device),
        activation(dim=1)
    )
    return fc
