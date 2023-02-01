from typing import Iterable, Optional, Union

import torch
import torch.nn as nn


def _create_conv_layer(out_channel: int,
                       kernel_size: Union[int, Iterable],
                       stride: Union[int, Iterable] = 1,
                       padding: Union[int, Iterable] = 0,
                       groups: int = 1,
                       has_bn: bool = True,
                       activation: Optional[str] = "ReLU",
                       device: Optional[Union[torch.device, str]] = None
):
    """
    a function to create a conv2d->bn->activation block

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
        has_bn: bool = True,
            has batch normalize or not
        activation: Optional[str] = "ReLU"
            if it isn't None then do activation to output
        device: Optional[Union[torch.device, str]] = None
            target device which you would run on
        
    """
    # TODO: implement for 3d too
    layer = nn.Sequential()
    bias = True
    if has_bn:
        bias = False
    layer.add_module(
        "Conv2d",
        nn.LazyConv2d(out_channels=out_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=bias,
                      device=device)
    )
    if has_bn:
        layer.add_module("BatchNorm2d", nn.BatchNorm2d(out_channel, device=device))

    # TODO: better handle
    if activation is not None:
        if activation.capitalize() == "Relu":
            layer.add_module("ReLU", nn.ReLU(True))
        else:
            raise NotImplementedError(f"{activation} is not implemented yet!")

    return layer

def _create_fc(num_classes: int,
               final_activation: str = "Softmax",
               device: Optional[Union[torch.device, str]] = None):
    if final_activation.capitalize() == "Softmax":
        activation = nn.Softmax
    elif final_activation.capitalize() == "Logsoftmax":
        activation = nn.LogSoftmax
    else:
        raise NotImplementedError(f"{activation} is not implemented yet!")
    fc = nn.Sequential(
        #TODO: delete AvgPool from fc
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.LazyLinear(num_classes, device=device),
        activation(dim=1)
    )
    return fc