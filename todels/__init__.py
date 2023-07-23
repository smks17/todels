from typing import Iterable, List, Optional, Union

import torch
import torch.nn as nn


import warnings
warnings.filterwarnings('ignore', module='torch.nn.modules.lazy')


class ConvBlock(nn.Module):
    def __init__(self,
                 out_channel: int,
                 kernel_size: Union[int, Iterable],
                 stride: Union[int, Iterable] = 1,
                 padding: Union[int, Iterable] = 0,
                 groups: int = 1,
                 type_norm: Optional[str] = None,
                 activation: Optional[str] = "ReLU",
                 device: Optional[Union[torch.device, str]] = None):
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
            has which type of normalization or it it is None it means don't use
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
            if activation.lower() == "relu":
                self.layer.add_module("ReLU", nn.ReLU(True))
            else:
                raise NotImplementedError(f"activation {activation} is not implemented yet!")

    def forward(self, x):
        return self.layer(x)


class MLP(nn.Module):
    """A fully connected module.
    Typically is used for fc layer.
    """
    def __init__(self,
                 layers: Iterable[int],
                 layers_activation_function: Optional[Union[List[str], str]] = "ReLU",
                 final_activation_function: Optional[str] = "Softmax",
                 dropout_rate: Optional[int] = None,
                 first_dropout = False,
                 device: Optional[Union[torch.device, str]] = None):
        """
        Parameters
        ----------
        layers: Iterable[int]
            It is a array that use for information about number of layers
            and number of perceptrons in each layer (output size). The rows are layers and
            columns are number of perceptrons in each layer.
        layers_activation_function: Optional[Union[List[str], str]] = "ReLU"
            A list of activation functions between each layer. It should be
            len(layers) - 1. Or could be a single str that show all
            layers activation function. Or if it passes None, it means won't
            use any activation function between each layer.
        final_activation_function: Optional[str] = "Softmax"
            The last activation function does on output. If it passes None,
            it means won't use activation on output.
        dropout_rate: Optional[int] = None
            The dropout betweens each layer. If it passes None, it means
            won't doing dropout.
        first_dropout: bool = False
            If it is True then at the first add a dropout before main layers
        device: Optional[Union[torch.device, str]] = None
            target device which you would run on.
        """
        super(self.__class__, self).__init__()
        self._n_layers = len(layers)
        if (layers_activation_function is not None
            and isinstance(layers_activation_function, str)
        ):
            layers_activation_function = [layers_activation_function] * (self._n_layers - 1)
        assert len(layers_activation_function) == self._n_layers - 1,  \
                "number of activation must be len(layers) - 1"
        self.network = nn.Sequential()
        if first_dropout and dropout_rate is not None:
            self.network.add_module("dropout_0", nn.Dropout(dropout_rate, True))
        for i, l in enumerate(layers):
            self.network.add_module(f"Linear_{i+1}", nn.LazyLinear(l, device=device))
            if i < (self._n_layers-1):
                if dropout_rate is not None:
                    self.network.add_module(f"Dropout_{i+1}", nn.Dropout(dropout_rate, True))
                if layers_activation_function is not None:
                    f = layers_activation_function[i]
                    if f.lower() == "relu":
                        self.network.add_module(f.upper()+f"{i+1}", nn.ReLU(True))
                    elif f.lower() == "sigmoid":
                        self.network.add_module(f.upper()+f"{i+1}", nn.Sigmoid())
                    else:
                        raise NotImplementedError(f"{activation} is not implemented yet!")
        if final_activation_function is not None:
            if final_activation_function.lower() == "softmax":
                activation = nn.Softmax
            elif final_activation_function.lower() == "logsoftmax":
                activation = nn.LogSoftmax
            else:
                raise NotImplementedError(f"{activation} is not implemented yet!")
            self.network.add_module(final_activation_function.upper(), activation(dim=1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)
