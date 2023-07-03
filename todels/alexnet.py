from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from . import MLP


class Alexnet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 dropout_rate: int = 0.5,
                 local_norm_size: int = 5,
                 device: Optional[Union[torch.device, str]] = None):
        """
        Parameters
        ----------
            in_channels: int
                the number of channels input
            num_classes: int
                the number of class for output layer
            dropout_rate: int = 0.5
                rate of each dropout in network
            local_norm_size: int = 5
                the norm size parameter for each normalize in network
            device: Optional[Union[torch.device, str]] = None)
                    target device which you would run on
        """
        super(self.__class__, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4),
            nn.ReLU(True),
            nn.LocalResponseNorm(size=local_norm_size, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.LocalResponseNorm(size=local_norm_size, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ).to(device)
        self.fc = MLP([4096, 4096, num_classes],
                      layers_activation_function = "ReLU",
                      final_activation_function = None,
                      dropout_rate = dropout_rate,
                      first_dropout = True,
                      device=device)

    def forward(self, x):
        out = self.network(x)
        out = self.fc(out)


class LigthAlexnet(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 dropout_rate = 0.5,
                 local_norm_size = 5,
                 device: Optional[Union[torch.device, str]] = None):
        """
        Parameters
        ----------
            in_channels: int
                the number of channels input
            num_classes: int
                the number of class for output layer
            dropout_rate: int = 0.5
                rate of each dropout in network
            local_norm_size: int = 5
                the norm size parameter for each normalize in network
            device: Optional[Union[torch.device, str]] = None)
                    target device which you would run on
        """
        super(self.__class__, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.LocalResponseNorm(size=local_norm_size, alpha=0.0001, beta=0.75, k=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=1),
            nn.ReLU(True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        ).to(device)
        self.fc = MLP([120, 84, num_classes],
                      layers_activation_function = "ReLU",
                      final_activation_function = None,
                      dropout_rate = dropout_rate,
                      first_dropout = True,
                      device=device)

    def forward(self, x):
        out = self.network(x)
        out = self.fc(out)

# TODO: use inheritance for light Alexnet