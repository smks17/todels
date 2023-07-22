from __future__ import annotations

from typing import List, Tuple, Iterable

import torch
import torch.nn as nn


class AE(nn.Module):
    """vanilla Autoencoder"""
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module | Iterable[nn.Module]):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(self.decoder, Iterable):
            outputs = []
            for d in self.decoder:
                outputs.append(d(z))
            return outputs
        else:
            return self.decoder(z)

    def forward(self, x) -> torch.Tensor | list[torch.Tensor]:
        return self.decode(self.encode(x))


class VAE(AE):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module | Iterable[nn.Module],
                 latent_size: int):
        super(VAE, self).__init__(encoder, decoder)
        if isinstance(self.decoder, Iterable):
            mean_net = []
            var_net = []
            for i in range(len(self.decoder)):
                mean_net.append(nn.LazyLinear(latent_size))
                var_net.append(nn.LazyLinear(latent_size))
            self.mean_net = mean_net
            self.var_net = var_net
        else:
            self.mean_net = nn.LazyLinear(latent_size)
            self.var_net = nn.LazyLinear(latent_size)

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor | list[torch.Tensor], torch.Tensor | list[torch.Tensor]]:
        z = self.encoder(x)
        return z, self.mean(z), self.var(z)

    def mean(self, z) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(self.mean_net, Iterable):
            outputs = []
            for m in self.mean_net:
                outputs.append(m(z))
            return outputs
        else:
            return self.mean_net(z)

    def var(self, z) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(self.var_net, Iterable):
            outputs = []
            for m in self.var_net:
                outputs.append(m(z))
            return outputs
        else:
            return self.var_net(z)

    def get_representation(self, var, mean) -> torch.Tensor | List[torch.Tensor]:
        if isinstance(self.var_net, Iterable):
            outputs = []
            for i in range(len(self.var_net)):
                std = torch.exp(var[i] * 0.5)
                epsilon = torch.randn_like(std)
                outputs.append(mean[i] + (epsilon * var[i]))
            return outputs
        else:
            std = torch.exp(var * 0.5)
            epsilon = torch.randn_like(std)
            return mean + (epsilon * var)

    def forward(self, x) -> torch.Tensor:
        z, mean, var = self.encode(x)
        if isinstance(self.decoder, Iterable):
            rz = self.get_representation(var, mean)
            outputs = []
            for i in range(len(rz)):
                outputs.append(self.decode(rz[i]))
            return outputs, mean, var
        else:
            return self.decode(self.get_representation(mean, var)), mean, var

    @staticmethod
    def cal_loss(x, recons_loss, mean, var, kld_weight=0.1):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mean ** 2 - var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return loss


class CAE(AE):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module | Iterable[nn.Module]):
        super(CAE, self).__init__(encoder, decoder)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        z = self.sigmoid(self.encoder(x))
        out = self.decoder(z)
        return out, z

    @staticmethod
    def cal_loss(z, abs_loss, encoder_weight, contract_weight=0.01):
        dh = z * (1 - z)
        contractive = torch.sum(dh ** 2 * torch.sum(encoder_weight[0] ** 2), axis=1)
        return abs_loss + contract_weight * contractive

