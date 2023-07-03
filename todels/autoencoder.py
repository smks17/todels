from typing import List, Tuple

import torch
import torch.nn as nn


class AE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


class MultiAE(nn.Module):
    """Have one input encoder and one or more output"""
    def __init__(self, encoder: nn.Module, decoders: List[nn.Module]):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoders = decoders

    def forward(self, x) -> List[torch.Tensor]:
        outputs = []
        z = self.encoder(x)
        for decoder in self.decoders:
            outputs.append(decoder(z))
        return outputs


class VAE(AE):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_size: int):
        super(VAE, self).__init__(encoder, decoder)
        self.mean_net = nn.LazyLinear(latent_size)
        self.var_net = nn.LazyLinear(latent_size)

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.mean_net(z), self.var_net(z)

    def decode(self, x):
        return self.decoder(x)

    def get_representation(self, var, mean) -> torch.Tensor:
        std = torch.exp(var * 0.5)
        epsilon = torch.randn_like(std)
        return mean + (epsilon * var)

    def forward(self, x) -> torch.Tensor:
        mean, var = self.encode(x)
        return self.decoder(self.get_representation(var, mean)), mean, var

    @staticmethod
    def cal_loss(x, recons_loss, mean, var, kld_weight=0.1):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mean ** 2 - var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return loss
