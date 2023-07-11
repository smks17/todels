from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class CatLayer(nn.Module):
    def __init__(self, dim=0):
        super(self.__class__, self).__init__()
        self.dim = dim

    def forward(self, *args):
        assert len(args) > 0, "Empty args for cat is invalid"
        if len(args) == 1 and isinstance(args[0], list):
            return torch.cat(args[0], dim=self.dim)
        return torch.cat(args, dim=self.dim)


class SumLayer(nn.Module):
    def __init__(self, dim=0):
        super(self.__class__, self).__init__()
        self.dim = dim

    def forward(self, *args):
        assert len(args) > 0, "Empty args for sum is invalid"
        if len(args) == 1 and isinstance(args[0], list):
            return torch.sum(args[0], dim=self.dim)
        return torch.sum(args, dim=self.dim)


class ReshapeLayer(nn.Module):
    def __init__(self, size: int | Tuple, use_view=True):
        super(self.__class__, self).__init__()
        self.size = size
        self.use_view = use_view

    def forward(self, x: torch.Tensor):
        if self.use_view:
            return x.view(self.size)
        return x.reshape(self.size)


class BranchAndCatLayer(nn.Module):
    """Forwarding input to all branches and to concat them at the end"""
    def __init__(self, branches: List[nn.Module] | nn.ModuleList, cat_dim: int = -3):
        super(self.__class__, self).__init__()
        if isinstance(branches, list):
            branches = nn.ModuleList(branches)
        self.branches = branches
        self.cat_layer = CatLayer(cat_dim)

    def forward(self, x: torch.Tensor):
        outs = []
        for branch in self.branches:
            outs.append(branch(x))
        return self.cat_layer(*outs)
