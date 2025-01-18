# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import numpy as np

from typing import Tuple
from abc import abstractmethod
from torch import nn
import models.neural_sheaf.lib.laplace as lap
# from lib import laplace as lap


class SheafLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.L = None

    def set_L(self, weights):
        self.L = weights.clone().detach()


class LocalConcatSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels, out_shape, sheaf_act='tanh'):
        super().__init__()
        self.out_shape = out_shape
        self.linear1 = nn.Linear(in_channels*2, int(np.prod(out_shape)), bias=False)
        if sheaf_act=='id':
            self.act = lambda x: x
        elif sheaf_act=='tanh':
            self.act = torch.tanh
        elif sheaf_act=='elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, 0, row)
        x_col = torch.index_select(x, 0, col)
        maps = self.linear1(torch.cat([x_row, x_col], dim=1))
        maps = self.act(maps)
        if len(self.out_shape)==2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])


class LocalConcatSheafLearnerVariant(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, d, hidden_channels, out_shape, sheaf_act='tanh'):
        super().__init__()
        self.d = d
        self.hidden_channels = hidden_channels
        self.out_shape = out_shape
        self.linear1 = nn.Linear(hidden_channels*2, int(np.prod(out_shape)), bias=False)

        if sheaf_act=='id':
            self.act = lambda x: x
        elif sheaf_act=='tanh':
            self.act = torch.tanh
        elif sheaf_act=='elu':
            self.act = F.elu
        else:
            raise ValueError(f"Unsupported act {sheaf_act}")

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, 0, row)
        x_col = torch.index_select(x, 0, col)

        x_cat = torch.cat([x_row, x_col], dim=-1)
        x_cat = x_cat.view(-1, self.d, self.hidden_channels*2).sum(dim=1)

        maps = self.linear1(x_cat)
        maps = self.act(maps)

        if len(self.out_shape)==2:
            return maps.view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return maps.view(-1, self.out_shape[0])


class AttentionSheafLearner(SheafLearner):

    def __init__(self, in_channels, d):
        super(AttentionSheafLearner, self).__init__()
        self.d = d
        self.linear1 = torch.nn.Linear(in_channels*2, d**2, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.linear1(torch.cat([x_row, x_col], dim=1)).view(-1, self.d, self.d)

        id = torch.eye(self.d, device=edge_index.device, dtype=maps.dtype).unsqueeze(0)
        return id - torch.softmax(maps, dim=-1)


class EdgeWeightLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, edge_index):
        super(EdgeWeightLearner, self).__init__()
        self.in_channels = in_channels
        self.linear1 = torch.nn.Linear(in_channels*2, 1, bias=False)
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)

    def forward(self, x, edge_index):
        _, full_right_idx = self.full_left_right_idx

        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        weights = self.linear1(torch.cat([x_row, x_col], dim=1))
        weights = torch.sigmoid(weights)

        edge_weights = weights * torch.index_select(weights, index=full_right_idx, dim=0)
        return edge_weights

    def update_edge_index(self, edge_index):
        self.full_left_right_idx, _ = lap.compute_left_right_map_index(edge_index, full_matrix=True)


class QuadraticFormSheafLearner(SheafLearner):
    """Learns a sheaf by concatenating the local node features and passing them through a linear layer + activation."""

    def __init__(self, in_channels: int, out_shape: Tuple[int]):
        super(QuadraticFormSheafLearner, self).__init__()
        assert len(out_shape) in [1, 2]
        self.out_shape = out_shape

        tensor = torch.eye(in_channels).unsqueeze(0).tile(int(np.prod(out_shape)), 1, 1)
        self.tensor = nn.Parameter(tensor)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.map_builder(torch.cat([x_row, x_col], dim=1))

        if len(self.out_shape) == 2:
            return torch.tanh(maps).view(-1, self.out_shape[0], self.out_shape[1])
        else:
            return torch.tanh(maps).view(-1, self.out_shape[0])
