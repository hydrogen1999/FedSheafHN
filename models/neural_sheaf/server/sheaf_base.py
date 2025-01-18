# models/neural_sheaf/server/sheaf_base.py

import torch
from torch import nn


class SheafDiffusion(nn.Module):
    """
    Base class cho SheafDiffusion model. 
    """
    def __init__(self, edge_index, args):
        super().__init__()
        self.d = args['server_d']
        self.edge_index = edge_index
        self.add_lp = args['add_lp']
        self.add_hp = args['add_hp']

        self.final_d = self.d
        if self.add_hp:
            self.final_d += 1
        if self.add_lp:
            self.final_d += 1

        self.hidden_dim = args['server_hidden_channels'] * self.final_d
        self.layers = args['server_layers']
        self.normalised = args['server_normalised']
        self.deg_normalised = args['server_deg_normalised']
        self.nonlinear = (not args['server_linear'])
        self.input_dropout = args['server_input_dropout']
        self.dropout = args['server_dropout']
        self.left_weights = args['left_weights']
        self.right_weights= args['right_weights']
        self.sparse_learner= args['sparse_learner']
        self.use_act = args['server_use_act']
        self.sheaf_act = args['server_sheaf_act']
        self.second_linear = args['server_second_linear']
        self.orth = args['orth']
        self.edge_weights = args['edge_weights']
        self.t = args['max_t']

        self.laplacian_builder = None

        self.hidden_channels = args['server_hidden_channels']

    def update_edge_index(self, edge_index):
        if self.laplacian_builder is not None:
            self.laplacian_builder = self.laplacian_builder.create_with_new_edge_index(edge_index)

    def grouped_parameters(self):
        sheaf_learner, others = [], []
        for name, param in self.named_parameters():
            if "sheaf_learner" in name:
                sheaf_learner.append(param)
            else:
                others.append(param)
        return sheaf_learner, others