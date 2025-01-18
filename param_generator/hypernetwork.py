# param_generator/hypernetwork.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from param_generator.base import ParamGenBase

class GNNHyperNetwork(nn.Module):
    """
    HyperNetwork gốc, sinh param GCN cho client.
    """
    def __init__(self, num_clients, feature_dim, hidden_dim, gcn_layer_dims, hn_dropout):
        super().__init__()
        self.num_clients = num_clients
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.gcn_layer_dims = gcn_layer_dims
        self.hn_dropout = hn_dropout

        self.attention_matrix = nn.Parameter(torch.eye(num_clients))
        self.dropout = nn.Dropout(p=self.hn_dropout)

        self.layer1 = nn.Linear(feature_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        total_params = sum(in_f*out_f + out_f for in_f,out_f in gcn_layer_dims)
        self.param_generator = nn.Linear(hidden_dim, total_params)

    def forward(self, features):
        # features shape: [num_clients, feature_dim]
        weighted = torch.matmul(self.attention_matrix, features)
        weighted = self.dropout(weighted)

        x = self.relu(self.layer1(weighted))
        x = F.dropout(x, p=self.hn_dropout, training=self.training)
        x = self.relu(self.layer2(x))
        x = F.dropout(x, p=self.hn_dropout, training=self.training)

        gcn_params = self.param_generator(x) # [num_clients, total_params]
        return gcn_params


class HyperNetworkParamGen(ParamGenBase):
    """
    Lớp Param Generator, bọc GNNHyperNetwork
    """
    def __init__(self, args):
        super().__init__(args)
        self.num_clients = args['n_clients']
        self.hidden_dim  = args['HN_hidden_dim']
        self.hn_dropout  = args['hn_dropout']
        self.gcn_layer_dims = args.get('gcn_layer_dims', [(128,128),(128,128)])

        self.hn_model = None
        self.optimizer_hn = None
        self.feature_dim  = None

        self.server_gcn_params = None

    def init_hn(self, feature_dim):
        self.feature_dim = feature_dim
        self.hn_model = GNNHyperNetwork(
            num_clients=self.num_clients,
            feature_dim=feature_dim,
            hidden_dim=self.hidden_dim,
            gcn_layer_dims=self.gcn_layer_dims,
            hn_dropout=self.hn_dropout
        ).cuda()

        self.optimizer_hn = torch.optim.Adam(
            self.hn_model.parameters(),
            lr=self.args['server_hn_lr']
        )
        
    def prepare_params(self, server, clients, selected_client_ids):
        """
        Dựa trên server.updated_embedding => forward HN => 
        store param cho mỗi client.
        """
        if server.updated_embedding is None:
            return
        # Lần đầu
        if self.hn_model is None:
            feat_dim = server.updated_embedding.shape[1]
            self.init_hn(feat_dim)

        # Forward HN
        self.hn_model.train()
        self.optimizer_hn.zero_grad()

        emb = server.updated_embedding.detach().clone() # [num_clients, feature_dim]
        emb.requires_grad_(True)

        gcn_params = self.hn_model(emb) # [num_clients, total_params]
        self.server_gcn_params = gcn_params
        # Gán param cho các client được chọn
        for i, cid in enumerate(selected_client_ids):
            param_vec = gcn_params[i,:]
            pointer = 0
            weights = {}
            for idx, (in_f, out_f) in enumerate(self.gcn_layer_dims):
                w_size = in_f*out_f
                b_size = out_f
                w_key = f"gcn{idx+1}.weight"
                b_key = f"gcn{idx+1}.bias"
                w_val = param_vec[pointer:pointer+w_size].view(in_f, out_f)
                pointer += w_size
                b_val = param_vec[pointer:pointer+b_size].view(out_f)
                pointer += b_size
                weights[w_key] = w_val.detach().clone()
                weights[b_key] = b_val.detach().clone()

            server.sd[cid] = {'generated model params': weights}

    def backprop_hn(self, server, local_updates):
        """
        Lấy delta param (all_delta) từ local_updates => autograd => update HN.
        => Giúp HN học sinh param "tốt hơn" vòng sau.

        local_updates: list, mỗi phần tử {'client_id': cid, 'delta': dict_param_delta}.
        """
        # Nếu không có update hoặc chưa forward HN => skip
        if not local_updates or self.server_gcn_params is None:
            return

        # Thu local delta => stack
        collected = []
        # Tạo keys_order dynamic dựa trên self.gcn_layer_dims
        keys_order = []
        for i in range(len(self.gcn_layer_dims)):
            keys_order.append(f'gcn{i+1}.weight')
            keys_order.append(f'gcn{i+1}.bias')

        # Ghép delta từng client => 1 tensor shape [num_sel, total_params]
        for upd in local_updates:
            delta_dict = upd['delta']
            flat_list = []
            for k in keys_order:
                flat_list.append(delta_dict[k].view(-1))
            cat_ = torch.cat(flat_list, dim=0)
            collected.append(cat_)

        all_delta = torch.stack(collected, dim=0)  # [num_sel, total_params]

        # Backprop HN => d(server_gcn_params)/d(hn_model.parameters)
        self.optimizer_hn.zero_grad()

        # Tính gradient w.r.t. self.hn_model.parameters
        # grad_outputs=all_delta => chain rule
        grads_hn = torch.autograd.grad(
            self.server_gcn_params,           # [num_clients, total_params]
            self.hn_model.parameters(),       # Param HN
            grad_outputs=all_delta,           # [num_sel, total_params]
            retain_graph=False
        )

        # Gán grad => update
        for p, g in zip(self.hn_model.parameters(), grads_hn):
            if p.grad is not None:
                p.grad.zero_()
            if g is not None:
                p.grad = g

        self.optimizer_hn.step()

        # Clear
        self.server_gcn_params = None
