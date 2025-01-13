# param_generator/hypernetwork.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from param_generator.base import ParamGenBase

class GNNHyperNetwork(nn.Module):
    """
    Bản gốc HNmodel.
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

        total_params = sum(in_f*out_f + out_f for in_f, out_f in gcn_layer_dims)
        self.param_generator = nn.Linear(hidden_dim, total_params)

    def forward(self, features):
        # features shape [num_clients, feature_dim]
        weighted_embeddings = torch.matmul(self.attention_matrix, features)
        weighted_embeddings = self.dropout(weighted_embeddings)

        x = self.relu(self.layer1(weighted_embeddings))
        x = F.dropout(x, p=self.hn_dropout, training=self.training)
        x = self.relu(self.layer2(x))
        x = F.dropout(x, p=self.hn_dropout, training=self.training)

        gcn_params = self.param_generator(x)
        return gcn_params


class HyperNetworkParamGen(ParamGenBase):
    """
    Lớp Param Generator, bọc GNNHyperNetwork
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.num_clients = args['n_clients']
        self.hidden_dim  = args['HN_hidden_dim']
        self.feature_dim = None   # sẽ xác định sau
        self.hn_dropout  = args['hn_dropout']
        self.gcn_layer_dims = args.get('gcn_layer_dims', [(1433,128),(128,128)])

        # Tạo model HN
        # feature_dim = client_graph.num_features => ta sẽ set động
        self.hn_model = None
        self.optimizer_hn = None
        self.updated_embedding = None
        self.gcn_params = None

    def init_hn(self, feature_dim):
        self.feature_dim = feature_dim
        self.hn_model = GNNHyperNetwork(
            num_clients=self.num_clients,
            feature_dim=feature_dim,
            hidden_dim=self.hidden_dim,
            gcn_layer_dims=self.gcn_layer_dims,
            hn_dropout=self.hn_dropout
        ).cuda()

        import torch
        self.optimizer_hn = torch.optim.Adam(self.hn_model.parameters(), lr=self.args['server_hn_lr'])

    def prepare_params(self, server, clients, selected_client_ids):
        """
        Gọi khi server có updated_embedding => pass vào HN => generate gcn params
        Lưu param vào server.sd[client_id]['generated model params']
        """
        if server.updated_embedding is None:
            return

        if self.hn_model is None:
            # lần đầu, init HN
            self.init_hn(feature_dim=server.updated_embedding.shape[1])

        self.hn_model.train()
        self.optimizer_hn.zero_grad()

        eb_tmp = server.updated_embedding.detach().clone()
        eb_tmp.requires_grad_(True)
        # forward
        gcn_params = self.hn_model(eb_tmp)
        server.gcn_params = gcn_params  # server lưu tạm

        # Tách param => client
        for i, cid in enumerate(selected_client_ids):
            client_params_tmp = gcn_params[i, :]  # shape [sum(in_f*out_f + out_f)]
            pointer = 0
            weights = {}
            for idx, (in_f, out_f) in enumerate(self.gcn_layer_dims):
                weight_size = in_f * out_f
                bias_size   = out_f
                w_key = f"gcn{idx+1}.weight"
                b_key = f"gcn{idx+1}.bias"
                weights[w_key] = client_params_tmp[pointer:pointer+weight_size].view(in_f, out_f)
                pointer += weight_size
                weights[b_key] = client_params_tmp[pointer:pointer+bias_size].view(out_f)
                pointer += bias_size

            server.sd[cid] = {'generated model params': {k:v.clone().detach() for k,v in weights.items()}}


    def backprop_hn(self, server, updated_client_ids):
        """
        Nhận delta param => autograd => update self.hn_model
        """
        import torch
        # Thu thập local delta
        collected_delta_params = []
        keys_order = ['gcn1.weight','gcn1.bias','gcn2.weight','gcn2.bias']
        for cid in updated_client_ids:
            delta_param = server.sd[cid].pop('delta param')
            flattened_params = []
            for key in keys_order:
                flattened_params.append(delta_param[key].reshape(-1))
            delta_gcn_params = torch.cat(flattened_params, dim=0)
            collected_delta_params.append(delta_gcn_params)

        # Xếp thành tensor
        all_delta_params = torch.stack(collected_delta_params, dim=0)  # shape [num_sel, total_params]

        # Tính grad wrt self.hn_model
        server.grad_tensor = torch.autograd.grad(
            server.gcn_params, server.updated_embedding,
            grad_outputs=all_delta_params, retain_graph=True
        )[0].clone()

        # zero grad
        self.optimizer_hn.zero_grad()
        average_grads = torch.autograd.grad(
            server.gcn_params, self.hn_model.parameters(),
            grad_outputs=all_delta_params
        )

        for p, g in zip(self.hn_model.parameters(), average_grads):
            if p.grad is not None:
                p.grad.zero_()
            if g is not None:
                p.grad = g

        self.optimizer_hn.step()
        collected_delta_params.clear()
