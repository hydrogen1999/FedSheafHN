# aggregator/sheaf.py

import random
import torch
import torch.nn.functional as F
from aggregator.base import AggregatorBase

from param_generator.hypernetwork import GNNHyperNetwork
from models.neural_sheaf.server.disc_models import DiscreteDiagSheafDiffusion

class SheafDiffusionAggregator(AggregatorBase):
    """
    Aggregator SheafDiffusion + HyperNetwork.
    Giữ lại logic cũ: 'train_server_GNN', 'train_server_HN', 'update_server_HN'.
    Bổ sung defense_filter(...) để chặn local updates khi attack_frac > 0.0, attack_tau > 0.0.
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # SheafDiffusion model + optimizer
        self.model = None
        self.optimizer_gnn = None

        self.initialized = False

        # Attack/defense config
        self.attack_frac = self.args.get('attack_frac', 0.0)  # Tỉ lệ drop
        self.attack_tau  = self.args.get('attack_tau', 0.0)   # Ngưỡng norm delta
        
    def _init_sheaf_model(self, server):
        """Khởi tạo DiscreteDiagSheafDiffusion + optimizer GNN (train_server_GNN if rnd=0)."""
        if self.initialized:
            return
        edge_index = server.client_graph.edge_index
        self.model = DiscreteDiagSheafDiffusion(edge_index, self.args).cuda(server.gpu_id)

        sheaf_params, other_params = self.model.grouped_parameters()
        self.optimizer_gnn = torch.optim.Adam([
            {'params': sheaf_params,  'weight_decay': self.args['server_sheaf_decay']},
            {'params': other_params,  'weight_decay': self.args['server_weight_decay']}
        ], lr=self.args['server_lr'])

        self.initialized = True

    # def _init_hypernetwork(self, server):
    #     """Khởi tạo HyperNetwork HN model + optimizer."""
    #     if self.model_hn is not None:
    #         return
    #     feature_dim = server.client_graph.x.shape[1]
    #     n_clients   = self.args['n_clients']
    #     hidden_dim  = self.args['HN_hidden_dim']
    #     gcn_layer_dims = self.args.get('gcn_layer_dims', [(128,128),(128,128)])
    #     hn_dropout  = self.args['hn_dropout']

    #     self.model_hn = GNNHyperNetwork(
    #         num_clients=n_clients,
    #         feature_dim=feature_dim,
    #         hidden_dim=hidden_dim,
    #         gcn_layer_dims=gcn_layer_dims,
    #         hn_dropout=hn_dropout
    #     ).cuda(server.gpu_id)

    #     self.optimizer_hn = torch.optim.Adam(
    #         self.model_hn.parameters(), 
    #         lr=self.args['server_hn_lr']
    #     )

    def _train_server_GNN(self, server):
        """
        Tính updated embedding = model(...).
        Lưu vào server.updated_embedding để param_generator có thể dùng.
        """
        self.model.train()
        self.optimizer_gnn.zero_grad()

        x_input = server.client_graph.x
        updated_emb = self.model(x_input)
        # Không backward local updates ở đây (chỉ train forward)
        # => updated_emb chỉ là output để param_generator dùng
        server.updated_embedding = updated_emb

    # def _train_server_HN(self, server):
    #     """Tương tự 'train_server_HN': forward HN => gcn_params => store."""
    #     self.model_hn.train()
    #     self.optimizer_hn.zero_grad()

    #     if self.eb_tmp is not None:
    #         del self.eb_tmp
    #     self.eb_tmp = self.updated_embedding.clone()
    #     self.eb_tmp.requires_grad_(True)
    #     self.eb_tmp.retain_grad()

    #     self.gcn_params = self.model_hn(self.eb_tmp)

    #     # Tách param => server.sd[c_id]['generated model params']
    #     n_clients = self.gcn_params.shape[0]
    #     for c_id in range(n_clients):
    #         client_params_tmp = self.gcn_params[c_id, :]
    #         pointer = 0
    #         weights = {}
    #         for i, (in_f, out_f) in enumerate(self.args['gcn_layer_dims']):
    #             w_size = in_f*out_f
    #             b_size = out_f
    #             w_key  = f"gcn{i+1}.weight"
    #             b_key  = f"gcn{i+1}.bias"

    #             weights[w_key] = client_params_tmp[pointer:pointer+w_size].view(in_f,out_f).detach().clone()
    #             pointer += w_size
    #             weights[b_key] = client_params_tmp[pointer:pointer+b_size].view(out_f).detach().clone()
    #             pointer += b_size

    #         # Gán param cho client c_id
    #         server.sd[c_id] = {'generated model params': weights}

    # def _update_server_HN(self, server, local_updates):
    #     """
    #     Tương đương 'update_server_HN': 
    #      - thu local deltas => stack => autograd => backward => step => update self.model_hn & self.model
    #     """
    #     if not local_updates:
    #         print("[SheafDiffusionAggregator] no local updates => skip update_server_HN.")
    #         return

    #     all_delta_params = []
    #     keys_order = ['gcn1.weight','gcn1.bias','gcn2.weight','gcn2.bias']  # tuỳ theo GCN layer
    #     for upd in local_updates:
    #         delta = upd['delta']
    #         flatten=[]
    #         for k in keys_order:
    #             flatten.append(delta[k].reshape(-1))
    #         cat_ = torch.cat(flatten)
    #         all_delta_params.append(cat_)

    #     import torch
    #     all_delta_params = torch.stack(all_delta_params, dim=0)

    #     # 1) backward HN
    #     self.optimizer_hn.zero_grad()
    #     gnet_grads = torch.autograd.grad(
    #         self.gcn_params, 
    #         self.eb_tmp,
    #         grad_outputs=all_delta_params,
    #         retain_graph=True
    #     )
    #     self.grad_tensor = gnet_grads[0].clone()

    #     # average grads => self.model_hn
    #     average_grads = torch.autograd.grad(
    #         self.gcn_params, 
    #         self.model_hn.parameters(),
    #         grad_outputs=all_delta_params
    #     )
    #     for p,g in zip(self.model_hn.parameters(), average_grads):
    #         if p.grad is not None:
    #             p.grad.zero_()
    #         if g is not None:
    #             p.grad = g
    #     self.optimizer_hn.step()

    #     # 2) backward GNN
    #     self.optimizer_gnn.zero_grad()
    #     torch.autograd.backward(self.updated_embedding, grad_tensors=[self.grad_tensor])
    #     self.optimizer_gnn.step()

    def defense_filter(self, local_updates):
        """
        Lọc local_updates theo các tiêu chí defense, 
        ví dụ: drop ngẫu nhiên (attack_frac) hay drop theo norm (attack_tau).
        """
        if (self.attack_frac <= 0.0) and (self.attack_tau <= 0.0):
            return local_updates  # không làm gì

        safe_list = []
        for upd in local_updates:
            cid   = upd['client_id']
            delta = upd['delta']

            # random drop
            if random.random() < self.attack_frac:
                print(f"[SheafAggregator] drop update from client {cid} (attack_frac).")
                continue

            # norm check
            if self.attack_tau > 0.0:
                norm_val = 0.0
                for _, v in delta.items():
                    norm_val += v.norm(p=2).item()
                if norm_val > self.attack_tau:
                    print(f"[SheafAggregator] drop update from client {cid}, norm={norm_val:.3f} > {self.attack_tau}.")
                    continue

            safe_list.append(upd)

        return safe_list

    def aggregate(self, local_updates, server):
        """
        Aggregator step: 
         1) init sheaf model (nếu chưa)
         2) train_server_GNN => sinh server.updated_embedding
         3) tuỳ ý lọc local_updates (nếu aggregator muốn)
         4) return local_updates (hoặc None)

        Ở đây, ta ví dụ: ta chỉ training GNN (forward) => update embedding, 
        KHÔNG backward với local updates. => local updates tạm để param_generator xử lý.
        """
        if server.client_graph is None:
            print("[SheafDiffusionAggregator] server.client_graph is None => skip aggregator.")
            return local_updates

        self._init_sheaf_model(server)
        self._train_server_GNN(server)

        # (Tuỳ bạn có muốn filter ở đây hay không)
        filtered_updates = self.defense_filter(local_updates)

        return filtered_updates
