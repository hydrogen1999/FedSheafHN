# aggregator/sheaf.py

import torch
import torch.nn.functional as F
from aggregator.base import AggregatorBase

# Import mô hình SheafDiffusion + HyperNetwork
from param_generator.hypernetwork import GNNHyperNetwork
from models.neural_sheaf.server.disc_models import DiscreteDiagSheafDiffusion

import random

class SheafDiffusionAggregator(AggregatorBase):
    """
    Triển khai aggregator SheafDiffusion + HyperNetwork 
    mô phỏng logic cũ: train_server_GNN, train_server_HN, update_server_HN.
    Ngoài ra, có thêm defense_filter() để "chặn" local updates bị nghi ngờ tấn công.
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # SheafDiffusion model + optimizer
        self.model = None
        self.optimizer_gnn = None

        # HyperNetwork + optimizer
        self.model_hn = None
        self.optimizer_hn = None

        # Tạm giữ intermediate
        self.updated_embedding = None
        self.eb_tmp = None
        self.gcn_params = None
        self.grad_tensor = None

        self.initialized = False

        # Thêm config defense/attack
        self.attack_frac = self.args.get('attack_frac', 0.0)  # Tỉ lệ drop random
        self.attack_tau  = self.args.get('attack_tau', 1.0)  # Ngưỡng norm delta

    def _init_sheaf_model(self, server):
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

    def _init_hypernetwork(self, server):
        if self.model_hn is not None:
            return
        feature_dim = server.client_graph.x.shape[1]
        n_clients   = self.args['n_clients']
        hidden_dim  = self.args['HN_hidden_dim']
        gcn_layer_dims = self.args.get('gcn_layer_dims', [(128,128),(128,128)])
        hn_dropout  = self.args['hn_dropout']

        self.model_hn = GNNHyperNetwork(n_clients, feature_dim, hidden_dim, gcn_layer_dims, hn_dropout)
        self.model_hn = self.model_hn.cuda(server.gpu_id)
        self.optimizer_hn = torch.optim.Adam(self.model_hn.parameters(), lr=self.args['server_hn_lr'])

    def _train_server_GNN(self, server):
        self.model.train()
        self.optimizer_gnn.zero_grad()
        x_input = server.client_graph.x
        self.updated_embedding = self.model(x_input)

    def _train_server_HN(self, server):
        self.model_hn.train()
        self.optimizer_hn.zero_grad()

        if hasattr(self, 'eb_tmp') and self.eb_tmp is not None:
            del self.eb_tmp
        self.eb_tmp = self.updated_embedding.clone()
        self.eb_tmp.requires_grad_(True)
        self.eb_tmp.retain_grad()

        self.gcn_params = self.model_hn(self.eb_tmp)

        # Tách param => 'generated model params' cho từng client
        n_clients = self.gcn_params.shape[0]
        for c_id in range(n_clients):
            client_params_tmp = self.gcn_params[c_id, :]
            pointer = 0
            weights = {}
            for i, (in_f, out_f) in enumerate(self.args['gcn_layer_dims']):
                w_size = in_f*out_f
                b_size = out_f
                w_key  = f"gcn{i+1}.weight"
                b_key  = f"gcn{i+1}.bias"
                weights[w_key] = client_params_tmp[pointer:pointer+w_size].view(in_f,out_f).detach().clone()
                pointer += w_size
                weights[b_key] = client_params_tmp[pointer:pointer+b_size].view(out_f).detach().clone()
                pointer += b_size

            server.sd[c_id] = {'generated model params': weights}

    def _update_server_HN(self, server, local_updates):
        """
        Tương đương update_server_HN: 
        1) Lấy local delta => stack => grad => backward => step
        2) step self.model_hn, self.model
        """
        if len(local_updates)==0:
            print("[sheaf aggregator] no local updates => skip")
            return

        all_delta_params = []
        keys_order = ['gcn1.weight','gcn1.bias','gcn2.weight','gcn2.bias']  # or tuỳ
        for upd in local_updates:
            delta = upd['delta']
            flatten=[]
            for k in keys_order:
                flatten.append(delta[k].view(-1))
            cat_ = torch.cat(flatten)
            all_delta_params.append(cat_)
        all_delta_params = torch.stack(all_delta_params, dim=0)

        # Tính grad => updated_embedding => model_hn
        self.optimizer_hn.zero_grad()
        gnet_grads = torch.autograd.grad(
            self.gcn_params, 
            self.eb_tmp, 
            grad_outputs=all_delta_params,
            retain_graph=True
        )
        self.grad_tensor = gnet_grads[0].clone()

        average_grads = torch.autograd.grad(
            self.gcn_params,
            self.model_hn.parameters(),
            grad_outputs=all_delta_params
        )
        for p,g in zip(self.model_hn.parameters(), average_grads):
            if p.grad is not None:
                p.grad.zero_()
            if g is not None:
                p.grad = g
        self.optimizer_hn.step()

        # Cuối cùng backward SheafDiffusion
        self.optimizer_gnn.zero_grad()
        torch.autograd.backward(self.updated_embedding, grad_tensors=[self.grad_tensor])
        self.optimizer_gnn.step()

    ### Thêm hàm defense_filter
    def defense_filter(self, local_updates):
        """
        Lọc hoặc drop local updates nếu 
        1) random < attack_frac
        2) norm(delta) > attack_tau
        """
        if self.attack_frac<=0.0 and self.attack_tau<=0.0:
            # Không làm gì
            return local_updates

        filtered = []
        import random
        for upd in local_updates:
            cid = upd['client_id']
            delta = upd['delta']

            # random drop
            if random.random() < self.attack_frac:
                print(f"[sheaf aggregator] drop update from client {cid} (attack_frac).")
                continue

            # check norm delta
            norm_val = 0.0
            for k,v in delta.items():
                norm_val += v.norm(p=2).item()
            if norm_val > self.attack_tau:
                print(f"[sheaf aggregator] drop update from client {cid}, delta norm {norm_val:.2f} > {self.attack_tau}.")
                continue

            filtered.append(upd)
        return filtered

    def aggregate(self, local_updates, server):
        """
        aggregator: 
          1) init sheaf model & HN
          2) train_server_GNN => updated_embedding
          3) train_server_HN => gcn_params => store 'generated model params'
          4) client train => local_updates
          5) defense_filter => local_updates => filtered
          6) update_server_HN => backprop => step
        """
        if not self.initialized:
            self._init_sheaf_model(server)
        if self.model_hn is None:
            self._init_hypernetwork(server)

        # train_server_GNN
        self._train_server_GNN(server)
        # train_server_HN
        self._train_server_HN(server)

        # defense filter
        safe_updates = self.defense_filter(local_updates)

        # update_server_HN
        self._update_server_HN(server, safe_updates)

        return None
