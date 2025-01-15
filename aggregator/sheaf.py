# aggregator/sheaf.py

import torch
import torch.nn.functional as F
from aggregator.base import AggregatorBase

# Import mô hình SheafDiffusion + HyperNetwork
# (hoặc tuỳ nơi bạn để HN, có thể import param_generator/hypernetwork)
# Ví dụ ta import GNNHyperNetwork
from param_generator.hypernetwork import GNNHyperNetwork
from models.neural_sheaf.server.disc_models import DiscreteDiagSheafDiffusion

class SheafDiffusionAggregator(AggregatorBase):
    """
    Triển khai aggregator SheafDiffusion + HyperNetwork 
    mô phỏng logic cũ 'train_server_GNN', 'train_server_HN', 'update_server_HN'.
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
        self.updated_embedding = None   # Tương đương 'self.updated_embedding' cũ
        self.eb_tmp = None             # Tương đương 'self.eb_tmp'
        self.gcn_params = None         # Tương đương 'self.gcn_params'
        self.grad_tensor = None        # Tương đương 'self.grad_tensor'

        self.initialized = False       # Đánh dấu đã init model hay chưa

    def _init_sheaf_model(self, server):
        """
        Khởi tạo SheafDiffusion (DiscreteDiagSheafDiffusion) + optimizer GNN
        Giống logic cũ 'train_server_GNN' if curr_rnd == 0.
        """
        if self.initialized:
            return

        edge_index = server.client_graph.edge_index
        self.model = DiscreteDiagSheafDiffusion(edge_index, self.args).cuda(server.gpu_id)

        sheaf_learner_params, other_params = self.model.grouped_parameters()
        import torch
        self.optimizer_gnn = torch.optim.Adam(
            [
                {'params': sheaf_learner_params, 'weight_decay': self.args['server_sheaf_decay']},
                {'params': other_params, 'weight_decay': self.args['server_weight_decay']}
            ],
            lr=self.args['server_lr']
        )
        self.initialized = True

    def _init_hypernetwork(self, server):
        """
        Khởi tạo HyperNetwork (nếu bạn muốn train_server_HN).
        Giống logic cũ 'train_server_HN' if curr_rnd==0.
        """
        if self.model_hn is not None:
            return

        # Số chiều client_graph.x
        feature_dim = server.client_graph.x.shape[1]
        num_clients = self.args['n_clients']
        hidden_dim  = self.args['HN_hidden_dim']
        gcn_layer_dims = self.args.get('gcn_layer_dims', [(128,128),(128,128)])
        hn_dropout  = self.args['hn_dropout']

        self.model_hn = GNNHyperNetwork(num_clients, feature_dim, hidden_dim, gcn_layer_dims, hn_dropout)
        self.model_hn = self.model_hn.cuda(server.gpu_id)

        import torch
        self.optimizer_hn = torch.optim.Adam(self.model_hn.parameters(), lr=self.args['server_hn_lr'])

    def _train_server_GNN(self, server):
        """
        Tương đương logic 'train_server_GNN' cũ:
          1) forward SheafDiffusion => self.updated_embedding
          2) grad_tensor = 0
        """
        self.model.train()
        self.optimizer_gnn.zero_grad()
        x_input = server.client_graph.x
        self.updated_embedding = self.model(x_input)
        # Tương đương: self.grad_tensor = torch.zeros_like(self.updated_embedding)
        # có thể ta chưa set grad_tensor ở đây, 
        # tuỳ logic cũ (chỉ set =0 ban đầu).

    def _train_server_HN(self, server):
        """
        Tương đương logic 'train_server_HN':
         1) forward HN => gcn_params
         2) store param => server.sd[c_id]['generated model params']
        """
        self.model_hn.train()
        self.optimizer_hn.zero_grad()

        # Tạo self.eb_tmp = updated_embedding clone
        if hasattr(self, 'eb_tmp') and self.eb_tmp is not None:
            del self.eb_tmp
        self.eb_tmp = self.updated_embedding.clone()
        self.eb_tmp.requires_grad_(True)
        self.eb_tmp.retain_grad()

        self.gcn_params = self.model_hn(self.eb_tmp)
        
        # Tách param -> clients
        # Giả sử shape [num_clients, total_params]
        # Mỗi client c_id => param
        pointer_idx = 0
        n_clients = self.gcn_params.shape[0]
        for c_id in range(n_clients):
            client_params_tmp = self.gcn_params[c_id, :]
            weights = {}
            pointer = 0
            for i, (in_f, out_f) in enumerate(self.args['gcn_layer_dims']):
                w_size = in_f * out_f
                b_size = out_f

                w_key = f'gcn{i+1}.weight'
                b_key = f'gcn{i+1}.bias'

                weights[w_key] = client_params_tmp[pointer:pointer+w_size].view(in_f, out_f).detach().clone()
                pointer += w_size
                weights[b_key] = client_params_tmp[pointer:pointer+b_size].view(out_f).detach().clone()
                pointer += b_size

            server.sd[c_id] = {'generated model params': weights}

    def _update_server_HN(self, server, local_updates):
        """
        Tương đương logic 'update_server_HN':
          1) gather local delta param => stacked => autograd => 
          2) backward => update self.model_hn + self.model
        """
        # Thu thập local delta
        collected_delta_params = []
        keys_order = ['gcn1.weight','gcn1.bias','gcn2.weight','gcn2.bias']
        for lu in local_updates:
            cid = lu['client_id']
            delta_param = lu['delta']
            flattened_params=[]
            for key in keys_order:
                flattened_params.append(delta_param[key].view(-1))
            delta_gcn_params = torch.cat(flattened_params)
            collected_delta_params.append(delta_gcn_params)

        import torch
        all_delta_params = torch.stack(collected_delta_params, dim=0)  # shape [num_sel, total_params]

        # Tính grad wrt self.gcn_params => self.eb_tmp
        # Giống: 
        #   gnet_grads = torch.autograd.grad(self.gcn_params, self.eb_tmp, grad_outputs=all_delta_params, retain_graph=True)
        #   self.grad_tensor = gnet_grads[0].clone()
        # => update self.model_hn
        self.optimizer_hn.zero_grad()
        gnet_grads = torch.autograd.grad(
            self.gcn_params, 
            self.eb_tmp,
            grad_outputs=all_delta_params,
            retain_graph=True
        )
        self.grad_tensor = gnet_grads[0].clone()

        # average_grads = torch.autograd.grad(...)
        average_grads = torch.autograd.grad(
            self.gcn_params,
            self.model_hn.parameters(),
            grad_outputs=all_delta_params
        )
        # Gán grad
        for p,g in zip(self.model_hn.parameters(), average_grads):
            if p.grad is not None:
                p.grad.zero_()
            if g is not None:
                p.grad = g
        self.optimizer_hn.step()

        # Cuối cùng backward cho SheafDiffusion
        self.optimizer_gnn.zero_grad()
        # torch.autograd.backward(self.updated_embedding, grad_tensors=[self.grad_tensor])
        # Ở code cũ: 
        torch.autograd.backward(self.updated_embedding, grad_tensors=[self.grad_tensor])
        self.optimizer_gnn.step()

    def aggregate(self, local_updates, server):
        """
        Nơi aggregator thực hiện logic 
        1) init sheaf model (lần đầu)
        2) init hypernetwork (lần đầu)
        3) train_server_GNN => self.updated_embedding
        4) train_server_HN => self.gcn_params => gán "generated model params" 
           (Mỗi client c_id: server.sd[c_id] = {...})
        5) client train local => local_updates (delta param)
        6) update_server_HN => map local delta => grad => backward => step
        7) return None (không trả param global)

        'local_updates': List[{'client_id': cid, 'delta': ...}], 
        'server': Server => server.client_graph, .gpu_id, ...
        """
        curr_rnd = getattr(server, 'curr_rnd', 0)  # Lấy round number (nếu cần logic ==0)

        # 1) init sheaf model
        if not self.initialized:
            self._init_sheaf_model(server)

        # 2) init hypernetwork (nếu xài)
        if self.model_hn is None:
            self._init_hypernetwork(server)

        # 3) train_server_GNN (forward SheafDiffusion => updated_embedding)
        self._train_server_GNN(server)

        # 4) train_server_HN => gcn_params => store "generated model params"
        self._train_server_HN(server)

        # 5) aggregator chờ client => local_updates (đã pass param => client => client => local_updates).
        #    Giờ ta collect 'delta param' -> backward
        # 6) update_server_HN => autograd => step
        self._update_server_HN(server, local_updates)

        # 7) aggregator xong => return None
        return None
