# fl/server.py

import torch
import time
from torch_geometric.data import Data

from utils.logger import Logger
from utils.torch_utils import torch_save, torch_load, get_state_dict, set_state_dict

class Server:
    def __init__(self, args, shared_dict, gpu_id):
        self.args = args
        self.sd = shared_dict
        self.gpu_id = gpu_id

        self.logger = Logger(self.args, gpu_id, is_server=True)

        self.updated_embedding = None
        self.gcn_params = None
        self.grad_tensor = None

        # Bổ sung 1 dictionary cho log 
        self.log = {'total_val_acc':[], 'total_test_acc':[]}

        # placeholders => model sheaf diffusion, HNmodel => nay do aggregator/paramgen handle
        self.model = None  
        self.model_hn = None
        self.client_graph = None

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd

    def construct_graph(self, updated_client_ids, curr_rnd):
        # build client graph
        st = time.time()
        client_embeddings = []
        for cid in updated_client_ids:
            emb = self.sd[cid].pop('functional_embedding')
            client_embeddings.append(emb)
        embeddings = torch.cat(client_embeddings, dim=0).cuda(self.gpu_id)

        # Xây edges fully connected
        edges = []
        n = len(updated_client_ids)
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append([i,j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().cuda(self.gpu_id)

        self.client_graph = Data(x=embeddings, edge_index=edge_index).cuda(self.gpu_id)

        # set input_dim, etc. (nếu SheafDiffusion model cần)
        self.args['graph_size'] = self.client_graph.x.size(0)
        self.args['input_dim']  = self.client_graph.num_features
        self.args['output_dim'] = self.client_graph.num_features

        self.logger.print(f"client graph constructed in {time.time()-st:.2f}s")

    def round_end(self, curr_rnd, all_clients, selected):
        val_acc, test_acc = [], []
        for cid in all_clients:
            if cid in selected:
                val_acc.append(self.sd[cid]['val_acc'])
                test_acc.append(self.sd[cid]['test_acc'])
            # Xoá key cũ
            self.sd[cid].pop('val_acc',None)
            self.sd[cid].pop('test_acc',None)

        self.log['total_val_acc'].append(sum(val_acc)/len(val_acc))
        self.log['total_test_acc'].append(sum(test_acc)/len(test_acc))
        print(f"[server] Round {curr_rnd} test_acc = {self.log['total_test_acc'][-1]}")

    def save_state(self):
        out = {
            'log': self.log
        }
        torch_save(self.args['checkpt_path'], 'server_state.pt', out)

    def load_state(self):
        loaded = torch_load(self.args['checkpt_path'], 'server_state.pt')
        self.log = loaded['log']

    # Tuỳ bạn, có thể bổ sung train_server_GNN, train_server_HN,... 
    # HOẶC chuyển logic sang aggregator/param_generator
