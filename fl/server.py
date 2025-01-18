# fl/server.py
import time
import torch
from torch_geometric.data import Data
from utils.logger import Logger
from utils.torch_utils import torch_save, torch_load

class Server:
    def __init__(self, args, shared_dict, gpu_id):
        self.args = args
        self.sd = shared_dict
        self.gpu_id = gpu_id

        self.logger = Logger(self.args, gpu_id, is_server=True)

        # Nơi lưu client_graph, updated_embedding, ...
        self.client_graph = None
        self.updated_embedding = None

        # Log
        self.log = {'total_val_acc':[], 'total_test_acc':[]}

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd

    def construct_graph(self, updated_client_ids, curr_rnd):
        """
        Xây graph fully-connected giữa client embeddings => server.client_graph.
        (Mỗi client phải có functional_embedding trong self.sd[client_id].)
        """
        start = time.time()
        embeddings = []
        for cid in client_ids:
            if 'functional_embedding' in self.sd[cid]:
                emb = self.sd[cid].pop('functional_embedding')
                embeddings.append(emb)
            else:
                # fallback: zero
                emb = torch.zeros(1, self.args['input_dim'])
                embeddings.append(emb)
                
        X = torch.cat(embeddings, dim=0).cuda(self.gpu_id)
        n = len(client_ids)
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append([i,j])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().cuda(self.gpu_id)

        self.client_graph = Data(x=X, edge_index=edge_index)
        self.client_graph = self.client_graph.cuda(self.gpu_id)

        self.args['graph_size'] = self.client_graph.num_nodes
        self.args['input_dim']  = self.client_graph.num_features
        self.args['output_dim'] = self.args['input_dim']

        self.logger.print(f"Constructed client graph with {n} nodes in {time.time()-start:.2f}s")

    def round_end(self, rnd, all_cids, selected):
        # Tính avg val_acc, test_acc
        val_acc_list = []
        test_acc_list= []
        for cid in all_cids:
            if 'val_acc' in self.sd[cid]:
                val_acc_list.append(self.sd[cid]['val_acc'])
                self.sd[cid].pop('val_acc', None)
            if 'test_acc' in self.sd[cid]:
                test_acc_list.append(self.sd[cid]['test_acc'])
                self.sd[cid].pop('test_acc', None)

        if len(val_acc_list)>0:
            mean_val = sum(val_acc_list)/len(val_acc_list)
            self.log['total_val_acc'].append(mean_val)
        else:
            mean_val = 0.0

        if len(test_acc_list)>0:
            mean_test= sum(test_acc_list)/len(test_acc_list)
            self.log['total_test_acc'].append(mean_test)
        else:
            mean_test= 0.0
        self.logger.print(f"Round {rnd} => val_acc={mean_val:.3f}, test_acc={mean_test:.3f}")

    def save_state(self):
        out = {'log': self.log}
        torch_save(self.args['checkpt_path'], 'server_state.pt', out)

    def load_state(self):
        loaded = torch_load(self.args['checkpt_path'], 'server_state.pt')
        self.log = loaded['log']
