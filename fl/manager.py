# fl/manager.py

import os
import random
import numpy as np
import torch
import time

from fl.server import Server
from fl.client import Client

from aggregator.factory import get_aggregator
from param_generator.factory import get_param_generator
from utils.logger import Logger

class FLManager:
    def __init__(self, args):
        self.args = args
        self.set_seed(args['seed'])

        self.sd = {}
        self.server = Server(self.args, self.sd, gpu_id=0)

        self.clients = {}
        for cid in range(self.args['n_clients']):
            self.clients[cid] = Client(self.args, 0, self.sd, cid)

        self.aggregator = get_aggregator(args)           # SheafDiffusionAggregator, etc.
        self.param_gen  = get_param_generator(args)      # HyperNetworkParamGen, etc.

        os.makedirs(self.args['checkpt_path'], exist_ok=True)
        os.makedirs(self.args['log_path'], exist_ok=True)

        self.n_connected = max(1, int(self.args['n_clients'] * self.args['frac']))
        self.logger = Logger(self.args, 0, is_server=True)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True

    def start(self):
        n_rnds = self.args['n_rnds']
        for rnd in range(n_rnds):
            self.server.on_round_begin(rnd)

            # Chọn client
            selected = sorted(np.random.choice(self.args['n_clients'], self.n_connected, replace=False))

            # ROUND 0: Mỗi client generate vector => server.construct_graph
            if rnd == 0:
                for cid in range(self.args['n_clients']):
                    self.clients[cid].switch_state(cid)
                    self.clients[cid].on_receive_message(rnd, 'client_generate_vector_start')
                    self.clients[cid].generate_vector(rnd)
                self.server.construct_graph(range(self.args['n_clients']))

            # 1) aggregator => train_server_GNN => server.updated_embedding
            #    (Ở đây aggregator chưa dùng local_updates, 
            #     vì client chưa train => chưa có delta)
            #    => aggregator.aggregate(...) 
            dummy_updates = []  # chưa có local updates
            self.aggregator.aggregate(dummy_updates, self.server)

            # 2) param_gen => dùng server.updated_embedding => generate param => store ...
            if self.param_gen is not None and self.server.updated_embedding is not None:
                self.param_gen.prepare_params(self.server, self.clients, selected)

            # 3) Client train => local_updates
            local_updates = []
            for cid in selected:
                self.clients[cid].on_receive_message(rnd, 'client_train_on_generated_model_prams')
                # update_client_embedding = True if rnd>0 and rnd%5==0 else False
                # tuỳ logic
                update_client_embedding = (rnd>0 and rnd%5==0)
                self.clients[cid].train_client_model(update_client_embedding=update_client_embedding)

                if 'delta' in self.sd[cid]:
                    local_updates.append({'client_id': cid, 'delta': self.sd[cid].pop('delta')})

            # 4) aggregator => (tùy) filter local_updates 
            safe_updates = self.aggregator.defense_filter(local_updates)

            # 5) param_gen => backprop hypernetwork => update 
            if self.param_gen is not None:
                self.param_gen.backprop_hn(self.server, safe_updates)

            # 6) Nếu có update embedding => reconstruct graph 
            #    (vd mỗi 5 rounds)
            if rnd>0 and rnd%5==0:
                self.server.construct_graph(range(self.args['n_clients']))

            # 7) round_end => logging
            self.server.round_end(rnd, range(self.args['n_clients']), selected)

            print(f"[FLManager] Round {rnd} done.\n")

        print("[FLManager] Finished training.")
        return
