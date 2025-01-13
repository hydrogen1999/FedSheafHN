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

        self.aggregator = get_aggregator(args)           # SheafDiffusionAggregator, etc.
        self.param_gen  = get_param_generator(args)      # HyperNetworkParamGen, etc.

        self.server = Server(args, self.sd, gpu_id=0)
        self.clients={}
        for cid in range(args['n_clients']):
            self.clients[cid] = Client(args, gpu_id=0, shared_dict=self.sd, client_id=cid)

        self.logger = Logger(args, 0, is_server=True)

        os.makedirs(args['checkpt_path'], exist_ok=True)
        os.makedirs(args['log_path'], exist_ok=True)
        self.n_connected = max(1, int(args['n_clients']*args['frac']))

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED']=str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True

    def start(self):
        n_rnds = self.args['n_rnds']
        for rnd in range(n_rnds):
            selected = sorted(np.random.choice(self.args['n_clients'], self.n_connected, replace=False))
            st = time.time()

            self.server.on_round_begin(rnd)

            # Round 0 => generate vector
            if rnd==0:
                for cid in range(self.args['n_clients']):
                    self.clients[cid].switch_state(cid)
                    self.clients[cid].on_receive_message(rnd, 'client_generate_vector_start')
                    self.clients[cid].generate_vector(cid)
                self.server.construct_graph(range(self.args['n_clients']))

            # param_gen => prepare param => ...
            if self.param_gen is not None:
                self.param_gen.prepare_params(self.server, self.clients, selected)

            # client train
            local_updates=[]
            for cid in selected:
                self.clients[cid].on_receive_message(rnd, 'client_train_on_generated_model_prams')
                self.clients[cid].train_client_model(update_client_embedding=(rnd%5==0 and rnd>0))
                # aggregator feed local_updates
                delta = self.sd[cid]['delta param']
                local_updates.append({'client_id': cid, 'delta': delta})

            # aggregator => aggregate
            if self.aggregator is not None:
                # aggregator cài logic train_server_GNN
                aggregated_params = self.aggregator.aggregate(local_updates, self.server)
                # aggregator “SheafDiffusionAggregator” => có thể None

            # reconstruct graph => if update embedding
            if rnd%5==0 and rnd>0:
                self.server.construct_graph(range(self.args['n_clients']))

            self.server.round_end(rnd, range(self.args['n_clients']), selected)
            print(f"[FLManager] Round {rnd} done in {time.time()-st:.2f}s")

        print("[FLManager] Finished training.")
        return
