# data/loader.py

import os
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from utils.torch_utils import torch_load  # Chuyển từ misc.utils sang utils.torch_utils

def get_data(args, client_id):
    """
    Load dữ liệu partition cho 1 client:
    file:  <args['data_path']>/<dataset>_<mode>/<n_clients>/partition_{client_id}.pt
    """
    base_dir = os.path.join(
        args['data_path'], 
        f"{args['dataset']}_{args['mode']}", 
        str(args['n_clients'])
    )
    fname = f"partition_{client_id}.pt"
    loaded = torch_load(base_dir, fname)  # => {'client_data': ...}
    return [loaded['client_data']]

class FedDataLoader:
    """Dataloader cho mỗi client trong mô hình Federated Learning."""
    def __init__(self, args):
        self.args = args
        self.n_workers = 1
        self.client_id = None
        self.pa_loader = None

    def switch(self, client_id):
        if self.client_id == client_id:
            return
        self.client_id = client_id

        partition = get_data(self.args, client_id)
        self.pa_loader = PyGDataLoader(
            dataset=partition,
            batch_size=1,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=False
        )

    def get_loader(self):
        return self.pa_loader
