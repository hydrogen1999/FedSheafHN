# data/loader.py

import os
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from utils.torch_utils import torch_load  # Chuyển từ misc.utils sang utils.torch_utils

def get_data(args, client_id):
    """
    Hàm load dữ liệu partition cho 1 client, 
    giống code cũ: 
        torch_load(args.data_path, f'{args.dataset}_{args.mode}/{args.n_clients}/partition_{client_id}.pt')
    """
    # Xây dựng path
    base_dir = os.path.join(args['data_path'], f"{args['dataset']}_{args['mode']}", str(args['n_clients']))
    fname = f"partition_{client_id}.pt"
    loaded = torch_load(base_dir, fname)  # => dict{'client_data': ...}
    return [loaded['client_data']]       # Trả về list

class FedDataLoader:
    """
    Thay cho class DataLoader cũ. 
    """
    def __init__(self, args):
        self.args = args
        self.n_workers = 1
        self.client_id = None
        self.DataLoaderClass = PyGDataLoader
        self.pa_loader = None  # PyG DataLoader

    def switch(self, client_id):
        """Chuyển sang client_id, load partition => tạo DataLoader."""
        if self.client_id == client_id:
            return  # Không cần load lại
        self.client_id = client_id

        self.partition = get_data(self.args, client_id)
        self.pa_loader = self.DataLoaderClass(
            dataset=self.partition,
            batch_size=1,
            shuffle=False,
            num_workers=self.n_workers,
            pin_memory=False
        )

    def get_loader(self):
        return self.pa_loader
