# data/generators.py
import os
import random
import numpy as np

import torch
import networkx as nx
import pymetis
import torch_geometric
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_networkx

############################################
# 1) Hàm generate_disjoint_data(cfg)
############################################
def generate_disjoint_data(cfg):
    """
    Sinh partition kiểu 'disjoint' cho dữ liệu graph, lưu thành partition_{cid}.pt.

    Tham số cfg (dict) ví dụ:
    {
       'dataset': 'Cora',
       'data_path': './datasets',
       'n_clients': 5,
       'mode': 'disjoint',
       'seed': 1234,
       'ratio_train': 0.2,
       ...
    }

    Kết quả:
    - Lưu partition_{client_id}.pt trong thư mục
      <cfg['data_path']>/<dataset>_disjoint/<n_clients>/
    """
    dataset = cfg['dataset']
    data_path = cfg['data_path']
    n_clients = cfg['n_clients']
    mode = cfg.get('mode', 'disjoint')
    ratio_train = cfg.get('ratio_train', 0.2)
    seed = cfg.get('seed', 1234)

    # Fix seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1) Load data gốc
    data = _get_raw_data(dataset, data_path)

    # 2) Chia train/val/test mask
    data = _split_train(data, ratio_train=ratio_train)

    # 3) Gọi Metis partition => disjoint
    partition_dir = os.path.join(
        data_path,
        f"{dataset}_{mode}",
        str(n_clients)
    )
    os.makedirs(partition_dir, exist_ok=True)

    # Tạo adjacency_list (dạng list of neighbors) cho pymetis
    G_undirected = to_networkx(data, to_undirected=True)
    # (Nếu dataset gốc có self-loops, nên remove self-loops hoặc tuỳ)
    adjacency_list = []
    for i in range(G_undirected.number_of_nodes()):
        adjacency_list.append(list(G_undirected.neighbors(i)))

    # Metis
    n_cuts, membership = pymetis.part_graph(
        adjacency=adjacency_list, 
        nparts=n_clients
    )
    print(f"[generate_disjoint_data] n_partitions={len(set(membership))}, n_cuts={n_cuts}")

    # Dùng adjacency dense => tách subgraph
    # => to_dense_adj: shape [1, N, N]
    dense_adj = to_dense_adj(data.edge_index)[0]

    for client_id in range(n_clients):
        # Lấy danh sách node thuộc partition client_id
        node_indices = np.where(np.array(membership) == client_id)[0]
        node_indices = list(node_indices)

        sub_adj = dense_adj[node_indices][:, node_indices]
        sub_edge_index, _ = dense_to_sparse(sub_adj)
        sub_edge_index = sub_edge_index.cpu()  # (nên đưa về CPU)

        # Tạo Data cho client
        client_data = _build_partition_data(
            data,
            node_indices,
            sub_edge_index
        )

        # Lưu
        path_pt = os.path.join(partition_dir, f"partition_{client_id}.pt")
        torch.save(
            {
              'client_data': client_data,
              'client_id': client_id
            },
            path_pt
        )
        print(f"[generate_disjoint_data] => client_id={client_id}, #nodes={len(node_indices)} => {path_pt}")

    print(f"[generate_disjoint_data] DONE => partitions saved at {partition_dir}")


############################################
# 2) Hàm _get_raw_data(dataset, data_path)
############################################
def _get_raw_data(dataset, data_path):
    """
    Load dataset gốc (Cora, CiteSeer, PubMed, Computers, Photo, ogbn-arxiv).
    Bám sát logic code cũ.
    """
    import torch_geometric.transforms as T

    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        ds = Planetoid(root=data_path, name=dataset, 
                       transform=T.NormalizeFeatures())
        data = ds[0]

    elif dataset in ['Computers', 'Photo']:
        ds = Amazon(
            root=data_path, 
            name=dataset, 
            transform=T.NormalizeFeatures()
        )
        data = ds[0]
        # Tạo sẵn mask (0)
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)

    elif dataset=='ogbn-arxiv':
        ds = PygNodePropPredDataset(name='ogbn-arxiv', root=data_path, 
                                    transform=T.ToUndirected())
        data = ds[0]
        data.y = data.y.view(-1)
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)
    else:
        raise ValueError(f"_get_raw_data: unsupported dataset={dataset}")

    return data


############################################
# 3) Hàm _split_train(data, ratio_train=0.2)
############################################
def _split_train(data, ratio_train=0.2):
    """
    Tạo train/val/test mask ngẫu nhiên:
      - ratio_train
      - ratio_test = ratio_val = (1 - ratio_train)/2
    """
    n_data = data.num_nodes
    ratio_test = (1 - ratio_train) / 2
    n_train = int(n_data * ratio_train)
    n_test  = int(n_data * ratio_test)

    perm = torch.randperm(n_data)
    train_idx = perm[:n_train]
    test_idx  = perm[n_train : n_train+n_test]
    val_idx   = perm[n_train+n_test:]

    data.train_mask[train_idx] = True
    data.test_mask[test_idx]   = True
    data.val_mask[val_idx]     = True

    return data


############################################
# 4) Hàm _build_partition_data(...)
############################################
def _build_partition_data(original_data, node_indices, sub_edge_index):
    """
    Tạo 1 Data() cho client partition: x, y, edge_index, mask, ...
    """
    import torch
    node_indices_t = torch.tensor(node_indices, dtype=torch.long)

    x = original_data.x[node_indices_t]
    y = original_data.y[node_indices_t]

    train_mask = original_data.train_mask[node_indices_t]
    val_mask   = original_data.val_mask[node_indices_t]
    test_mask  = original_data.test_mask[node_indices_t]

    part_data = Data(
        x=x,
        y=y,
        edge_index=sub_edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    return part_data
