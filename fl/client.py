# fl/client.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import time

from data.loader import FedDataLoader
from utils.logger import Logger
from utils.torch_utils import torch_save, torch_load, get_state_dict, set_state_dict

# Ví dụ import model GCN
from models.gcn import GCN

class Client:
    def __init__(self, args, gpu_id, shared_dict, client_id):
        """
        args: dict config
        gpu_id: int
        shared_dict: dict dùng chia sẻ (sd)
        client_id: int
        """
        self.args = args
        self.gpu_id = gpu_id
        self.sd = shared_dict
        self.client_id = client_id

        self.loader = FedDataLoader(self.args)
        self.logger = Logger(self.args, gpu_id=self.gpu_id, is_server=False)

        # Khởi tạo GCN (mặc định)
        # Thay cho self.model = GCN ...
        self.model_class = GCN
        self.model = None
        self.optimizer = None
        self.curr_rnd = 0

        # Xác định số classes
        if args['dataset'] == 'Cora':
            self.args['n_classes'] = 7
        elif args['dataset'] == 'CiteSeer':
            self.args['n_classes'] = 6
        elif args['dataset'] == 'PubMed':
            self.args['n_classes'] = 3
        # ... v.v.

    def switch_state(self, client_id):
        """Hàm giống logic cũ, load hoặc init state nếu chưa có."""
        self.client_id = client_id
        self.loader.switch(client_id)
        self.logger.switch(client_id)

        # Kiểm tra file state
        ckpt_path = os.path.join(self.args['checkpt_path'], f"{client_id}_state.pt")
        if os.path.exists(ckpt_path):
            time.sleep(0.1)
            self.load_state()
        else:
            self.init_state()

    def init_state(self):
        # Lấy batch đầu để xác định graph_size, input_dim
        data_loader = self.loader.get_loader()
        for batch in data_loader:
            self.args['graph_size'] = batch.x.size(0)
            self.args['input_dim'] = batch.num_features
            break

        # Tạo model
        self.model = self.model_class(
            n_feat=self.args['input_dim'],
            n_dims=self.args['client_hidden_dim'],
            n_clss=self.args['n_classes'],
            args=self.args
        ).cuda(self.gpu_id)

        self.parameters = list(self.model.parameters())
        self.optimizer = torch.optim.Adam(
            self.parameters,
            lr=self.args['client_lr'],
            weight_decay=self.args['client_weight_decay']
        )

    def load_state(self):
        ckpt_file = f"{self.client_id}_state.pt"
        loaded = torch_load(self.args['checkpt_path'], ckpt_file)
        if self.model is None:
            # Tạo model trống
            self.init_state()  # Hoặc init 1 model trống
        set_state_dict(self.model, loaded['model'], strict=True)
        self.optimizer.load_state_dict(loaded['optimizer'])

    def save_state(self):
        ckpt_file = f"{self.client_id}_state.pt"
        out = {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model)
        }
        torch_save(self.args['checkpt_path'], ckpt_file, out)

    def on_receive_message(self, curr_rnd, message_type):
        self.curr_rnd = curr_rnd
        if message_type == 'client_generate_vector_start':
            print(f"[client{self.client_id}] round{curr_rnd} generate vector start")
        if message_type == 'client_train_on_generated_model_prams':
            print(f"[client{self.client_id}] round{curr_rnd} begin training on generated model params")

    def generate_vector(self, client_id):
        """Giống logic cũ: train tạm 30 epochs => lấy embedding."""
        self.model.train()
        data_loader = self.loader.get_loader()

        embedding_tmp = None
        for batch in data_loader:
            batch = batch.cuda(self.gpu_id)
            for epoch in range(self.args['client_vector_epochs']):
                self.optimizer.zero_grad()
                out = self.model(batch)
                train_mask = batch.train_mask
                loss = F.cross_entropy(out[train_mask], batch.y[train_mask])
                loss.backward()
                self.optimizer.step()
                if epoch == self.args['client_vector_epochs'] - 1:
                    self.model.eval()
                    with torch.no_grad():
                        embedding_tmp = self.model(batch, is_proxy=True)

        average_embedding = torch.mean(embedding_tmp, dim=0, keepdim=True)
        self.sd[client_id] = {'functional_embedding': average_embedding.clone().detach()}

    @torch.no_grad()
    def eval_model(self, mode='test'):
        self.model.eval()
        data_loader = self.loader.get_loader()
        target, pred, loss_vals = [], [], []
        for batch in data_loader:
            batch = batch.cuda(self.gpu_id)
            if mode == 'test':
                mask = batch.test_mask
            elif mode == 'valid':
                mask = batch.val_mask
            else:
                mask = batch.train_mask
            out = self.model(batch)
            if torch.sum(mask).item() == 0:
                lss = 0.0
            else:
                lss = F.cross_entropy(out[mask], batch.y[mask])
                loss_vals.append(lss.item())
            pred.append(out[mask])
            target.append(batch.y[mask])

        if len(target) > 0:
            preds_stacked = torch.cat(pred, dim=0)
            targets_stacked = torch.cat(target, dim=0)
            predicted_label = preds_stacked.argmax(dim=1)
            acc = (predicted_label == targets_stacked).float().mean().item()
        else:
            acc = 1.0

        mean_loss = float(np.mean(loss_vals)) if len(loss_vals) > 0 else 0.0
        return acc, mean_loss

    def train_client_model(self, update_client_embedding):
        # Giống code cũ: load 'generated model params', train => tính delta => lưu sd
        generated_param = self.sd[self.client_id].pop('generated model params')

        # Gán param
        self.model.conv1.weight = nn.Parameter(generated_param['gcn1.weight'])
        self.model.conv1.bias   = nn.Parameter(generated_param['gcn1.bias'])
        self.model.conv2.weight = nn.Parameter(generated_param['gcn2.weight'])
        self.model.conv2.bias   = nn.Parameter(generated_param['gcn2.bias'])

        # Evaluate on generated model
        val_gen_acc, _   = self.eval_model(mode='valid')
        test_gen_acc, _  = self.eval_model(mode='test')
        train_gen_acc, _ = self.eval_model(mode='train')

        # Train c_epoch
        c_epoch = self.args['client_train_epochs']
        data_loader = self.loader.get_loader()
        for epoch in range(c_epoch):
            self.model.train()
            for batch in data_loader:
                batch = batch.cuda(self.gpu_id)
                self.optimizer.zero_grad()
                out = self.model(batch)
                train_mask = batch.train_mask
                lss = F.cross_entropy(out[train_mask], batch.y[train_mask])
                lss.backward()
                self.optimizer.step()

        final_param = self.model.state_dict()

        # Evaluate
        val_train_acc, _   = self.eval_model(mode='valid')
        test_train_acc, _  = self.eval_model(mode='test')
        train_train_acc, _ = self.eval_model(mode='train')

        # update_client_embedding => generate new embedding
        if update_client_embedding:
            self.model.eval()
            embedding_tmp = None
            for batch in data_loader:
                batch = batch.cuda(self.gpu_id)
                embedding_tmp = self.model(batch, is_proxy=True)
            avg_embed = torch.mean(embedding_tmp, dim=0, keepdim=True)
            self.sd[self.client_id] = {'functional_embedding': avg_embed.clone().detach()}

        # Tính delta param
        delta_param = OrderedDict()
        # Khoá GCN1
        delta_param['gcn1.weight'] = final_param['conv1.weight'] - generated_param['gcn1.weight']
        delta_param['gcn1.bias']   = final_param['conv1.bias']   - generated_param['gcn1.bias']
        # Khoá GCN2
        delta_param['gcn2.weight'] = final_param['conv2.weight'] - generated_param['gcn2.weight']
        delta_param['gcn2.bias']   = final_param['conv2.bias']   - generated_param['gcn2.bias']

        # Lưu vào sd
        self.sd[self.client_id].update({
            'delta param': {k: v.clone().detach() for k,v in delta_param.items()},
            'train_acc': train_train_acc,
            'val_acc': val_train_acc,
            'test_acc': test_train_acc
        })

        print(f"[client{self.client_id}] rnd{self.curr_rnd}, val_acc={val_train_acc}, test_acc={test_train_acc}")

        self.save_state()
