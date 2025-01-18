# fl/client.py

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np

from data.loader import FedDataLoader
from utils.logger import Logger
from utils.torch_utils import torch_save, torch_load, get_state_dict, set_state_dict
from models.gcn import GCN

class Client:
    def __init__(self, args, gpu_id, shared_dict, client_id):

        self.args = args
        self.gpu_id = gpu_id
        self.sd = shared_dict
        self.client_id = client_id

        self.loader = FedDataLoader(self.args)
        self.logger = Logger(self.args, gpu_id=self.gpu_id, is_server=False)

        # Thay cho self.model = GCN ...
        self.model_class = GCN
        self.model = None
        self.optimizer = None
        # self.curr_rnd = 0

        if args['dataset'] == 'Cora':
            self.args['n_classes'] = 7
        elif args['dataset'] == 'CiteSeer':
            self.args['n_classes'] = 6
        elif args['dataset'] == 'PubMed':
            self.args['n_classes'] = 3
        elif args.dataset == 'ogbn-arxiv':
            self.args.n_classes = 40
        elif args['dataset'] == 'Computers':
            self.args['n_classes'] = 10
        elif args['dataset'] == 'Photo':
            self.args['n_classes'] = 8
        else:
            self.args['n_classes'] = 10

    def switch_state(self, client_id):
        
        self.client_id = client_id
        self.loader.switch(client_id)
        self.logger.switch(client_id)

        # Nếu đã có file ckpt => load, ngược lại => init
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
            self.init_state()
        set_state_dict(self.model, loaded['model'], strict=True)
        self.optimizer.load_state_dict(loaded['optimizer'])

    def save_state(self):
        ckpt_file = f"{self.client_id}_state.pt"
        out = {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model)
        }
        torch_save(self.args['checkpt_path'], ckpt_file, out)

    def on_receive_message(self, curr_rnd, msg_type):
        if msg_type == 'client_generate_vector_start':
            print(f"[client {self.client_id}] round{curr_rnd} => generate vector start.")
        elif msg_type == 'client_train_on_generated_model_prams':
            print(f"[client {self.client_id}] round{curr_rnd} => train on generated model params.")

    def generate_vector(self, curr_rnd):
        self.model.train()
        data_loader = self.loader.get_loader()

        embedding_tmp = None
        for batch in data_loader:
            batch = batch.cuda(self.gpu_id)
            for epoch in range(self.args['client_vector_epochs']):
                self.optimizer.zero_grad()
                out = self.model(batch)
                train_mask = batch.train_mask
                loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()
                self.optimizer.step()
                if epoch == self.args['client_vector_epochs'] - 1:
                    self.model.eval()
                    with torch.no_grad():
                        embedding_tmp = self.model(batch, is_proxy=True)

        if embedding_tmp is not None:
            avg_embed = torch.mean(embedding_tmp, dim=0, keepdim=True)
            self.sd[self.client_id] = {'functional_embedding': avg_embed.detach().clone()}

    @torch.no_grad()
    def eval_model(self, mode='test'):
        self.model.eval()
        dl = self.loader.get_loader()
        losses = []
        preds, tgts = [], []
        for batch in dl:
            batch = batch.cuda(self.gpu_id)
            if mode == 'test':
                mask = batch.test_mask
            elif mode == 'valid':
                mask = batch.val_mask
            else:
                mask = batch.train_mask
            out = self.model(batch)
            if m.sum().item() > 0:
                loss = F.cross_entropy(out[m], batch.y[m])
                losses.append(loss.item())
                preds.append(out[m])
                tgts.append(batch.y[m])

        if len(preds) == 0:
            return 1.0, 0.0
        cat_pred = torch.cat(preds, dim=0)
        cat_tgts = torch.cat(tgts, dim=0)
        acc = (cat_pred.argmax(dim=1) == cat_tgts).float().mean().item()

        return acc, float(np.mean(losses))

    def train_client_model(self, update_client_embedding):
        """
        Lấy 'generated model params' từ self.sd, gán vào model, 
        train c_epoch => tính delta => store vào self.sd[client_id].
        """
        if 'generated model params' not in self.sd[self.client_id]:
            # Nothing to train
            return

        gen_params = self.sd[self.client_id].pop('generated model params')

        # Gán param
        self.model.conv1.weight = nn.Parameter(gen_params['gcn1.weight'])
        self.model.conv1.bias   = nn.Parameter(gen_params['gcn1.bias'])
        self.model.conv2.weight = nn.Parameter(gen_params['gcn2.weight'])
        self.model.conv2.bias   = nn.Parameter(gen_params['gcn2.bias'])

        # Evaluate trước khi train
        val_gen_acc, _ = self.eval_model('valid')
        test_gen_acc,_ = self.eval_model('test')
        train_gen_acc,_= self.eval_model('train')

        # Train c_epoch
        c_epoch = self.args['client_train_epochs']
        dl = self.loader.get_loader()
        for epoch in range(c_epoch):
            self.model.train()
            for batch in dl:
                batch = batch.cuda(self.gpu_id)
                self.optimizer.zero_grad()
                out = self.model(batch)
                loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
                loss.backward()
                self.optimizer.step()

        # Evaluate sau khi train
        val_acc, _   = self.eval_model('valid')
        test_acc,_   = self.eval_model('test')
        train_acc,_  = self.eval_model('train')

        # update_client_embedding => generate new embedding
        if update_client_embedding:
            emb_tmp = None
            self.model.eval()
            for batch in dl:
                batch = batch.cuda(self.gpu_id)
                emb_tmp = self.model(batch, is_proxy=True)
            if emb_tmp is not None:
                avg_emb = torch.mean(emb_tmp, dim=0, keepdim=True)
                self.sd[self.client_id] = {'functional_embedding': avg_emb.detach().clone()}
        # Tính delta param
        # Tính delta
        final_param = self.model.state_dict()
        delta = {}
        delta['gcn1.weight'] = final_param['conv1.weight'] - gen_params['gcn1.weight']
        delta['gcn1.bias']   = final_param['conv1.bias']   - gen_params['gcn1.bias']
        delta['gcn2.weight'] = final_param['conv2.weight'] - gen_params['gcn2.weight']
        delta['gcn2.bias']   = final_param['conv2.bias']   - gen_params['gcn2.bias']

        # Lưu vào sd
        self.sd[self.client_id].update({
            'delta': {k: v.detach().clone() for k,v in delta.items()},
            'val_acc': val_acc,
            'test_acc': test_acc
        })

        print(f"[client {self.client_id}] trained => val_acc={val_acc:.3f}, test_acc={test_acc:.3f}")
        self.save_state()
