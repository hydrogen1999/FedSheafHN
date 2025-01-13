# utils/torch_utils.py

import os
import json
import torch
import numpy as np
from collections import OrderedDict

def torch_save(base_dir, filename, data):
    os.makedirs(base_dir, exist_ok=True)
    fpath = os.path.join(base_dir, filename)
    torch.save(data, fpath)

def torch_load(base_dir, filename):
    fpath = os.path.join(base_dir, filename)
    return torch.load(fpath, map_location='cpu')

def get_state_dict(model):
    # Chuyển sang CPU np array
    sd = OrderedDict()
    for k,v in model.state_dict().items():
        sd[k] = v.cpu().clone()
    return sd

def set_state_dict(model, state_dict, strict=True):
    model.load_state_dict(state_dict, strict=strict)
