"""
Created on Tue Sep 28 16:53:37 CST 2021
@author: lab-chen.weidong
"""

import torch
from torch import distributed as dist

def load_model(model_type, device, **kwargs):
    if model_type=='ks_transformer':   
        from model.KS_transformer_model import build_ks_transformer_model
        model = build_ks_transformer_model(**kwargs)
    else:
        raise KeyError(f'Unknown model type: {model_type}')

    if device == 'cuda':
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])

    return model


