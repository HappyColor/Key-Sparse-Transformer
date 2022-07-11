
from .config import _C as Cfg
import os
import torch

def create_workshop(cfg, local_rank):
    modeltype = cfg.model.type
    database = cfg.dataset.database
    batch = cfg.train.batch_size
    lr = cfg.train.lr
    epoch = cfg.train.EPOCH
    
    world_size = torch.cuda.device_count()
    batch = batch * world_size

    if cfg.train.find_init_lr:
        cfg.mark = 'find_init_lr_' + cfg.mark
        config_name = f'./experiments/{modeltype}/{database}_b{batch}'
    else:
        config_name = f'./experiments/{modeltype}/{database}_e{epoch}_b{batch}_lr{lr}'

    if cfg.mark is not None:
        config_name = config_name + '_{}'.format(cfg.mark)

    cfg.workshop = os.path.join(config_name, f'fold_{cfg.train.current_fold}')
    cfg.ckpt_save_path = os.path.join(cfg.workshop, 'checkpoint')
    
    if local_rank == 0:
        if os.path.exists(cfg.workshop):
            raise ValueError(f'workshop {cfg.workshop} already existed.')
        else:
            os.makedirs(cfg.workshop)
            os.makedirs(cfg.ckpt_save_path)
    