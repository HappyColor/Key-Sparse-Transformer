
from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)
_C.model = CN(new_allowed=True)
_C.model.type = 'ks_transformer'

_C.train = CN(new_allowed=True)
_C.train.device = 'cuda'
_C.train.num_workers = 6

_C.train.EPOCH = 120
_C.train.batch_size = 16
_C.train.lr = 0.0005
_C.train.seed = 123
_C.train.device_id = '0,2'
_C.train.find_init_lr = False

_C.dataset = CN(new_allowed=True)
_C.dataset.database = 'iemocap'
_C.dataset.feature = ['wav2vec', 'roberta']   

_C.mark = None
