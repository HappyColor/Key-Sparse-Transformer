"""
Created on Thu Dec 16 11:22:32 CST 2021
@author: lab-chen.weidong
"""

import pandas as pd
import numpy as np
import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from scipy import io
import multiprocessing as mp

def identity(x):
    return x

class DistributedDalaloaderWrapper():
    def __init__(self, dataloader: DataLoader, collate_fn):
        self.dataloader = dataloader
        self.collate_fn = collate_fn
    
    def _epoch_iterator(self, it):
        for batch in it:
            yield self.collate_fn(batch)

    def __iter__(self):
        it = iter(self.dataloader)
        return self._epoch_iterator(it)

    def __len__(self):
        return len(self.dataloader)

    @property
    def dataset(self):
        return self.dataloader.dataset

    def set_epoch(self, epoch: int):
        self.dataloader.sampler.set_epoch(epoch)

def universal_collater(batch):
    all_data = [[] for _ in range(len(batch[0]))]
    for one_batch in batch:
        for i, (data) in enumerate(one_batch):
            all_data[i].append(data)
    return all_data

class Base_database():
    def __init__(self, names, emotion_labels, matdir=None, matkey=None, state=None, label_conveter=None):
        self.names = names
        self.emotion_labels = emotion_labels
        self.state = state
        self.matdir = matdir
        self.matkey = matkey
        self.conveter = label_conveter
        
    def get_wavfile_label(self, name):
        idx = self.names.index(name)
        label = self.emotion_labels[idx]
        return label

    def load_a_sample(self, idx=0, lenght=None):
        label = self.emotion_labels[idx]
        x = np.float32(io.loadmat(os.path.join(self.matdir, self.names[idx]))[self.matkey])
        x = pad_input(x, lenght) if lenght is not None else x
        y = torch.tensor(self.label_2_index(label))
        return x, y

    def get_wavfile_path(self):
        raise NotImplementedError
    
    def get_sample_name(self, idx):
        return self.names[idx]
    
    def label_2_index(self, label):
        index = self.conveter[label]
        return index

class Base_dataset(Dataset):
    def __init__(self, database: Base_database):
        super().__init__()
        self.database = database
    
    def __len__(self):
        return len(self.database.names)

    def __getitem__(self, idx):
        return _getitem(idx, self.database)

class IEMOCAP(Base_database):
    def __init__(self, matdir=None, matkey=None, length=None, state=None, meta_csv_file='/148Dataset/data-chen.weidong/iemocap/feature/name_label_text.csv'):
        self.length = length
        df = pd.read_csv(meta_csv_file)
        df_sad = df[df.label == 'sad']
        df_neu = df[df.label == 'neu']
        df_ang = df[df.label == 'ang']
        df_hap = df[df.label == 'hap']
        df_exc = df[df.label == 'exc']
        df_list = [df_sad, df_neu, df_ang, df_hap, df_exc]
        df = pd.concat(df_list)

        names, emotion_labels = [], []
        for row in df.iterrows():
            names.append(row[1]['name'])
            emotion_labels.append(row[1]['label'])

        label_conveter = {'ang': 0, 'neu': 1, 'hap': 2, 'exc': 2, 'sad': 3}
        super().__init__(names, emotion_labels, matdir, matkey, state, label_conveter)
    
    def load_a_sample(self, idx=0):
        ''' Load audio and text information together.
        '''
        label = self.emotion_labels[idx]
        x0 = np.float32(io.loadmat(os.path.join(self.matdir[0], self.names[idx]))[self.matkey[0]])
        x1 = np.float32(io.loadmat(os.path.join(self.matdir[1], self.names[idx]))[self.matkey[1]])
        x0, x0_padding_mask = pad_input(x0, self.length[0]) if self.length is not None else x0
        x1, x1_padding_mask = pad_input(x1, self.length[1]) if self.length is not None else x1
        y = torch.tensor(self.label_2_index(label))
        return x0, x1, x0_padding_mask, x1_padding_mask, y
    
    def foldsplit(self, fold, strategy='5cv'):
        if strategy == '5cv':
            assert 1<=fold<=5, print('leave-one-session-out 5-fold cross validation, but got fold {}'.format(fold))
        elif strategy == '10cv':
            assert 1<=fold<=10, print('leave-one-speaker-out 10-fold cross validation , but got fold {}'.format(fold))
        else:
            raise KeyError('Wrong cross validation setting')

        name_fold, y_fold = [], []
        if strategy == '5cv':
            testSes = 'Ses0{}'.format(6-fold)
            if self.state == 'test':
                for i, name in enumerate(self.names):
                    if testSes in name:
                        name_fold.append(name)
                        y_fold.append(self.emotion_labels[i])
            else:
                for i, name in enumerate(self.names):
                    if testSes not in name:
                        name_fold.append(name)
                        y_fold.append(self.emotion_labels[i])
        else:
            gender = 'F' if fold%2 == 0 else 'M'
            fold = math.ceil(fold/2)
            testSes = 'Ses0{}'.format(6-fold)
            if self.state == 'test':
                for i, name in enumerate(self.names):
                    if (testSes in name) and (gender in name.split('_')[-1]):
                        name_fold.append(name)
                        y_fold.append(self.emotion_labels[i])
            else:
                for i, name in enumerate(self.names):
                    if (testSes not in name) or (gender not in name.split('_')[-1]): 
                        name_fold.append(name)
                        y_fold.append(self.emotion_labels[i])
            
        self.names = name_fold
        self.emotion_labels = y_fold

    def get_wavfile_path(self, name):
        '''
        name: wav file name with no extension

        return: 
            wav file absolute path
        '''
        clue = name.split('_M')[0] if '_M' in name else name.split('_F')[0]
        ses = clue.split('M_')[0] if 'M_' in clue else clue.split('F_')[0]
        ses = ses.replace('Ses0', 'Session')

        wavpath = '/148Dataset/data-chen.weidong/iemocap/{}/sentences/wav/{}'.format(ses, clue)
        wavfile = os.path.join(wavpath, name+'.wav')

        return wavfile

class IEMOCAP_dataset(Base_dataset):
    def __init__(self, matdir, matkey, length, state, meta_csv_file, fold=1, strategy='5cv'):
        database = IEMOCAP(matdir, matkey, length, meta_csv_file=meta_csv_file, state=state)
        database.foldsplit(fold, strategy)
        super().__init__(database)

class DataloaderFactory():
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, state, **kwargs):
        if self.cfg.dataset.database == 'iemocap':
            dataset = IEMOCAP_dataset(
                fold=self.cfg.train.current_fold, 
                strategy=self.cfg.train.strategy,
                state=state,
                **kwargs
            )
        else:
            raise KeyError(f'Unsupported database: {self.cfg.dataset.database}')
        
        collate_fn = universal_collater
        sampler = DistributedSampler(dataset, shuffle=state == 'train')
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.cfg.train.batch_size, 
            drop_last=False, 
            num_workers=self.cfg.train.num_workers, 
            collate_fn=identity,
            sampler=sampler, 
            multiprocessing_context=mp.get_context('fork')
        )

        return DistributedDalaloaderWrapper(dataloader, collate_fn)

def _getitem(idx: int, database: Base_database):
    x0, x1, x0_padding_mask, x1_padding_mask, y = database.load_a_sample(idx)
    return x0, x1, x0_padding_mask, x1_padding_mask, y

def pad_input(x: np.ndarray, lenght, pad_value=0):
    t = x.shape[0]
    mask = torch.zeros(lenght)
    if lenght > t:
        x = np.pad(x, ((0,lenght-t), (0,0)), 'constant', constant_values=(pad_value, pad_value))
        mask[-(lenght-t):] = 1
    else:
        x = x[:lenght,:]
    x = torch.from_numpy(x)
    mask = mask.eq(1)
    return x, mask
