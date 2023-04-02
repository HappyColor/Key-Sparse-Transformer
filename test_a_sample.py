#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Dec 16 17:40:17 CST 2021
@author: lab-chen.weidong
'''

import soundfile
import torch
import numpy as np
import scipy.signal as signal
import json
from fairseq.models.wav2vec import Wav2VecModel
from fairseq.models.roberta import RobertaModel
from model.KS_transformer import build_ks_transformer
import utils

def extract_wav2vec(wavfile, model_path):
    print('Extracting wav2vec feature...')
    cp = torch.load(model_path, map_location='cpu')
    wav2vec = Wav2VecModel.build_model(cp['args'], task=None)
    wav2vec.load_state_dict(cp['model'])
    wav2vec.eval()

    sample_rate = 16000
    wavs, fs = soundfile.read(wavfile)
    
    if fs != sample_rate:
        result = int((wavs.shape[0]) / fs * sample_rate)
        wavs = signal.resample(wavs, result)

    if wavs.ndim > 1:
        wavs = np.mean(wavs, axis=1)

    wavs = torch.from_numpy(np.float32(wavs)).unsqueeze(0)
    
    z = wav2vec.feature_extractor(wavs)
    feature = wav2vec.feature_aggregator(z)
    feature = feature.transpose(1,2).squeeze(dim=0).detach().numpy()   # (t, 512)
    return feature

def extract_roberta(txtfile, model_path):
    print('Extracting RoBERTa feature...')
    roberta = RobertaModel.from_pretrained(model_path, checkpoint_file='model.pt')
    roberta.eval()

    with open(txtfile, 'r') as f:
        text = f.read()
    tokens = roberta.encode(text)
    embedding = roberta.extract_features(tokens)

    embedding = embedding.squeeze(dim=0).detach().numpy()   # (t, 1024)
    return embedding

def load_model(ckpt=None, device='cuda', num_classes=4):
    print('Loading model...')
    with open('./config/model_config.json', 'r') as f1:
        model_json = json.load(f1)['ks_transformer']
    model = build_ks_transformer(num_classes=num_classes, **model_json).to(device)
    model_state_dict = torch.load(ckpt, map_location=device)['model']
    if device == 'cuda':
        model.load_state_dict(model_state_dict)
    else:
        params = {k.replace('module.', ''): v for k,v in model_state_dict.items()}
        model.load_state_dict(params)
    return model

def run(model, x_a, x_t, x_a_padding_mask, x_t_padding_mask, device):
    print('Inferring...')
    x_a = x_a.unsqueeze(dim=0).to(device)
    x_t = x_t.unsqueeze(dim=0).to(device)
    x_a_padding_mask = x_a_padding_mask.unsqueeze(dim=0).to(device)
    x_t_padding_mask = x_t_padding_mask.unsqueeze(dim=0).to(device)
    out = model(x_a, x_t, x_a_padding_mask, x_t_padding_mask)
    y_pred = torch.argmax(out, dim=1).item()
    label = index_2_label(y_pred)
    return label

def index_2_label(index):
    label_conveter = {0: 'angry', 1: 'neutral', 2: 'happy/excited', 3: 'sad'}
    return label_conveter[index]

if __name__ == '__main__':
    ''' You should modify the following paths to your own paths.
    '''
    wav2vec_model_path = '/148Dataset/data-chen.weidong/pre_trained_model/wav2vec/wav2vec_large.pt'
    roberta_model_path = '/148Dataset/data-chen.weidong/pre_trained_model/roberta/roberta.large'
    ckpt = './experiments/ks_transformer/iemocap_e120_b32_lr0.0005/fold_1/checkpoint/best.pt'
    wavfile = '/148Dataset/data-chen.weidong/iemocap/Session5/sentences/wav/Ses05F_impro03/Ses05F_impro03_M063.wav'
    txtfile = '/148Dataset/data-chen.weidong/iemocap/feature/raw_text/Ses05F_impro03_M063.txt'
    device='cpu'
    
    a_length = 460
    t_length = 20
    x_a = extract_wav2vec(wavfile, wav2vec_model_path)
    x_t = extract_roberta(txtfile, roberta_model_path)
    x_a, x_a_padding_mask = utils.dataset.pad_input(x_a, a_length)
    x_t, x_t_padding_mask = utils.dataset.pad_input(x_t, t_length)

    model = load_model(ckpt, device=device, num_classes=4)
    pred_label = run(model, x_a, x_t, x_a_padding_mask, x_t_padding_mask, device=device)
    print(f'The emotion contains in the input sample is: {pred_label}')
