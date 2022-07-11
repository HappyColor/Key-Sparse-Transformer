
import os
import numpy as np
import librosa
import librosa.display
import torch
import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def makesure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calculate_score(preds, labels):
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    ua = recall_score(labels, preds, average='macro', zero_division=0)
    confuse_matrix = confusion_matrix(labels, preds)
    return accuracy, f1, precision, ua, confuse_matrix

def calculate_accuracy(preds, labels):
    accuracy = accuracy_score(labels, preds)
    return accuracy


def tidy_csvfile(csvfile, colname, ascending=True):
    '''
    tidy csv file base on a particular column.
    '''
    print(f'tidy file: {csvfile}, base on column: {colname}')
    df = pd.read_csv(csvfile)
    df = df.sort_values(by=[colname], ascending=ascending, na_position='last')
    df = df.round(3)
    df.to_csv(csvfile, index=False, sep=',')

    
