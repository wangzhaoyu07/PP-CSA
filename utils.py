import torch
from torch.utils.data import Dataset, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import json
import os
import pdb


def get_model_type(model):
    '''
    Given a specified model for an experiment, return the type of model ('rnn'
    or 'transformer')
    '''
    model_to_model_type = {
        'adept': 'rnn',
        'custom_rnn': 'rnn',
        }
    if model in model_to_model_type.keys():
        model_type = model_to_model_type[model]
    else:
        raise Exception("Specified model not in current list of available "
                        "models.")
    return model_type


class EarlyStopping(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, checkpoint_dict, mod_name):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            torch.save(checkpoint_dict, mod_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of '
                  f'{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            checkpoint_dict['checkpoint_early_stopping'] = self.counter
            torch.save(checkpoint_dict, mod_name)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def decode_rewritten(rewritten, preprocessor, remove_special_tokens=True,
                     labels=False, model_type='transformer',is_pruning=False):
    decoded = decode_rewritten_rnn(
        rewritten, preprocessor,
        remove_special_tokens=remove_special_tokens, labels=labels,is_pruning=is_pruning)
    
    return decoded

def decode_rewritten_rnn(rewritten, preprocessor, remove_special_tokens=True,
                         labels=False, is_pruning=False):
    '''
    rewritten: torch tensor size batch X max_seq_len-1, type int64
    preprocessor: preprocessing class from preprocessing.py
    remove_special_tokens: ignore <pad>, <unk>, <sos> and <eos> tokens

    decoded: list of strings, with predicted tokens separated by a space
    '''
    special_tokens = {0, 1, 2, 3}
    if is_pruning:
        decoded = [[preprocessor.idx2word[idx] for idx in preprocessor._restore_path(batch) if (not remove_special_tokens) or (idx not in special_tokens)]
                for batch in rewritten]
    else:
        decoded = [[preprocessor.idx2word[idx.item()] for idx in batch if (not remove_special_tokens) or (idx.item() not in special_tokens)]
                for batch in rewritten]

    if not labels:
        decoded = [';'.join(batch) for batch in decoded]

    # For empty strings
    decoded = [doc if doc != '' else 'UNK' for doc in decoded]

    return decoded
