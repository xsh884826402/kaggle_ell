# ====================================================
# Library
# ====================================================
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset


import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import DataCollatorWithPadding
from config import InferCFG
from utils import *
from model import *
os.environ.setdefault("TOKENIZERS_PARALLELISM", 'false')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====================================================
# tokenizer
# ====================================================
InferCFG.tokenizer = AutoTokenizer.from_pretrained(InferCFG.path+'tokenizer/')

# ====================================================
# oof
# ====================================================
oof_df = pd.read_pickle(InferCFG.path+'oof_df.pkl')
labels = oof_df[InferCFG.target_cols].values
preds = oof_df[[f"pred_{c}" for c in InferCFG.target_cols]].values
score, scores = get_score(labels, preds)

# ====================================================
# infer log
# ====================================================
OUTPUT_DIR = '../data/output/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
LOGGER = get_logger(os.path.join(OUTPUT_DIR, 'infer'))

LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')

# ====================================================
# Data Loading
# ====================================================
test = pd.read_csv('../data/input/test.csv')
submission = pd.read_csv('../data/input/sample_submission.csv')

print(f"test.shape: {test.shape}")
print(test.head())
print(f"submission.shape: {submission.shape}")
print(submission.head())


# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        #max_length=InferCFG.max_len,
        #pad_to_max_length=True,
        #truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        return inputs


# ====================================================
# inference
# ====================================================
def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions


test_dataset = TestDataset(InferCFG, test)
test_loader = DataLoader(test_dataset,
                         batch_size=InferCFG.batch_size,
                         shuffle=False,
                         collate_fn=DataCollatorWithPadding(tokenizer=InferCFG.tokenizer, padding='longest'),
                         num_workers=InferCFG.num_workers, pin_memory=True, drop_last=False)
predictions = []
for fold in InferCFG.trn_fold:
    model = CustomModel(InferCFG, LOGGER, config_path=InferCFG.config_path, pretrained=False)
    state = torch.load(InferCFG.path+f"{InferCFG.model.replace('/', '-')}_fold{fold}_best.pth",
                       map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, device)
    predictions.append(prediction)
    del model, state, prediction; gc.collect()
    torch.cuda.empty_cache()
predictions = np.mean(predictions, axis=0)
test[InferCFG.target_cols] = predictions
submission = submission.drop(columns=InferCFG.target_cols).merge(test[['text_id'] + InferCFG.target_cols], on='text_id', how='left')
print(submission.head())
submission[['text_id'] + InferCFG.target_cols].to_csv('submission.csv', index=False)