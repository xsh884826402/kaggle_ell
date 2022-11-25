import numpy as np
import pandas as pd
import importlib
import sys
import random
from tqdm import tqdm
import gc
import argparse
import torch
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import SequentialSampler, DataLoader
from pytorchtools import EarlyStopping
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import yaml
from types import SimpleNamespace
import os
import re


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


sys.path.append("models")
sys.path.append("datasets")


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_train_dataloader(train_ds, cfg):
    train_dataloader = DataLoader(
        train_ds,
        sampler=None,
        shuffle=True,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.environment.number_of_workers,
        pin_memory=False,
        # collate_fn=cfg.CustomDataset.get_train_collate_fn,
        drop_last=cfg.training.drop_last_batch,
        worker_init_fn=worker_init_fn,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_val_dataloader(val_ds, cfg):
    val_dataloader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=cfg.predicting.batch_size//2,
        num_workers=cfg.environment.number_of_workers,
        pin_memory=True,
        # collate_fn=cfg.CustomDataset.get_validation_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    print(f"val: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader


def get_scheduler(cfg, optimizer, total_steps):
    if cfg.training.schedule == "Linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                cfg.training.warmup_epochs * (total_steps // cfg.training.batch_size)
            ),
            num_training_steps=cfg.training.epochs
            * (total_steps // cfg.training.batch_size),
        )
    else:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(
                cfg.training.warmup_epochs * (total_steps // cfg.training.batch_size)
            ),
            num_training_steps=cfg.training.epochs
            * (total_steps // cfg.training.batch_size),
        )
    return scheduler


def load_checkpoint(cfg, model, path):
    d = torch.load(path, map_location="cpu")

    if "model" in d:
        model_weights = d["model"]
    else:
        model_weights = d

    # if (
    #     model.backbone.embeddings.word_embeddings.weight.shape[0]
    #     < model_weights["backbone.embeddings.word_embeddings.weight"].shape[0]
    # ):
    #     print("resizing pretrained embedding weights")
    #     model_weights["backbone.embeddings.word_embeddings.weight"] = model_weights[
    #         "backbone.embeddings.word_embeddings.weight"
    #     ][: model.backbone.embeddings.word_embeddings.weight.shape[0]]

    try:
        model.load_state_dict(model_weights, strict=True)
    except Exception as e:
        print("removing unused pretrained layers")
        for layer_name in re.findall("size mismatch for (.*?):", str(e)):
            model_weights.pop(layer_name, None)
        model.load_state_dict(model_weights, strict=False)

    print(f"Weights loaded from: {cfg.architecture.pretrained_weights}")


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_model(cfg):
    Net = importlib.import_module(cfg.model_class).Net
    return Net(cfg)

def get_predict_func(cfg):
    return importlib.import_module(cfg.model_class).predict

def get_loss_fn(cfg):
    return importlib.import_module(cfg.model_class).loss_fn


def get_kfold(cfg):
    if cfg.dataset.train_dataframe.endswith(".pq"):
        train_df = pd.read_parquet(cfg.dataset.train_dataframe)
    else:
        train_df = pd.read_csv(cfg.dataset.train_dataframe)
    mskf = MultilabelStratifiedKFold(n_splits=cfg.dataset.folds, shuffle=True, random_state=cfg.environment.seed)
    print('label cols', cfg.dataset.label_columns)
    for fold, (train_index, val_index) in enumerate(mskf.split(train_df, train_df[cfg.dataset.label_columns])):
        train_df.loc[val_index, 'fold'] = int(fold)
    return train_df


def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores

parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
cfg = yaml.safe_load(open(parser_args.config).read())

for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)
cfg.CustomDataset = importlib.import_module(cfg.dataset_class).CustomDataset

if __name__ == "__main__":

    if cfg.environment.seed < 0:
        cfg.environment.seed = np.random.randint(1_000_000)
    else:
        cfg.environment.seed = cfg.environment.seed

    set_seed(cfg.environment.seed)

    df = pd.read_csv(cfg.dataset.train_dataframe_add_fold_label_path)
    preds = []

    for fold in range(cfg.dataset.folds):
        print(f'\n\n---------------------------FODL {fold}----------------------------------\n\n')

        val_df = df[df['fold'] == fold].reset_index(drop=True)
        val_dataset = cfg.CustomDataset(val_df, cfg=cfg)
        val_dataloader = get_val_dataloader(val_dataset, cfg)

        model = get_model(cfg)
        load_checkpoint(cfg, model, f"output/{cfg.experiment_name}/fold{fold}_checkpoint.pth")
        model.to(cfg.device)
        model.eval()

        progress_bar = tqdm(range(len(val_dataloader)))
        val_it = iter(val_dataloader)
        preds_per_fold = []
        for itr in progress_bar:
            inputs, labels = next(val_it)
            inputs = cfg.CustomDataset.collate_fn(inputs)
            batch = cfg.CustomDataset.batch_to_device(inputs, cfg.device)
            labels = cfg.CustomDataset.batch_to_device(labels, cfg.device)

            if cfg.environment.mixed_precision:
                with autocast():
                    outputs = model(batch['input_ids'], batch['attention_mask'])
            else:
                outputs = model(batch['input_ids'], batch['attention_mask'])
            predict_func = get_predict_func(cfg)

            if cfg.predicting.return_probs:
                labels, probs = predict_func(outputs.detach() if cfg.architecture.direct_result else outputs.logits.detach(), cfg)
                target_columns = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
                df_labels = pd.DataFrame(labels, columns=[item + '_label' for item in target_columns])
                df_probs = pd.DataFrame(probs, columns=[item + '_prob' for item in target_columns])
                preds_per_fold.append(pd.concat([df_labels, df_probs], axis=1))
            else:
                labels = predict_func(outputs.detach() if cfg.architecture.direct_result else outputs.logits.detach(), cfg)
                target_columns = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
                df_labels = pd.DataFrame(labels, columns=[item + '_label' for item in target_columns])
                preds_per_fold.append(df_labels)

        df_preds_per_fold = pd.concat(preds_per_fold, axis=0)
        print(f'len : {len(df_preds_per_fold)}')
        preds.append(pd.concat([val_df.reset_index(drop=True), df_preds_per_fold.reset_index(drop=True)], axis=1))
        del model
        _ = gc.collect()
    df_final = pd.concat(preds, axis=0)
    df_final.to_csv(f"output/{cfg.experiment_name}/predicts.csv", index=False)
    oof_score = get_score(y_trues=df_final[cfg.dataset.label_columns].values, y_preds=df_final[[item + '_label' for item in cfg.dataset.label_columns]])
    print(f'Experiment {cfg.experiment_name} oof_score: {oof_score}')











