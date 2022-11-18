import numpy as np
import pandas as pd
import importlib
import sys
import random
from tqdm import tqdm
import gc
import argparse
import torch

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
        batch_size=cfg.predicting.batch_size,
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


def merge_k_folds_result(df, direct_result):
    if not direct_result:
        def transform_func(x, column):
            columns_label_folds = [column + '_label' + f'_fold{i}' for i in range(5)]
            labels = x[columns_label_folds].values
            result = dict()
            for label, prob in zip(labels, probs):
                if label not in result.keys():
                    result[label] = [1, prob]
                else:
                    result[label][0] += 1
                    result[label][1] += prob
            max_count = 0
            avg_prob = 0
            label = -1
            for k, v in result.items():
                count, prob = v
                if count > max_count:
                    label = k
                    max_count = count
                    avg_prob = prob / count
            return label, avg_prob

        columns = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        for column in columns:
            df[column + '_label'], df[column + '_prob'] = zip(*df.apply(transform_func, args=(column,), axis=1))
        reserve_columns = ['text_id', 'full_text'] + [column + '_label' for column in columns] + [column + '_prob' for
                                                                                                  column in columns]
        return df[reserve_columns]
    else:
        def transform_func(x, column):
            columns_label_folds = [column + '_label' + f'_fold{i}' for i in range(5)]
            columns_prob_folds = [column + '_prob' + f'_fold{i}' for i in range(5)]
            labels = x[columns_label_folds].values
            return np.mean(labels)

        columns = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        for column in columns:
            df[column + '_label']= df.apply(transform_func, args=(column,), axis=1)
        reserve_columns = ['text_id', 'full_text'] + [column + '_label' for column in columns]
        return df[reserve_columns]


parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
cfg = yaml.safe_load(open(parser_args.config).read())

for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)
cfg.CustomInferDataset = importlib.import_module(cfg.dataset_class).CustomInferDataset

if __name__ == "__main__":

    if cfg.environment.seed < 0:
        cfg.environment.seed = np.random.randint(1_000_000)
    else:
        cfg.environment.seed = cfg.environment.seed

    set_seed(cfg.environment.seed)
    # 整理合并第一阶段模型输出
    first_stage_outputs = []
    for index, (file_path, with_prob) in enumerate(zip(cfg.dataset.first_stage_outputs, cfg.dataset.with_probs)):
        df_first_stage = pd.read_csv(file_path)
        if not with_prob:
            keys = [item + "_label" for item in cfg.dataset.label_columns]
            values = [item + "_label" + f"_{index}" for item in cfg.dataset.label_columns]
            columns_dict = dict(zip(keys, values))
        else:
            label_keys = [item + "_label" for item in cfg.dataset.label_columns]
            label_values = [item + "_label" + f"_{index}" for item in cfg.dataset.label_columns]
            prob_keys = [item + "_prob" for item in cfg.dataset.label_columns]
            prob_values = [item + "_prob" + f"_{index}" for item in cfg.dataset.label_columns]
            columns_dict = dict(zip(label_keys + prob_keys, label_values + prob_values))
        df_first_stage.rename(columns=columns_dict, inplace=True)
        first_stage_outputs.append(df_first_stage)
    df = first_stage_outputs[0]
    for tmp in first_stage_outputs[1:]:
        df = df.merge(tmp, on=['text_id', 'full_text'])
    print(df.columns)
    keys = [item + "_x" for item in cfg.dataset.label_columns]
    values = [item for item in cfg.dataset.label_columns]
    df.rename(columns=dict(zip(keys, values)), inplace=True)

    val_df = df
    df_k_folds = []

    for fold in range(cfg.dataset.folds):
        print(f'\n\n---------------------------FODL {fold}----------------------------------\n\n')

        val_dataset = cfg.CustomInferDataset(val_df, cfg=cfg)
        val_dataloader = get_val_dataloader(val_dataset, cfg)

        model = get_model(cfg)
        load_checkpoint(cfg, model, f"output/{cfg.experiment_name}/fold{fold}_checkpoint.pth")
        model.to(cfg.device)
        model.eval()

        progress_bar = tqdm(range(len(val_dataloader)))
        val_it = iter(val_dataloader)
        preds_per_fold = []
        for itr in progress_bar:
            inputs = next(val_it)
            batch = cfg.CustomInferDataset.batch_to_device(inputs, cfg.device)

            outputs = model(batch)
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
            # print(f'df_labels {df_labels.head()}')
        df_preds_per_fold = pd.concat(preds_per_fold, axis=0)

        df_columns = df_preds_per_fold.columns
        new_columns = [column + f"_fold{fold}" for column in df_columns]
        df_preds_per_fold.columns = new_columns
        df_k_folds.append(df_preds_per_fold.reset_index(drop=True))

        del model
        if "cuda" in cfg.device:
            torch.cuda.empty_cache()
        gc.collect()
    df_final = pd.concat([val_df]+df_k_folds, axis=1)
    print(f'df final {df_final.head()}')
    df_final = merge_k_folds_result(df_final, direct_result=cfg.architecture.direct_result)
    df_final.to_csv(f"output/{cfg.experiment_name}/{cfg.infering.infer_result_path}")










