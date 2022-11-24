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
import transformers

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
        batch_size=cfg.training.batch_size//2,
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


def load_checkpoint(cfg, model):
    d = torch.load(cfg.architecture.pretrained_weights, map_location="cpu")

    if "model" in d:
        model_weights = d["model"]
    else:
        model_weights = d

    if (
        model.backbone.embeddings.word_embeddings.weight.shape[0]
        < model_weights["backbone.embeddings.word_embeddings.weight"].shape[0]
    ):
        print("resizing pretrained embedding weights")
        model_weights["backbone.embeddings.word_embeddings.weight"] = model_weights[
            "backbone.embeddings.word_embeddings.weight"
        ][: model.backbone.embeddings.word_embeddings.weight.shape[0]]

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


def get_loss_fn(cfg):
    return importlib.import_module(cfg.model_class).loss_fn


def get_kfold(cfg, train_df=None):
    if train_df is None:
        if cfg.dataset.train_dataframe.endswith(".pq"):
            train_df = pd.read_parquet(cfg.dataset.train_dataframe)
        else:
            train_df = pd.read_csv(cfg.dataset.train_dataframe)
    mskf = MultilabelStratifiedKFold(n_splits=cfg.dataset.folds, shuffle=True, random_state=cfg.environment.seed)
    print('label cols', cfg.dataset.label_columns)
    for fold, (train_index, val_index) in enumerate(mskf.split(train_df, train_df[cfg.dataset.label_columns])):
        train_df.loc[val_index, 'fold'] = int(fold)
    return train_df


def merge_first_stage_outputs(cfg):
    # 整理合并第一阶段模型输出
    first_stage_outputs = []
    for index, (file_path, with_prob) in enumerate(zip(cfg.dataset.first_stage_outputs, cfg.dataset.with_probs)):
        df_first_stage = pd.read_csv(file_path)
        if not with_prob:
            keys = cfg.dataset.input_columns
            values = [item + "_label" + f"_{index}" for item in cfg.dataset.input_columns]
            columns_dict = dict(zip(keys, values))

        df_first_stage.rename(columns=columns_dict, inplace=True)
        first_stage_outputs.append(df_first_stage)
    df = first_stage_outputs[0]
    for tmp in first_stage_outputs[1:]:
        df = df.merge(tmp.drop(columns=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']),
                      on=['text_id', 'full_text'])
    keys = [item + "_x" for item in cfg.dataset.label_columns]
    values = [item for item in cfg.dataset.label_columns]
    df.rename(columns=dict(zip(keys, values)), inplace=True)
    return df

parser = argparse.ArgumentParser(description="")
parser.add_argument("-C", "--config", help="config filename")
parser_args, _ = parser.parse_known_args(sys.argv)
cfg = yaml.safe_load(open(parser_args.config).read())

for k, v in cfg.items():
    if type(v) == dict:
        cfg[k] = SimpleNamespace(**v)
cfg = SimpleNamespace(**cfg)
print(cfg)
for limit in cfg.training.gpu_limit:
    torch.cuda.set_per_process_memory_fraction(limit['fraction'], limit['device'])

os.makedirs(f"output/{cfg.experiment_name}", exist_ok=True)
cfg.CustomDataset = importlib.import_module(cfg.dataset_class).CustomDataset

if __name__ == "__main__":
    device = cfg.device

    if cfg.environment.seed < 0:
        cfg.environment.seed = np.random.randint(1_000_000)
    else:
        cfg.environment.seed = cfg.environment.seed
 
    set_seed(cfg.environment.seed)
    df = merge_first_stage_outputs(cfg)
    df = get_kfold(cfg, df)
    for fold in range(cfg.dataset.folds):
        print(f'\n\n---------------------------FODL {fold}----------------------------------\n\n')
        early_stopping = EarlyStopping(cfg.training.early_stop_patience, verbose=True)
        train_df = df[df['fold'] != fold].reset_index(drop=True)
        val_df = df[df['fold'] == fold].reset_index(drop=True)

        train_dataset = cfg.CustomDataset(train_df, cfg=cfg)
        val_dataset = cfg.CustomDataset(val_df, cfg=cfg)
        print(f'input size {cfg.dataset.input_size}')
        cfg.train_dataset = train_dataset

        train_dataloader = get_train_dataloader(train_dataset, cfg)
        val_dataloader = get_val_dataloader(val_dataset, cfg)

        model = get_model(cfg)
        model.to(device)

        if cfg.architecture.pretrained_weights != "":
            try:
                load_checkpoint(cfg, model)
            except:
                print("WARNING: could not load checkpoint")

        total_steps = len(train_dataset)

        params = model.parameters()

        no_decay = ["bias", "LayerNorm.weight"]
        differential_layers = cfg.training.differential_learning_rate_layers

        layer_wise_lr_decay_params = []
        if cfg.training.layer_wise_lr_decay != -1:
            for name, p in model.named_parameters():
                if name.startswith("model.encoder.layer"):
                    # print(f'name: {name}: :')
                    index = int(name.split(".")[3])
                    layer_wise_lr_decay_params.append({"params": [p],
                                                       "lr": cfg.training.learning_rate*cfg.training.layer_wise_lr_decay**(index+1),
                                                       "weight_decay": 0
                                                       })
                elif name.startswith("model.encoder"):
                    layer_wise_lr_decay_params.append({"params": [p],
                                                       "lr": cfg.training.learning_rate * cfg.training.layer_wise_lr_decay ** (
                                                                   13 + 1),
                                                       "weight_decay": 0
                                                       })

        if  cfg.training.layer_wise_lr_decay != -1:
            optimizer = torch.optim.AdamW(
                layer_wise_lr_decay_params,
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
            )
        else:
            print('no lr decay \n\n\n')
            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": [
                            param
                            for name, param in model.named_parameters()
                            if (not any(layer in name for layer in differential_layers))
                               and (not any(nd in name for nd in no_decay))
                        ],
                        "lr": cfg.training.learning_rate,
                        "weight_decay": cfg.training.weight_decay,
                    },
                    {
                        "params": [
                            param
                            for name, param in model.named_parameters()
                            if (not any(layer in name for layer in differential_layers))
                               and (any(nd in name for nd in no_decay))
                        ],
                        "lr": cfg.training.learning_rate,
                        "weight_decay": 0,
                    },
                    {
                        "params": [
                            param
                            for name, param in model.named_parameters()
                            if (any(layer in name for layer in differential_layers))
                               and (not any(nd in name for nd in no_decay))
                        ],
                        "lr": cfg.training.differential_learning_rate,
                        "weight_decay": cfg.training.weight_decay,
                    },
                    {
                        "params": [
                            param
                            for name, param in model.named_parameters()
                            if (any(layer in name for layer in differential_layers))
                               and (any(nd in name for nd in no_decay))
                        ],
                        "lr": cfg.training.differential_learning_rate,
                        "weight_decay": 0,
                    },
                ],
                lr=cfg.training.learning_rate,
                weight_decay=cfg.training.weight_decay,
            )

        scheduler = get_scheduler(cfg, optimizer, total_steps)

        if cfg.environment.mixed_precision:
            scaler = GradScaler()

        cfg.curr_step = 0
        i = 0
        best_val_loss = np.inf
        optimizer.zero_grad()
        best_score = np.inf
        for epoch in range(cfg.training.epochs):
            set_seed(cfg.environment.seed + epoch)

            cfg.curr_epoch = epoch
            print("EPOCH:", epoch)
            progress_bar = tqdm(range(len(train_dataloader)))
            tr_it = iter(train_dataloader)
            losses = []
            gc.collect()
            model.train()
            optimizer.zero_grad()
            # print(f'model:  {model}')

            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1
                cfg.curr_step += cfg.training.batch_size
                inputs, labels = next(tr_it)
                batch = cfg.CustomDataset.batch_to_device(inputs, device)
                labels = cfg.CustomDataset.batch_to_device(labels, device)
                if cfg.environment.mixed_precision:
                    with autocast():
                        outputs = model(batch)
                else:
                    outputs = model(batch)

                # cfg.architecture.loss_weights.to(device)
                loss = model.loss_fn(outputs if cfg.architecture.direct_result else outputs.logits, labels)

                losses.append(loss.item())

                if cfg.training.grad_accumulation != 1:
                    loss /= cfg.training.grad_accumulation

                if cfg.environment.mixed_precision:
                    scaler.scale(loss).backward()
                    if i % cfg.training.grad_accumulation == 0:
                        if cfg.training.gradient_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), cfg.training.gradient_clip
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if i % cfg.training.grad_accumulation == 0:
                        if cfg.training.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), cfg.training.gradient_clip
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if cfg.curr_step % cfg.training.batch_size == 0:
                    progress_bar.set_description(
                        f"lr: {np.round(optimizer.param_groups[0]['lr'],7)}, loss: {np.mean(losses[-10:]):.4f}"
                    )

            progress_bar = tqdm(range(len(val_dataloader)))
            val_it = iter(val_dataloader)

            model.eval()
            preds = []
            losses = []
            probabilities = []
            all_targets = []
            for itr in progress_bar:
                inputs, labels = next(val_it)
                batch = cfg.CustomDataset.batch_to_device(inputs, device)
                labels = cfg.CustomDataset.batch_to_device(labels, device)

                if cfg.environment.mixed_precision:
                    with autocast():
                        outputs = model(batch)
                else:
                    outputs = model(batch)

                loss = model.loss_fn(outputs.detach() if cfg.architecture.direct_result else outputs.logits.detach(), labels, cfg.architecture.loss_weights).cpu().numpy()
                losses.append(loss)
            print(f'losses: {losses}')
            metric = np.mean(losses)
            print("Validation metric", metric)
            if metric < best_score:
                best_score = metric
                checkpoint = {"model": model.state_dict()}
                torch.save(checkpoint, f"output/{cfg.experiment_name}/fold{fold}_checkpoint.pth")

            early_stopping(metric, model)
            if early_stopping.early_stop:
                print("Early Stopping")
                break
        del model
        if "cuda" in cfg.device:
            torch.cuda.empty_cache()
        gc.collect()










