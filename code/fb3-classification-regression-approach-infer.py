import os
import math
import string
import random
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from IPython.display import display

# import matplotlib.pyplot as plt
# import seaborn as sns

import torch
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers import EarlyStoppingCallback, TrainerCallback
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

import gc

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

NotebookConfig = {
    "model_name": "../data/debertav3base",
    "seed": 42,
    "target_cols": ["cohesion", "syntax", "vocabulary",
                    "phraseology", "grammar", "conventions"],
    "num_classes": 6,
    "num_classes_in_group": 9,

    "prediction_mode": "weighted_avg",  # operation to get predictions: ["weighted_avg","top"]

    "loss_weights": {
        "cross_entropy": 0.1,
        "cross_entropy_smooth": 0,
        "mse": 0.9,
        "smooth_l1": 0,
    },

    "decay_loss": True,
    "decay_min_weights": {
        # "mse": 0.7,
        # "smooth_l1": 0.05,
        # "cross_entropy_smooth": 0.7,
        "cross_entropy": 0.01
    },
    "loss_decay_mode": "cos",
    "decay_epochs": 0.4,

    "ensemble_size": 1,
    "use_ensemble_weights": False,  # weights are not necessary

    "dynamic_seed": True,

    # Training config
    "epochs": 10,
    "learning_rate": 8e-6,
    "min_lr": 1e-6,
    "weight_decay": 1e-3,
    "train_batch_size": 8,
    "valid_batch_size": 8,
    "max_length": 512,
    "n_fold": 5,
    "n_accumulate": 1,
    "max_grad_norm": 100,
    "early_stopping_patience": 5,
    "eval_per_epoch": 1
}

NotebookConfig["total_num_classes"] = int(NotebookConfig["num_classes"] * NotebookConfig["num_classes_in_group"])
NotebookConfig["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(NotebookConfig["seed"])

df=pd.read_csv("../data/feedback-prize-english-language-learning/custom_train.csv")
df[NotebookConfig["target_cols"]]=(df[NotebookConfig["target_cols"]]//0.5-2).astype(np.int)

#label values
group_values_cpu = torch.arange(1.0,5.5,step=0.5)
group_values_gpu=group_values_cpu.to(NotebookConfig["device"])


class FeedBackDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.texts = df['full_text'].values
        self.targets = df[NotebookConfig['target_cols']].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.texts[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len
        )
        targets = self.targets[index]

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'target': targets
        }


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0
        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        self.dropouts_len = 3
        self.dropouts = nn.ModuleList([nn.Dropout(p=0.2 * i) for i in range(1, 1 + self.dropouts_len)])

        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, NotebookConfig['total_num_classes'])

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, attention_mask)

        outputs = []

        # Multi-sample dropout
        for i in range(0, self.dropouts_len):
            x = self.dropouts[i](out)
            x = self.fc(x)

            outputs += [x]

        outputs = torch.stack(outputs, dim=0)
        outputs = torch.mean(outputs, dim=0)

        return SequenceClassifierOutput(logits=outputs)


class EnsembledModel(nn.Module):
    def __init__(self, model_name, ensemble_size):
        super(EnsembledModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0
        self.ensemble_size = ensemble_size

        # ensembles
        self.models = nn.ModuleList(
            [AutoModel.from_pretrained(model_name, config=self.config) for i in range(0, self.ensemble_size)])
        self.drops = nn.ModuleList([nn.Dropout(p=0.2) for i in range(0, self.ensemble_size)])
        self.poolers = nn.ModuleList([MeanPooling() for i in range(0, self.ensemble_size)])

        self.fc = nn.Linear(self.config.hidden_size * self.ensemble_size, NotebookConfig['total_num_classes'])

        if (NotebookConfig["use_ensemble_weights"]):
            ensemble_weights = 0.9 + 0.2 * torch.rand(self.ensemble_size)
            self.ensemble_weights = torch.nn.Parameter(ensemble_weights)

    def forward(self, input_ids, attention_mask):

        ensemble_out = None

        if (NotebookConfig["use_ensemble_weights"]):
            ensemble_weights_sm = F.softmax(self.ensemble_weights, dim=0)

        for i in range(0, self.ensemble_size):
            out = self.models[i](input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 output_hidden_states=False)
            out = self.poolers[i](out.last_hidden_state, attention_mask)
            out = self.drops[i](out)

            if (NotebookConfig["use_ensemble_weights"]):
                out = out * ensemble_weights_sm[i]

            if (ensemble_out is None):
                ensemble_out = out
            else:
                ensemble_out = torch.cat((ensemble_out, out), dim=1)

        outputs = self.fc(ensemble_out)
        return SequenceClassifierOutput(logits=outputs)


# CrossEntropy loss for each group of one-hoted class
class GroupCrossEntropy(nn.Module):
    def __init__(self, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.sm_ce = nn.CrossEntropyLoss(reduction=reduction,
                                         label_smoothing=label_smoothing)

    def forward(self, y_pred, y_true):
        y_pred = y_pred.reshape(
            (y_pred.shape[0], NotebookConfig["num_classes_in_group"], NotebookConfig["num_classes"]))
        # y_true=y_true.reshape((y_true.shape[0],NotebookConfig["num_classes"]))

        loss = self.sm_ce(y_pred, y_true)

        return loss


# MSE-like loss for top value of each group of one-hoted class
class GroupMSE(nn.Module):
    def __init__(self, reduction='mean', mode="top", mse_loss=nn.MSELoss):
        super().__init__()

        self.sm = nn.Softmax(dim=1)
        self.mse = mse_loss(reduction=reduction)
        self.mode = mode

    def forward(self, y_pred, y_true):

        y_pred = y_pred.reshape(
            (y_pred.shape[0], NotebookConfig["num_classes_in_group"], NotebookConfig["num_classes"]))
        # y_true=y_true.reshape((y_true.shape[0],NotebookConfig["num_classes"]))

        y_pred_sm = self.sm(y_pred)

        if (self.mode == "top"):
            top_y_pred_sm = torch.argmax(y_pred_sm, dim=1)

            top_y_pred_sm = (top_y_pred_sm * 0.5) + 1.0
            y_pred_vals = top_y_pred_sm

        elif (self.mode == "weighted_avg"):
            if (y_pred_sm.is_cuda):
                weighted_avg_y_pred_sm = (y_pred_sm * group_values_gpu[None, :, None]).sum(dim=1)
            else:
                weighted_avg_y_pred_sm = (y_pred_sm * group_values_cpu[None, :, None]).sum(dim=1)

            y_pred_vals = weighted_avg_y_pred_sm

        y_true = (y_true * 0.5) + 1.0
        loss = self.mse(y_pred_vals, y_true)
        return loss


def loss_fn(outputs, labels, loss_weights, reduction='mean'):
    total_loss = 0.0

    if (loss_weights["cross_entropy"] != 0):
        loss_func = GroupCrossEntropy(reduction=reduction, label_smoothing=loss_weights["cross_entropy_smooth"])
        total_loss += loss_weights["cross_entropy"] * loss_func(outputs.float(), labels)

    if (loss_weights["mse"] != 0):
        gmse_loss = GroupMSE(reduction=reduction,
                             mode=NotebookConfig["prediction_mode"], mse_loss=nn.MSELoss)
        total_loss += loss_weights["mse"] * gmse_loss(outputs.float(), labels.float())

    if (loss_weights["smooth_l1"] != 0):
        gsmooth_l1_loss = GroupMSE(reduction=reduction,
                                   mode=NotebookConfig["prediction_mode"], mse_loss=nn.SmoothL1Loss)
        total_loss += loss_weights["smooth_l1"] * gsmooth_l1_loss(outputs.float(), labels.float())

    return total_loss


# mean accuracy for all classes
def accuracy_metric(logits, labels, mode="top"):
    logits = logits.reshape((logits.shape[0], NotebookConfig["num_classes_in_group"], NotebookConfig["num_classes"]))

    if (mode == "top"):
        predictions = torch.argmax(logits, dim=1)

    elif (mode == "weighted_avg"):
        logits_sm = F.softmax(logits, dim=1)
        predictions = (logits_sm * group_values_cpu[None, :, None]).sum(dim=1)
        # to class indices
        predictions = torch.round((predictions - 1.0) / 0.5).int()

    corrects = (predictions == labels)

    accuracy = corrects.sum().float() / float(labels.shape[0] * NotebookConfig["num_classes"])
    return accuracy


def compute_metrics(p):
    predictions, labels = map(torch.from_numpy, p)

    loss_func = lambda logits, labels: loss_fn(logits, labels,
                                               loss_weights=NotebookConfig['loss_weights'], reduction="none")
    loss = torch.mean(loss_func(predictions.float(), labels))

    gmse = GroupMSE(reduction="none", mode=NotebookConfig["prediction_mode"])
    gmse_loss = torch.sqrt(gmse(predictions.float(), labels.float()) + 1e-9)

    metrics = {}
    rmse_postfix = None

    if (NotebookConfig["prediction_mode"] == "weighted_avg"):
        rmse_postfix = "wa"
    elif (NotebookConfig["prediction_mode"] == "top"):
        rmse_postfix = "top"

    metrics[f"cohesion_rmse_{rmse_postfix}"] = torch.mean(gmse_loss[:, 0])
    metrics[f"syntax_rmse_{rmse_postfix}"] = torch.mean(gmse_loss[:, 1])
    metrics[f"vocabulary_rmse_{rmse_postfix}"] = torch.mean(gmse_loss[:, 2])
    metrics[f"phraseology_rmse_{rmse_postfix}"] = torch.mean(gmse_loss[:, 3])
    metrics[f"grammar_rmse_{rmse_postfix}"] = torch.mean(gmse_loss[:, 4])
    metrics[f"conventions_rmse_{rmse_postfix}"] = torch.mean(gmse_loss[:, 5])

    metrics["accuracy_top"] = accuracy_metric(predictions, labels,
                                              mode="top")
    metrics["accuracy_wa"] = accuracy_metric(predictions, labels,
                                             mode="weighted_avg")

    gmse = GroupMSE(reduction="mean", mode="top")
    metrics["rmse_top"] = torch.sqrt(gmse(predictions.float(), labels.float()) + 1e-9)

    gmse = GroupMSE(reduction="mean", mode="weighted_avg")
    metrics["rmse_wa"] = torch.sqrt(gmse(predictions.float(), labels.float()) + 1e-9)

    return metrics


# callback to decay weights of losses
class LossDecayCallback(TrainerCallback):
    def __init__(self, decay_epochs, mode="linear"):
        super().__init__()

        self.skip_decay = True

        self.weights_by_epoch = {}

        self.epochs = decay_epochs

        # min weights as a fraction of the original weights
        min_weights_relative = NotebookConfig["decay_min_weights"]

        shifted = False
        for weight in min_weights_relative:
            if (NotebookConfig['loss_weights'][weight] > 0):
                max_weight = NotebookConfig['loss_weights'][weight]
                min_weight = max_weight * min_weights_relative[weight]

                x = np.array(list(range(0, self.epochs)))

                if (mode == "linear"):
                    step = (max_weight - min_weight) / float(self.epochs)
                    self.weights_by_epoch[weight] = x * step + min_weight

                elif (mode == "cos"):
                    self.weights_by_epoch[weight] = (1.0 - np.sin((np.pi / 2) * x / self.epochs)) * (
                                max_weight - min_weight) + min_weight

                elif (mode == "min_max"):
                    if (shifted):
                        self.weights_by_epoch[weight] = [max_weight if i % 2 == 0 else min_weight for i in x]
                    else:
                        self.weights_by_epoch[weight] = [min_weight if i % 2 == 0 else max_weight for i in x]

                    shifted = not shifted

                elif (mode == "even_cos"):
                    cos_weights = (1.0 - np.sin((np.pi / 2) * x / self.epochs)) * (max_weight - min_weight) + min_weight

                    self.weights_by_epoch[weight] = [cos_weights[i] if i % 2 == int(shifted) else 0 for i in x]

                    shifted = not shifted

        if (len(self.weights_by_epoch) != 0):
            self.skip_decay = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        if (not self.skip_decay):
            if (self.current_epoch < self.epochs):
                for weight in self.weights_by_epoch:
                    NotebookConfig['loss_weights'][weight] = self.weights_by_epoch[weight][self.current_epoch]

                self.current_epoch += 1

    def on_train_begin(self, args, state, control, **kwargs):
        self.current_epoch = 0
        display(self.weights_by_epoch)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        loss = loss_fn(outputs.logits, inputs['target'], loss_weights=NotebookConfig['loss_weights'])
        return (loss, outputs) if return_outputs else loss


# use temp dir to avoid exceeding of allocated output dir space
temp_dir = "./kaggle/temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

mskf = MultilabelStratifiedKFold(n_splits=NotebookConfig['n_fold'], shuffle=True, random_state=NotebookConfig['seed'])

tokenizer = AutoTokenizer.from_pretrained(NotebookConfig['model_name'])
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

history = []
ensemble_weights_history = []
rnd_seed = NotebookConfig['seed']

for fold, (train_ids, val_ids) in enumerate(mskf.split(X=df, y=df[NotebookConfig['target_cols']])):
    print(f"========== Fold: {fold} ==========")

    if (NotebookConfig['dynamic_seed']):
        rnd_seed = NotebookConfig['seed'] + fold * 11
        set_seed(rnd_seed)

    df_train = df.iloc[train_ids, :]
    df_valid = df.iloc[val_ids, :]

    train_dataset = FeedBackDataset(df_train, tokenizer=tokenizer, max_length=NotebookConfig['max_length'])
    valid_dataset = FeedBackDataset(df_valid, tokenizer=tokenizer, max_length=NotebookConfig['max_length'])

    if (NotebookConfig['ensemble_size'] == 1):
        model = FeedBackModel(NotebookConfig['model_name'])
    else:
        model = EnsembledModel(NotebookConfig['model_name'], ensemble_size=NotebookConfig['ensemble_size'])
    model.to(NotebookConfig['device'])

    # Define Optimizer and Scheduler
    param_optimizer = list(model.named_parameters())

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay) and n != "fc.weight")],
            "weight_decay": NotebookConfig['weight_decay'],
            "lr": NotebookConfig['learning_rate']
        },
        {
            "params": [p for n, p in param_optimizer if (not any(nd in n for nd in no_decay) and n == "fc.weight")],
            "weight_decay": NotebookConfig['weight_decay'],
            "lr": NotebookConfig['learning_rate'] / 10
        },
        {
            "params": [p for n, p in param_optimizer if (any(nd in n for nd in no_decay) and n == "fc.bias")],
            "weight_decay": 0.0,
            "lr": NotebookConfig['learning_rate'] / 10
        },
        {
            "params": [p for n, p in param_optimizer if (any(nd in n for nd in no_decay) and n != "fc.bias")],
            "weight_decay": 0.0,
            "lr": NotebookConfig['learning_rate']
        },
    ]
    optimizer = AdamW(optimizer_parameters, lr=NotebookConfig['learning_rate'])

    num_training_steps_per_epoch = len(train_dataset) // (
                NotebookConfig['train_batch_size'] * NotebookConfig['n_accumulate'])
    num_training_steps = num_training_steps_per_epoch * NotebookConfig['epochs']
    num_eval_steps_per_epoch = num_training_steps_per_epoch // NotebookConfig["eval_per_epoch"]

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=2 * num_training_steps_per_epoch,
        num_training_steps=num_training_steps
    )

    callbacks = []

    early_stopping = EarlyStoppingCallback(early_stopping_patience=NotebookConfig['early_stopping_patience'])
    callbacks += [early_stopping]

    if (NotebookConfig["decay_loss"]):
        loss_decay = LossDecayCallback(int(NotebookConfig['epochs'] * NotebookConfig['decay_epochs']),
                                       mode=NotebookConfig["loss_decay_mode"])
        callbacks += [loss_decay]

    training_args = TrainingArguments(
        output_dir=os.path.join(temp_dir, f"outputs-{fold}/"),
        evaluation_strategy="steps",
        eval_steps=num_eval_steps_per_epoch,
        per_device_train_batch_size=NotebookConfig['train_batch_size'],
        per_device_eval_batch_size=NotebookConfig['valid_batch_size'],
        num_train_epochs=NotebookConfig['epochs'],
        learning_rate=NotebookConfig['learning_rate'],
        weight_decay=NotebookConfig['weight_decay'],
        gradient_accumulation_steps=NotebookConfig['n_accumulate'],
        max_grad_norm=NotebookConfig['max_grad_norm'],
        seed=rnd_seed,
        group_by_length=True,
        metric_for_best_model='eval_rmse_wa',
        load_best_model_at_end=True,
        greater_is_better=False,
        save_strategy="steps",
        save_steps=num_eval_steps_per_epoch,
        save_total_limit=1,
        report_to="none",
        label_names=["target"],
        logging_steps=num_eval_steps_per_epoch
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    trainer.train()
    trainer.save_model()

    if (NotebookConfig['ensemble_size'] != 1 and NotebookConfig["use_ensemble_weights"]):
        for name, param in model.named_parameters():
            if (name == "ensemble_weights"):
                ensemble_weights = param.cpu().detach().numpy()
                ensemble_weights_history += [ensemble_weights]
                break

    fold_history = trainer.state.log_history
    history += [fold_history]

    del model
    _ = gc.collect()
    torch.cuda.empty_cache()

#show fold best val rmse
best_eval_rmses=[]
for fold_history in history:
    fold_best_eval_rmse=np.min(pd.DataFrame(fold_history)["eval_rmse_wa"])
    best_eval_rmses+=[fold_best_eval_rmse]

best_eval_rmses_df=pd.DataFrame({"fold":list(range(0,NotebookConfig["n_fold"])),"best_eval_rmses":best_eval_rmses})
display(best_eval_rmses_df)

best_fold=best_eval_rmses_df[best_eval_rmses_df["best_eval_rmses"]==np.min(best_eval_rmses_df["best_eval_rmses"])]
print(f"Fold {best_fold.fold.item()} has best eval rmse {best_fold.best_eval_rmses.item()}.")