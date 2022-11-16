import os
import math
import string
import random
import warnings
warnings.filterwarnings("ignore")
from tqdm.notebook import tqdm

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
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

import gc

# from catboost import CatBoostRegressor, Pool
# from sklearn.linear_model import Ridge
#
# from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split

NotebookConfig = {
    "model_name": "../data/debertav3base",
    "seed": 1111,
    "target_cols": ["cohesion", "syntax", "vocabulary",
                    "phraseology", "grammar", "conventions"],
    "num_classes": 6,
    "num_classes_in_group": 9,

    "use_ensemble_weights": False,
    "ensemble_size": 1,

    # Inference config
    "test_batch_size": 8,
    "max_length": 512,

    "prediction_mode": "weighted_avg",  # operation to get predictions: ["weighted_avg","top"]

    "n_fold": 5,
    "input_path": os.path.abspath(f"./kaggle/temp"),
    "inference_type": "mean",  # ["weighted", "mean", "best","cat_boost", "ridge"]
    "inference_sub_type": "inverse",  # [None, "linear", "inverse"]

    "round_submission": False
}

NotebookConfig["total_num_classes"] = int(NotebookConfig["num_classes"] * NotebookConfig["num_classes_in_group"])
NotebookConfig["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
group_values = np.arange(1.0,5.5,step=0.5)

model_paths = []
for fold in range(0, NotebookConfig["n_fold"]):
    model_paths += [os.path.join(NotebookConfig["input_path"],
                                 f"outputs-{fold}/pytorch_model.bin")]

df = pd.read_csv("../data/feedback-prize-english-language-learning/custom_test.csv")
train_df = pd.read_csv("../data/feedback-prize-english-language-learning/custom_train.csv")


class FeedBackDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.texts = df['full_text'].values

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

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
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
    def __init__(self, model_name, pretrained_path=None, detach_fc=False):
        super(FeedBackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0
        self.model = AutoModel.from_pretrained(model_name, config=self.config)

        self.dropouts_len = 3
        self.dropouts = nn.ModuleList([nn.Dropout(p=0.2 * i) for i in range(1, 1 + self.dropouts_len)])

        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, NotebookConfig['total_num_classes'])

        self.detach_fc = detach_fc

        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path))

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, attention_mask)

        outputs = []

        # Multi-sample dropout
        for i in range(0, self.dropouts_len):
            x = self.dropouts[i](out)

            if (not self.detach_fc):
                x = self.fc(x)

            outputs += [x]

        outputs = torch.stack(outputs, dim=0)
        outputs = torch.mean(outputs, dim=0)

        return SequenceClassifierOutput(logits=outputs)


class EnsembledModel(nn.Module):
    def __init__(self, model_name, ensemble_size, pretrained_path=None, detach_fc=False):
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
        self.detach_fc = detach_fc

        if (NotebookConfig["use_ensemble_weights"]):
            ensemble_weights = 0.9 + 0.2 * torch.rand(self.ensemble_size)
            self.ensemble_weights = torch.nn.Parameter(ensemble_weights)

        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path))

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
        if (not self.detach_fc):
            outputs = self.fc(ensemble_out)
        else:
            outputs = ensemble_out
        return SequenceClassifierOutput(logits=outputs)


def LogitsToPredictions(logits, mode="weighted_avg"):
    logits = torch.from_numpy(logits)
    logits = logits.reshape((logits.shape[0], NotebookConfig["num_classes_in_group"], NotebookConfig["num_classes"]))

    if (mode == "top"):
        predictions = torch.argmax(logits, dim=1) * 0.5 + 1.0

    elif (mode == "weighted_avg"):
        logits_sm = F.softmax(logits, dim=1)
        predictions = (logits_sm * group_values[None, :, None]).sum(dim=1)
        # to class indices
        if (NotebookConfig["round_submission"]):
            predictions = (torch.round((predictions - 1.0) / 0.5).int()) * 0.5 + 1.0

    return predictions.cpu().detach().numpy()


def get_predictions(model_paths, df, device, get_embeddings=False, gbs=None):
    tokenizer = AutoTokenizer.from_pretrained(NotebookConfig['model_name'])
    collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

    dataset = FeedBackDataset(df, tokenizer, max_length=NotebookConfig['max_length'])

    final_preds = []
    for i, path in enumerate(model_paths):
        if (NotebookConfig['ensemble_size'] == 1):
            model = FeedBackModel(NotebookConfig['model_name'], pretrained_path=path, detach_fc=get_embeddings)
        else:
            model = EnsembledModel(NotebookConfig['model_name'], ensemble_size=NotebookConfig['ensemble_size'],
                                   pretrained_path=path, detach_fc=get_embeddings)

        model.to(NotebookConfig['device'])

        print(f"Getting predictions for model {i}")
        training_args = TrainingArguments(
            output_dir=".",
            per_device_eval_batch_size=NotebookConfig['test_batch_size'],
            label_names=["target"]
        )
        trainer = Trainer(model=model,
                          args=training_args,
                          data_collator=collate_fn)
        predictions = trainer.predict(dataset)
        preds = predictions.predictions

        preds = LogitsToPredictions(preds, mode=NotebookConfig["prediction_mode"])

        final_preds.append(preds)

        del model
        _ = gc.collect()

    final_preds = np.array(final_preds)

    if (gbs is not None):
        for i in range(0, NotebookConfig["n_fold"]):
            final_preds[i] = gbs[i].predict(final_preds[i])

    return final_preds


# Inference functions
def fold_mean_inference(model_paths, test_dataset, device, get_embeddings=False, gbs=None):
    print("Fold mean inference")

    final_preds = get_predictions(model_paths, test_dataset, device, get_embeddings, gbs)
    final_preds = np.mean(final_preds, axis=0)
    return final_preds


def fold_weighted_mean_inference(model_paths, test_dataset, device, weights, get_embeddings=False, gbs=None):
    print("Fold weighted mean inference")

    final_preds = get_predictions(model_paths, test_dataset, device, get_embeddings, gbs)

    weights = torch.Tensor(weights)
    weights = F.softmax(weights, dim=0).numpy()

    final_preds = np.average(final_preds, axis=0, weights=weights)
    return final_preds


def best_fold_inference(model_paths, test_dataset, device, best_fold):
    print("Best fold inference")

    best_fold_path = [model_paths[best_fold]]
    final_preds = get_predictions(best_fold_path, test_dataset, device)
    return final_preds[0]


def cat_boost_inference(model_paths, test_dataset, device, cat_boost_model, get_embeddings=False, gbs=None):
    print("Cat boosted inference")

    final_preds = get_predictions(model_paths, test_dataset, device, get_embeddings, gbs)
    final_preds = np.concatenate(final_preds, axis=1)

    final_preds = cat_boost_model.predict(final_preds)
    return final_preds


def ridge_inference(model_paths, test_dataset, device, ridge_model, get_embeddings=False, gbs=None):
    print("Ridge inference")

    final_preds = get_predictions(model_paths, test_dataset, device, get_embeddings, gbs)
    final_preds = np.concatenate(final_preds, axis=1)

    final_preds = ridge_model.predict(final_preds)
    return final_preds


if (NotebookConfig["inference_type"] in ["cat_boost", "ridge"]):
    # train_preds = get_predictions(model_paths, train_df, NotebookConfig['device'], get_embeddings=True)
    # you can use saved predictions or/and embeddings here
    train_preds = np.load("../input/deberta-predictions/pedictions.npy")

    # train_preds = np.concatenate(train_preds,axis=1)
    print("train_preds.shape: ", train_preds.shape)

    train_labels = train_df[NotebookConfig["target_cols"]]
    print("train_labels.shape: ", train_labels.shape)

# display fold losses, find best fold and initialize weights
best_eval_rmses = []

if (NotebookConfig["prediction_mode"] == "top"):
    filter_metric = "eval_rmse_top"
elif (NotebookConfig["prediction_mode"] == "weighted_avg"):
    filter_metric = "eval_rmse_wa"

for fold in range(0, NotebookConfig["n_fold"]):
    fold_history_path = os.path.join(NotebookConfig["input_path"], f"history/hsitory_fold_{fold}.csv")
    fold_history_rmse = pd.read_csv(fold_history_path)[filter_metric]

    fold_min_rmse = np.min(fold_history_rmse)
    best_eval_rmses += [fold_min_rmse]

best_eval_rmses_df = pd.DataFrame(
    {"fold": list(range(0, NotebookConfig["n_fold"])), "best_eval_rmses": best_eval_rmses})
# display(best_eval_rmses_df)

best_eval_rmse = np.min(best_eval_rmses_df["best_eval_rmses"])
best_fold = best_eval_rmses_df[best_eval_rmses_df["best_eval_rmses"] == best_eval_rmse]
print(f"Fold {best_fold.fold.item()} has best eval rmse {best_eval_rmse}.")

weights = []
if (NotebookConfig["inference_type"] == "weighted"):
    if (NotebookConfig["inference_sub_type"] == "inverse"):
        # 1/rmse
        weights = 1 / (1 + (best_eval_rmses_df["best_eval_rmses"] - best_eval_rmse) * 10)
    elif (NotebookConfig["inference_sub_type"] == "linear"):
        # linear:  f(rmse)=k*rmse+d; f(max_rmse)=0; f(min_rmse)=1
        a = best_eval_rmse  # best fold
        b = np.max(best_eval_rmses_df["best_eval_rmses"])  # worst fold

        k = -1 / (b - a)
        d = b / (b - a)
        weights = k * best_eval_rmses_df["best_eval_rmses"] + d

    print("Weights: ", weights)

preds=None
if(NotebookConfig["inference_type"]=="best"):
    preds = best_fold_inference(model_paths, df, NotebookConfig['device'], best_fold=best_fold.fold.item())
elif(NotebookConfig["inference_type"]=="mean"):
    preds = fold_mean_inference(model_paths, df, NotebookConfig['device'])
elif(NotebookConfig["inference_type"]=="weighted"):
    preds = fold_weighted_mean_inference(model_paths, df, NotebookConfig['device'],weights)


if(NotebookConfig["round_submission"]):
    preds=np.rint(preds/0.5)*0.5
    preds
preds.to_csv('submission.csv', index=False)