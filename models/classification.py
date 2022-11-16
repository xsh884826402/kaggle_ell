import torch
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

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


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.config = AutoConfig.from_pretrained(cfg.architecture.model_name)
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0
        self.model = AutoModel.from_pretrained(cfg.architecture.model_name, config=self.config)

        self.dropouts_len = 3
        self.dropouts = nn.ModuleList([nn.Dropout(p=0.2 * i) for i in range(1, 1 + self.dropouts_len)])

        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, cfg.architecture.total_num_classes)
        self.cfg = cfg

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


    # @staticmethod
    def loss_fn(self, outputs, labels, loss_weights, reduction='mean'):
        total_loss = 0.0
        labels = labels.long()
        #  to do 是否进行类型转换 if  self.cfg.device
        if self.cfg.device.startswith("cuda"):
            label_smoothing_para = torch.tensor(loss_weights["cross_entropy_smooth"]).cuda()
        else:
            label_smoothing_para = loss_weights["cross_entropy_smooth"]
        if loss_weights["cross_entropy"] != 0:
            loss_func = GroupCrossEntropy(self.cfg, reduction=reduction,
                                          label_smoothing=label_smoothing_para)
            total_loss += loss_weights["cross_entropy"] * loss_func(outputs.float(), labels.long())

        if loss_weights["mse"] != 0:
            gmse_loss = GroupMSE(self.cfg, reduction=reduction,
                                 mode=self.cfg.architecture.prediction_mode, mse_loss=nn.MSELoss)
            total_loss += loss_weights["mse"] * gmse_loss(outputs.float(), labels.float())

        if (loss_weights["smooth_l1"] != 0):
            gsmooth_l1_loss = GroupMSE(self.cfg, reduction=reduction,
                                       mode=self.cfg.architecture.prediction_mode, mse_loss=nn.SmoothL1Loss)
            total_loss += loss_weights["smooth_l1"] * gsmooth_l1_loss(outputs.float(), labels.float())

        return total_loss


# CrossEntropy loss for each group of one-hoted class
class GroupCrossEntropy(nn.Module):
    def __init__(self, cfg, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.cfg = cfg
        self.sm_ce = nn.CrossEntropyLoss(reduction=reduction,
                                         label_smoothing=label_smoothing)

    def forward(self, y_pred, y_true):
        if self.cfg.device.startswith("cuda"):
            num_classes_in_group = torch.tensor(self.cfg.architecture.num_classes_in_group).cuda()
            num_classes = torch.tensor(self.cfg.architecture.num_classes).cuda()
        else:
            num_classes_in_group = self.cfg.architecture.num_classes_in_group
            num_classes = self.cfg.architecture.num_classes

        y_pred = y_pred.reshape(
            (y_pred.shape[0], num_classes_in_group, num_classes))
        # y_true=y_true.reshape((y_true.shape[0],NotebookConfig["num_classes"]))

        loss = self.sm_ce(y_pred, y_true)

        return loss


# MSE-like loss for top value of each group of one-hoted class
class GroupMSE(nn.Module):
    def __init__(self, cfg, reduction='mean', mode="top", mse_loss=nn.MSELoss):
        super().__init__()

        self.sm = nn.Softmax(dim=1)
        self.mse = mse_loss(reduction=reduction)
        self.mode = mode
        self.cfg = cfg

    def forward(self, y_pred, y_true):

        if self.cfg.device.startswith("cuda"):
            num_classes_in_group = torch.tensor(self.cfg.architecture.num_classes_in_group).cuda()
            num_classes = torch.tensor(self.cfg.architecture.num_classes).cuda()
        else:
            num_classes_in_group = self.cfg.architecture.num_classes_in_group
            num_classes = self.cfg.architecture.num_classes

        y_pred = y_pred.reshape(
            (y_pred.shape[0], num_classes_in_group, num_classes))
        # y_true=y_true.reshape((y_true.shape[0],NotebookConfig["num_classes"]))

        y_pred_sm = self.sm(y_pred)

        if (self.mode == "top"):
            top_y_pred_sm = torch.argmax(y_pred_sm, dim=1)

            top_y_pred_sm = (top_y_pred_sm * 0.5) + 1.0
            y_pred_vals = top_y_pred_sm

        elif (self.mode == "weighted_avg"):
            group_values_cpu = torch.arange(1.0, 5.5, step=0.5)
            group_values_gpu = group_values_cpu.to(self.cfg.device)

            if (y_pred_sm.is_cuda):
                weighted_avg_y_pred_sm = (y_pred_sm * group_values_gpu[None, :, None]).sum(dim=1)
            else:
                weighted_avg_y_pred_sm = (y_pred_sm * group_values_cpu[None, :, None]).sum(dim=1)

            y_pred_vals = weighted_avg_y_pred_sm

        y_true = (y_true * 0.5) + 1.0
        loss = self.mse(y_pred_vals, y_true)
        return loss


def predict(outputs, cfg):
    outputs = outputs.reshape(outputs.shape[0], cfg.architecture.num_classes_in_group, cfg.architecture.num_classes)
    outputs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
    results = outputs.reshape(outputs.shape[0], cfg.architecture.num_classes_in_group, cfg.architecture.num_classes)
    results_label = np.argmax(results, axis=1)
    results_probs = np.max(results, axis=1)
    return results_label, results_probs


