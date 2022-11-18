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
        self.cfg = cfg
        self.sizes = [256, 128, 64]
        self.sizes = [256, 128, 64]

        self.features = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(cfg.dataset.input_size, self.sizes[0])),
            nn.PReLU(),
            nn.Linear(self.sizes[0], self.sizes[1]),
            nn.PReLU(),
            nn.Linear(self.sizes[1], self.sizes[2]),
            nn.PReLU(),
        )

        self.head = nn.Linear(self.sizes[-1], 6)

    def forward(self, inputs):
        x = self.features(inputs)
        x = self.head(x)
        return x


    # @staticmethod
    def loss_fn(self, outputs, labels, loss_weights=None, reduction='mean'):
        return nn.MSELoss()(outputs, labels)

def predict(outputs, cfg):
    return outputs.detach().cpu().numpy()


