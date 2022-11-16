import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import SequenceClassifierOutput


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        # print(f'inputs  shape: {last_hidden_state.shape} attension shape{attention_mask.shape}')
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = -1e4
        max_embeddings, _ = torch.max(embeddings, dim=1)
        return max_embeddings


class MinPooling(nn.Module):
    def __init__(self):
        super(MinPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = last_hidden_state.clone()
        embeddings[input_mask_expanded == 0] = 1e-4
        min_embeddings, _ = torch.min(embeddings, dim=1)
        return min_embeddings


# Attention pooling
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask == 0] = float('-inf')
        w = torch.softmax(w, 1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings


# There may be a bug in my implementation because it does not work well.
class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 8, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, ft_all_layers):
        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start:, :, :, :]

        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        return weighted_average


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        self.config = AutoConfig.from_pretrained(cfg.architecture.model_name)
        self.config.hidden_dropout_prob = 0
        self.config.attention_probs_dropout_prob = 0
        self.model = AutoModel.from_pretrained(cfg.architecture.model_name, config=self.config)

        self.dropouts_len = 3
        self.dropouts = nn.ModuleList([nn.Dropout(p=0.2 * i) for i in range(1, 1 + self.dropouts_len)])

        self.mean_pooler = MeanPooling()
        self.weighted_layer_pooler = WeightedLayerPooling(self.config.num_hidden_layers)
        self.fc = nn.Linear(self.config.hidden_size, cfg.architecture.total_num_classes)
        self.softmax = nn.Softmax(dim=0)
        self.fc1 = nn.Linear(4, 1)
        self.fc2 = nn.Linear(self.config.hidden_size, cfg.architecture.num_classes)
        self.fc3 = nn.Linear(cfg.architecture.num_classes, cfg.architecture.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.cfg = cfg

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         output_hidden_states=True)
        # print(f'out.state.shape: {len(out.hidden_states)}')
        # print(f'out.state.shape: {out.last_hidden_state.shape}')
        stack_meanpool = torch.stack([self.mean_pooler(hidden_s, attention_mask) for hidden_s in out.hidden_states[-4:]], axis=2)
        weighted_layer_pool = self.fc1(self.softmax(stack_meanpool)).squeeze(axis=-1)
        six_class_output = self.fc2(weighted_layer_pool)

        return six_class_output


    # @staticmethod
    def loss_fn(self, outputs, labels, loss_weights=None, reduction='mean'):
        print(f'output : {outputs}')
        print(f' labels: {labels}')
        return nn.SmoothL1Loss(reduction='mean')(outputs, labels)

def predict(outputs, cfg):
    return outputs.detach().cpu().numpy()

