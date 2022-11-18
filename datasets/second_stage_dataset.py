import collections
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        # print("aa".s)
        input_columns = [column for column in df.columns if any(column.startswith(item + '_label') or column.startswith(item + '_prob') for item in cfg.dataset.label_columns)]
        self.cfg.dataset.input_size = len(input_columns)
        self.inputs = df[input_columns].values
        self.labels = df[cfg.dataset.label_columns].values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        inputs = torch.tensor(self.inputs[item], dtype=torch.float)
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label

    @staticmethod
    def batch_to_device(batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, collections.abc.Mapping):
            return {
                key: CustomDataset.batch_to_device(value, device)
                for key, value in batch.items()
            }
        elif isinstance(batch, collections.abc.Sequence):
            return [CustomDataset.batch_to_device(value, device) for value in batch]
        else:
            raise ValueError(f"Can not move {type(batch)} to device.")


class CustomInferDataset(Dataset):
    def __init__(self, df, cfg):
        self.cfg = cfg
        # print("aa".s)
        input_columns = [column for column in df.columns if any(column.startswith(item + '_label') or column.startswith(item + '_prob') for item in cfg.dataset.label_columns)]
        self.cfg.dataset.input_size = len(input_columns)
        self.inputs = df[input_columns].values

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        inputs = torch.tensor(self.inputs[item], dtype=torch.float)
        return inputs

    @staticmethod
    def batch_to_device(batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, collections.abc.Mapping):
            return {
                key: CustomDataset.batch_to_device(value, device)
                for key, value in batch.items()
            }
        elif isinstance(batch, collections.abc.Sequence):
            return [CustomDataset.batch_to_device(value, device) for value in batch]
        else:
            raise ValueError(f"Can not move {type(batch)} to device.")