from pbdl.torch.dataset import Dataset

import numpy as np
from torch.utils.data import BatchSampler


class PBDLConstantBatchSampler(BatchSampler):
    def __init__(self, dataset: PBDLDataset, batch_size, group_constants=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_constants = group_constants

        self.groups = self.group_by_constants()

    def group_by_constants(self):
        groups = {}
        for idx in range(len(self.dataset)):
            _, constants, _ = self.dataset[idx]

            if not self.group_constants is None:
                constants = tuple(
                    [constants[i] for i in self.group_constants]
                )  # project constants to self.group_constants

            if constants not in groups:
                groups[constants] = []
            groups[constants].append(idx)
        return list(groups.values())

    def __iter__(self):
        for group in self.groups:
            for batch in [
                group[i : i + self.batch_size]
                for i in range(0, len(group), self.batch_size)
            ]:
                yield batch

    def __len__(self):
        return sum(len(group) // self.batch_size for group in self.groups)
