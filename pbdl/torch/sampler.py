"""
TODO sub package description
"""

# non-local package imports
import numpy as np
import pbdl.torch.phi.dataset
import torch.utils.data

# local class imports
import pbdl.torch.phi.dataset


class ConstantBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self,
        dataset: pbdl.torch.phi.dataset.Dataset,
        batch_size=None,
        group_constants=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_constants = group_constants

        self.groups = self.group_by_constants()

    def group_by_constants(self):
        print("Prepare constant batch sampling...")

        groups = {}
        for idx, (_, _, _, constants) in enumerate(self.dataset):
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
