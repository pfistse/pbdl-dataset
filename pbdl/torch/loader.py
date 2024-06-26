"""
This module provides classes for data loading that are compatible with PyTorch.
"""
# non-local package imports
import numpy as np
import torch
import torch.utils.data

# local class imports
from pbdl.dataset import Dataset
from pbdl.torch.sampler import ConstantBatchSampler


def _collate_fn_(batch):
    """
    Concatenates data arrays with inflated constant layers and stacks them into batches.

    Returns:
        torch.Tensor: Data batch tensor
        torch.Tensor: Target batch tensor
    """

    data = np.stack(  # stack batch items
        [
            np.concatenate(  # concatenate data and constant layers
                [item[0]]
                + [
                    [
                        np.full_like(
                            item[0][0], constant
                        )  # inflate constants to constant layers
                    ]
                    for constant in item[1]
                ],
                axis=0,
            )
            for item in batch
        ],
        axis=0,
    )

    targets = np.stack([item[2] for item in batch])

    return torch.tensor(data), torch.tensor(targets)


class Dataloader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):

        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = _collate_fn_

        super().__init__(*args, **kwargs)
