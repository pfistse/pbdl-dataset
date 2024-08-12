"""
This module provides classes for data loading that are compatible with PyTorch and PhiFlow.
"""

import numpy as np
import torch
import pbdl.torch.loader

# local class imports
from pbdl.torch.phi.dataset import Dataset
from pbdl.torch.sampler import ConstantBatchSampler


def _collate_fn_solver_(batch):
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

    return torch.tensor(data), torch.tensor(targets), batch[0][3] # all constants are equal


class Dataloader(pbdl.torch.loader.Dataloader):
    def __init__(self, *args, batch_size=None, group_constants=None, **kwargs):

        if "collate_fn" not in kwargs:
            kwargs["collate_fn"] = _collate_fn_solver_

        kwargs["batch_sampler"] = ConstantBatchSampler(
            args[0], batch_size=batch_size, group_constants=group_constants
        )

        super().__init__(*args, **kwargs)
