"""
TODO sub package description
"""

# non-local package imports
import numpy as np
import torch

# local class imports
from pbdl.dataset import Dataset

# non-local class imports
import torch.utils.data


def collate_fn(batch):
    data = np.stack(  # stack batch items
        [
            np.concatenate(  # concatenate data and constant layers
                [item[0]]  # data
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
            kwargs["collate_fn"] = collate_fn

        super().__init__(*args, **kwargs)
