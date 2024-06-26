"""
This module provides classes for data loading that do not require any dependencies other than numpy.
"""

# non-local package imports
import numpy as np
from multiprocessing import Pool

# local class imports
from pbdl.dataset import Dataset

def _collate_fn_(batch):
    """
    Concatenates data arrays with inflated constant layers and stacks them into batches.

    Returns:
        numpy.ndarray: Data batch array
        numpy.ndarray: Target batch array
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

    return data, targets


class Dataloader:
    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None, num_workers=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.num_workers = num_workers

    def __iter__(self):
        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0

        if self.num_workers > 1:
            pass # TODO dataset must support multiprocessing first
            # return self.__iter_multiprocessing()
        else:
            return self

    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch = [self.dataset[idx] for idx in batch_indices]
        self.current_idx += self.batch_size
        if self.collate_fn:
            return self.collate_fn(batch)
        return batch
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    # parallel processing
    def __iter_multiprocessing(self):
        with Pool(self.num_workers) as pool:
            batch_indices = [
                self.indices[i:i + self.batch_size]
                for i in range(0, len(self.indices), self.batch_size)
            ]
            for batch in pool.imap(self._load_data, batch_indices):
                if self.collate_fn:
                    yield self.collate_fn(batch)
                else:
                    yield batch

    def _load_data(self, indices):
        return [self.dataset[idx] for idx in indices]