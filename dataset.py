import os
import urllib.request
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np

from itertools import groupby

# TODO load endpoints from server?
dataset_metadata = {
    # "karman-2d-train": {
    #     "endpoint": "https://physicsbaseddeeplearning.org/data/sol-karman-2d-train.pickle",
    #     "fields": "VVp",
    # },
    # "karman-2d-test": {
    #     "endpoint": "https://physicsbaseddeeplearning.org/data/sol-karman-2d-test.pickle",
    #     "fields": "VVp",
    # },
    "transonic-cylinder-flow-tiny": {
        "endpoint": "https://syncandshare.lrz.de/dl/fiYHg3ibgBSL6Hi7u7w5ND/transonic_cylinder_flow_tiny.npz",
        "fields": "VVdp",
    },
}

DATASET_DIR = "datasets/"
DATASET_EXT = ".npz"


class PBDLDataset(Dataset):
    def __init__(self, dataset, time_steps, normalized=True):

        if dataset not in dataset_metadata.keys():
            raise ValueError(f"dataset '{dataset}' not found")

        self.dataset = dataset
        self.time_steps = time_steps

        # downloading dataset from endpoint if not buffered
        path = DATASET_DIR + self.dataset + DATASET_EXT
        if not os.path.isfile(path):
            print(f"downloading '{dataset}'...")

            os.makedirs(os.path.dirname(DATASET_DIR), exist_ok=True)
            urllib.request.urlretrieve(dataset_metadata[dataset]["endpoint"], path)

        print(f"loading '{dataset}'...")
        with open(path, "rb") as f:
            loaded = np.load(path)
            self.data = loaded["data"]
            self.constants = loaded["constants"]

        # print(f"data shape {self.data.shape}, constants shape {self.constants.shape}")

        self.num_sims, self.num_steps, self.num_fields, *_ = self.data.shape
        self.num_spatial_dim = (
            self.data.ndim - 3
        )  # subtract sim, step, and field dimension

        print(
            f"successfully loaded { self.dataset } with { self.num_sims } simulations and {self.num_steps} steps each"
        )

        if normalized:
            if (
                "fields" in dataset_metadata[dataset]
                and len(dataset_metadata[dataset]["fields"]) == self.num_fields
            ):

                field_groups = [
                    "".join(g) for _, g in groupby(dataset_metadata[dataset]["fields"])
                ]

                groups_std = []
                idx = 0

                # normalize field groups
                for group in field_groups:
                    group_field = self.data[:, :, idx : (idx + len(group)), ...]

                    # vector norm
                    group_norm = np.linalg.norm(group_field, axis=2, keepdims=True)

                    # axes over which to compute the standard deviation (all axes except fields)
                    axes = (0, 1) + tuple(range(3, 3 + self.num_spatial_dim))
                    group_std = np.std(group_norm, axis=axes, keepdims=True)

                    groups_std.append(
                        np.broadcast_to(
                            group_std, (1, 1, len(group)) + (1,) * self.num_spatial_dim
                        )
                    )
                    idx += len(group)

                self.data /= np.concatenate(groups_std, axis=2)

                # normalize constants
                constants_mean = np.mean(self.constants, axis=0, keepdims=True)
                constants_std = np.std(self.constants, axis=0, keepdims=True)
                self.constants = (self.constants - constants_mean) / (
                    constants_std + 1e-8  # prevent divide-by-zero
                )

    def __len__(self):
        return self.num_sims * (self.num_steps - self.time_steps)

    def __getitem__(self, idx):
        """The data provided has the shape [channels (e.g. velocity_x, velocity_y, density, pressure), x-size, y-size (, z-size)]."""

        # create input-target pairs with interval time_steps from simulation steps
        samples_per_sim = self.num_steps - self.time_steps
        sim_idx = idx // samples_per_sim

        input_idx = idx % samples_per_sim
        target_idx = input_idx + self.time_steps

        input = self.data[sim_idx][input_idx]
        target = self.data[sim_idx][target_idx]

        # additional layers for constants
        input_ext = [
            np.full_like(input[0], self.constants[sim_idx][constant])
            for constant in range(self.constants.shape[1])
        ]

        return (
            torch.tensor(np.concatenate((input, input_ext), axis=0)),
            torch.tensor(target),
        )


# dataset = PBDLDataset("transonic-cylinder-flow-tiny", time_steps=20, normalized=True)
