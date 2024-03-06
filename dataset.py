import os
import json
import urllib.request
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np

from itertools import groupby
import sys

DATASETS_JSON_PATH = "datasets_global.json"
LOCAL_DATASETS_JSON_PATH = "datasets.json"
DATASET_DIR = "datasets/"
DATASET_EXT = ".npz"

# load metadata
try:
    if os.path.isfile(LOCAL_DATASETS_JSON_PATH):
        with open(LOCAL_DATASETS_JSON_PATH, "r") as f:
            local_metadata = json.load(f)
    else:
        local_metadata = {}
except json.JSONDecodeError:
    print(
        f"Error: {LOCAL_DATASETS_JSON_PATH} has the wrong format. Ignoring local metadata."
    )
    local_metadata = {}

with open(DATASETS_JSON_PATH, "r") as f:
    metadata = json.load(f)


class PBDLDataset(Dataset):
    def __init__(
        self,
        dataset,
        time_steps,
        normalize=True,
        simulations=[],
        start_offset=0,
        end_offset=0,
        step_size=1,
    ):

        self.dataset = dataset
        self.time_steps = time_steps
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.step_size = step_size

        if dataset in local_metadata.keys():
            self.fields = local_metadata[dataset]["fields"]
            self.field_desc = local_metadata[dataset]["field_desc"]
            self.constant_desc = local_metadata[dataset]["constant_desc"]

            path = local_metadata[dataset]["path"]

        elif dataset in metadata.keys():
            self.fields = metadata[dataset]["fields"]
            self.field_desc = metadata[dataset]["field_desc"]
            self.constant_desc = metadata[dataset]["constant_desc"]

            # downloading dataset from endpoint if not buffered
            path = DATASET_DIR + self.dataset + DATASET_EXT
            if not os.path.isfile(path):
                print(f"Downloading '{dataset}'...")

                os.makedirs(os.path.dirname(DATASET_DIR), exist_ok=True)
                urllib.request.urlretrieve(metadata[dataset]["endpoint"], path)
        else:
            suggestions = ", ".join(metadata.keys())
            raise ValueError(
                f"Dataset '{dataset}' not found, datasets available are: {suggestions}."
            )

        print(f"Loading '{dataset}'...")
        loaded = np.load(path)
        self.data = loaded["data"]
        self.constants = loaded["constants"]

        if len(self.data.shape) < 5:
            raise ValueError(
                f"Data must have shape (sim, frames, fields, dim, dim [, ...])."
            )

        if len(self.constants.shape) != 2:
            raise ValueError(f"Constant data must have shape (sim, n).")

        if len(self.fields) != self.data.shape[2]:
            raise ValueError(
                f"Inconsistent number of fields between metadata ({len(self.fields) }) and dataset ({ self.data.shape[2]})."
            )

        if simulations:
            self.data = self.data[simulations, ...]

        # print(f"data shape {self.data.shape}, constants shape {self.constants.shape}")

        self.num_sims, self.num_frames, self.num_fields, *_ = self.data.shape
        self.num_spatial_dim = (
            self.data.ndim - 3
        )  # subtract sim, step, and field dimension

        self.samples_per_sim = (
            self.num_frames - self.time_steps - self.start_offset - self.end_offset
        ) // self.step_size

        print(
            f"Successfully loaded { self.dataset } with { self.num_sims } simulations and {self.samples_per_sim} samples each."
        )

        if normalize:
            field_groups = ["".join(g) for _, g in groupby(self.fields)]

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

            self.constants = (self.constants - constants_mean) / constants_std

    def __len__(self):
        return self.num_sims * self.samples_per_sim

    def __getitem__(self, idx):
        """The data provided has the shape (channels [e.g. velocity_x, velocity_y, density, pressure], x-size, y-size [, z-size])."""

        # create input-target pairs with interval time_steps from simulation steps
        sim_idx = idx // self.samples_per_sim

        input_idx = self.start_offset + (idx % self.samples_per_sim)
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

# dataset = PBDLDataset("incompressible-wake-flow-tiny", time_steps=20, normalize=True,step_size=2,end_offset=1)
