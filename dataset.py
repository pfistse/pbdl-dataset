import os
import json
import urllib.request
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset

from phi.torch.flow import *

from itertools import groupby
import sys

DATASETS_JSON_PATH = "datasets_global.json"
LOCAL_DATASETS_JSON_PATH = "datasets.json"
DATASET_DIR = "datasets/"
DATASET_EXT = ".npz"

# phi flow
PHIFLOW_SPATIAL_DIM = ["x", "y", "z"]

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
        simulations=[],  # if empty, all simulations are loaded
        start_offset=0,
        end_offset=0,
        step_size=1,
        time_stepping=False,
    ):

        self.dataset = dataset
        self.time_steps = time_steps
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.step_size = step_size
        self.time_stepping = time_stepping

        # TODO retrieve number of simulations from endpoint

        # local dataset
        if dataset in local_metadata.keys():
            self.fields = local_metadata[dataset]["fields"]
            self.field_desc = local_metadata[dataset]["field_desc"]
            self.constant_desc = local_metadata[dataset]["constant_desc"]

            path = local_metadata[dataset]["path"]
            partitioned = not path.endswith(DATASET_EXT)

            if partitioned:
                if not simulations:
                    sim_files = [
                        (path + "/" + f)
                        for f in os.listdir(path)
                        if os.path.isfile(path + "/" + f)
                        and f.startswith(self.dataset + "-")
                    ]

                    if not sim_files:
                        raise FileNotFoundError(
                            f"No simulation files found for dataset '{self.dataset}'."
                        )
                else:
                    sim_files = []
                    for i in simulations:
                        sim_file = (
                            path + "/" + self.dataset + "-" + str(i) + DATASET_EXT
                        )
                        if os.path.isfile(sim_file):
                            sim_files.append(sim_file)
                        else:
                            raise ValueError(
                                f"Simulation {i} not found in local dataset '{self.dataset}'."
                            )

        # global dataset
        elif dataset in metadata.keys():
            self.fields = metadata[dataset]["fields"]
            self.field_desc = metadata[dataset]["field_desc"]
            self.constant_desc = metadata[dataset]["constant_desc"]

            partitioned = not metadata[dataset]["endpoint"].endswith(DATASET_EXT)

            if partitioned:

                os.makedirs(
                    os.path.dirname(DATASET_DIR + self.dataset + "/"), exist_ok=True
                )

                sim_files = []

                if not simulations:
                    raise ValueError(
                        "For partitioned global datasets, an explicit specification of the simulations is required."  # TODO read directory
                    )

                # check if all necessary simulations are cached
                for i in simulations:
                    sim_file = (
                        DATASET_DIR
                        + self.dataset
                        + "/"
                        + self.dataset
                        + "-"
                        + str(i)
                        + DATASET_EXT
                    )
                    sim_files.append(sim_file)

                    if not os.path.isfile(sim_file):
                        print(f"Simulation {i} not cached. Downloading...")

                        sim_file_endpoint = (
                            metadata[dataset]["endpoint"]
                            + "/"
                            + self.dataset
                            + "-"
                            + str(i)
                            + DATASET_EXT
                        )

                        urllib.request.urlretrieve(sim_file_endpoint, sim_file)

                # try to download normalization data
                try:
                    urllib.request.urlretrieve(
                        metadata[dataset]["endpoint"] + "/norm_data" + DATASET_EXT,
                        DATASET_DIR + self.dataset + "/norm_data" + DATASET_EXT,
                    )
                except urllib.error.URLError:
                    print("No precomputed normalization data found.")

                path = DATASET_DIR + self.dataset

            elif not partitioned:

                path = DATASET_DIR + self.dataset + DATASET_EXT

                if not os.path.isfile(DATASET_DIR + self.dataset + DATASET_EXT):
                    print(f"Single-file dataset '{dataset}' not cached. Downloading...")

                    os.makedirs(os.path.dirname(DATASET_DIR), exist_ok=True)

                    urllib.request.urlretrieve(metadata[dataset]["endpoint"], path)

        else:
            suggestions = ", ".join((metadata | local_metadata).keys())
            raise ValueError(
                f"Dataset '{dataset}' not found, datasets available are: {suggestions}."
            )

        print(f"Loading '{dataset}'...")

        if partitioned:

            loaded_sims = [np.load(sim_file) for sim_file in sim_files]

            try:
                self.data = np.stack([sim["data"] for sim in loaded_sims], axis=0)
            except Exception as e:
                print(e)
                raise Exception(
                    "Data in all simulation files must have a homogeneous shape."
                )

            try:
                self.constants = np.stack(
                    [sim["constants"] for sim in loaded_sims], axis=0
                )
            except Exception as e:
                raise Exception(
                    "Constant data in all simulation files must have a homogeneous shape."
                )

        else:
            loaded_sims = np.load(path)
            self.data = loaded_sims["data"]
            self.constants = loaded_sims["constants"]

            if simulations:
                self.data = self.data[simulations, ...]

        if len(self.data.shape) < 4:
            raise ValueError("Data must have shape (sim, frames, fields, dim [, ...]).")

        if len(self.constants.shape) != 2:
            raise ValueError(f"Constant data must have shape (sim, n).")

        if len(self.fields) != self.data.shape[2]:
            print(self.data.shape)
            raise ValueError(
                f"Inconsistent number of fields between metadata ({len(self.fields) }) and dataset ({ self.data.shape[2]})."
            )

        self.num_sims, self.num_frames, self.num_fields, *_ = self.data.shape
        self.num_spatial_dim = (
            self.data.ndim - 3
        )  # subtract sim, step, and field dimension

        self.num_const = self.constants.shape[1]

        self.samples_per_sim = (
            self.num_frames - self.time_steps - self.start_offset - self.end_offset
        ) // self.step_size

        print(
            f"Successfully loaded { self.dataset } with { self.num_sims } simulations and {self.samples_per_sim} samples each."
        )

        if normalize:

            norm_data_file_path = (
                DATASET_DIR + self.dataset + "/norm_data" + DATASET_EXT
            )
            field_groups = ["".join(g) for _, g in groupby(self.fields)]

            if os.path.isfile(norm_data_file_path):

                loaded_norm_data = np.load(norm_data_file_path)
                groups_std = loaded_norm_data["groups_std"]
                const_mean = loaded_norm_data["const_mean"]
                const_std = loaded_norm_data["const_std"]

                if groups_std.shape[2] != self.data.shape[2]:
                    raise ValueError(
                        "Inconsistent number of fields between normalization data and simulation data."
                    )

                if const_mean.shape[1] != self.constants.shape[1]:
                    raise ValueError(
                        "Mean data of constants does not match shape of constants."
                    )

                if const_std.shape[1] != self.constants.shape[1]:
                    raise ValueError(
                        "Std data of constants does not match shape of constants."
                    )

            else:

                print("No precalculated normalization data found. Calculating data...")

                groups_std = []
                idx = 0

                # calculate normalization constants by field
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

                groups_std = np.concatenate(groups_std, axis=2)
                const_mean = np.mean(self.constants, axis=0, keepdims=True)
                const_std = np.std(self.constants, axis=0, keepdims=True)

            # np.savez(
            #     DATASET_DIR + self.dataset + "/norm_data" + DATASET_EXT,
            #     **{
            #         "groups_std": groups_std,
            #         "const_mean": const_mean,
            #         "const_std": const_std,
            #     },
            # )

            # normalization
            self.data /= groups_std

            if abs(const_std) < 10e-10:
                self.constants = np.zeros_like(self.constants)
            else:
                self.constants = (self.constants - const_mean) / const_std

    def __len__(self):
        return self.num_sims * self.samples_per_sim

    def __getitem__(self, idx):
        """The data provided has the shape (channels [e.g. velocity_x, velocity_y, density, pressure], x-size, y-size [, z-size])."""

        # create input-target pairs with interval time_steps from simulation steps
        sim_idx = idx // self.samples_per_sim

        input_idx = self.start_offset + (idx % self.samples_per_sim) * self.step_size
        target_idx = input_idx + self.time_steps

        input = self.data[sim_idx][input_idx]

        if self.time_stepping:
            target = self.data[sim_idx][input_idx + 1 : target_idx + 1]
        else:
            target = self.data[sim_idx][target_idx]

        return (input, tuple(self.constants[sim_idx]), target)

    def to_phiflow(self, data):
        """Convert network input to solver input. Constant layers are ignored."""
        spatial_dim = ",".join(PHIFLOW_SPATIAL_DIM[0 : self.num_spatial_dim])
        return tensor(
            data[:, 0:1, ...],
            batch("b"),
            instance("time"),
            spatial(spatial_dim),
        )

    def from_phiflow(self, data):
        """Convert solver output to a network output-like format."""
        spatial_dim = ",".join(PHIFLOW_SPATIAL_DIM[0 : self.num_spatial_dim])
        return data.native(["b", "time", spatial_dim])

    def cat_constants(self, data, like):
        """Concatenate constants from `like` to `data`. Useful for mapping network outputs to network inputs of the next iteration."""
        return torch.cat(
            [data, like[:, self.num_fields : self.num_fields + self.num_const, ...]],
            axis=1,
        )  # dim 0 is batch dimension
