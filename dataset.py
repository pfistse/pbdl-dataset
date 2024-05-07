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

GLOBAL_DATASET_INDEX_URL = "https://syncandshare.lrz.de/dl/fiYHVktL6S9mZM6duNnSCP/datasets_global.json"  # TODO

GLOBAL_DATASET_INDEX_PATH = "datasets_global.json"
LOCAL_DATASET_INDEX_PATH = "datasets.json"
DATASET_DIR = "datasets/"
DATASET_EXT = ".npz"

# phi flow
PHIFLOW_SPATIAL_DIM = ["x", "y", "z"]

# load local dataset index
try:
    if os.path.isfile(LOCAL_DATASET_INDEX_PATH):
        with open(LOCAL_DATASET_INDEX_PATH, "r") as f:
            local_metadata = json.load(f)
    else:
        local_metadata = {}
except json.JSONDecodeError:
    print(
        f"Error: {LOCAL_DATASET_INDEX_PATH} has the wrong format. Ignoring local dataset index."
    )
    local_metadata = {}

# load global dataset index
try:
    urllib.request.urlretrieve(
        GLOBAL_DATASET_INDEX_URL, GLOBAL_DATASET_INDEX_PATH 
    )  # update dataset index
except urllib.error.URLError:
    print("Failed to fetch global dataset index.")

with open(GLOBAL_DATASET_INDEX_PATH, "r") as f:
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
        self.normalize = normalize
        self.partitioned = False

        # TODO get number of simulations from endpoint

        # local dataset
        if dataset in local_metadata.keys():
            self.fields = local_metadata[dataset]["fields"]
            self.field_desc = local_metadata[dataset]["field_desc"]
            self.constant_desc = local_metadata[dataset]["constant_desc"]

            path = local_metadata[dataset]["path"]

            if  not path.endswith(DATASET_EXT):
                self.partitioned = True

                if "num_part" in local_metadata[dataset]:
                    self.num_part = local_metadata[dataset]["num_part"]
                else:
                    raise ValueError(
                        f"No 'num_part' attribute for partitioned dataset '{self.dataset}' in dataset index."
                    )

            if self.partitioned:
                if not simulations:
                    # search for partition files
                    self.part_files = [
                        (path + f)
                        for f in os.listdir(path)
                        if os.path.isfile(path + f)
                        and f.startswith(self.dataset + "-")
                    ]

                    if not self.part_files:
                        raise FileNotFoundError(
                            f"No simulation files found for dataset '{self.dataset}'."
                        )
                else:
                    # find corresponding partition files for specified simulations
                    self.part_files = []

                    for i in simulations:
                        part_file = (
                            path + self.dataset + "-" + str(i) + DATASET_EXT
                        )
                        if os.path.isfile(part_file):
                            self.part_files.append(part_file)
                        else:
                            raise ValueError(
                                f"Simulation {i} not found in local dataset '{self.dataset}'."
                            )

        # global dataset
        elif dataset in metadata.keys():
            self.fields = metadata[dataset]["fields"]
            self.field_desc = metadata[dataset]["field_desc"]
            self.constant_desc = metadata[dataset]["constant_desc"]

            endpoint = metadata[dataset]["endpoint"]
            if  not endpoint.endswith(DATASET_EXT):
                self.partitioned = True

                if "num_part" in metadata[dataset]:
                    self.num_part = metadata[dataset]["num_part"]
                else:
                    raise ValueError(
                        f"No 'num_part' attribute for partitioned dataset '{self.dataset}' in dataset index."
                    )

            if self.partitioned:

                os.makedirs(
                    os.path.dirname(DATASET_DIR + self.dataset + "/"), exist_ok=True
                )

                self.part_files = []

                # if not simulations:
                #     raise ValueError(
                #         "For partitioned global datasets, an explicit specification of the simulations is required."
                #     )

                # check if all necessary simulations are cached
                for i in range(self.num_part):
                    part_file = (
                        DATASET_DIR
                        + self.dataset
                        + "/"
                        + self.dataset
                        + "-"
                        + str(i)
                        + DATASET_EXT
                    )
                    self.part_files.append(part_file)

                    if not os.path.isfile(part_file):
                        print(f"Simulation {i} not cached. Downloading...")

                        sim_file_endpoint = (
                            endpoint
                            + self.dataset
                            + "-"
                            + str(i)
                            + DATASET_EXT
                        )

                        urllib.request.urlretrieve(sim_file_endpoint, part_file)

                # try to download normalization data
                try:
                    urllib.request.urlretrieve(
                        endpoint + "norm_data" + DATASET_EXT,
                        DATASET_DIR + self.dataset + "norm_data" + DATASET_EXT,
                    )
                except urllib.error.URLError:
                    print("No precomputed normalization data found on server.")

                path = DATASET_DIR + self.dataset

            else:

                path = DATASET_DIR + self.dataset + DATASET_EXT

                if not os.path.isfile(DATASET_DIR + self.dataset + DATASET_EXT):
                    print(f"Single-file dataset '{dataset}' not cached. Downloading...")

                    os.makedirs(os.path.dirname(DATASET_DIR), exist_ok=True)

                    urllib.request.urlretrieve(endpoint, path)

        else:
            suggestions = ", ".join((metadata | local_metadata).keys())
            raise ValueError(
                f"Dataset '{dataset}' not found, datasets available are: {suggestions}."
            )

        print(f"Loading '{dataset}'...")

        if self.partitioned:
            # load only the first partition
            self.__load_partition__(0, match_first_part=False)
            self.part_data_shape = self.data.shape
            self.part_const_shape = self.const.shape

            # do validation checks lazy (done in __load_partition__)
        else:
            loaded_sims = np.load(path)
            self.data = loaded_sims["data"]
            self.const = loaded_sims["constants"]

            if simulations:
                self.data = self.data[simulations, ...]

            # do basic validation checks
            if len(self.data.shape) < 4:
                raise ValueError(
                    "Data must have shape (sim, frames, fields, dim [, ...])."
                )

            if len(self.const.shape) != 2:
                raise ValueError(f"Constant data must have shape (sim, n).")

            if len(self.fields) != self.data.shape[2]:
                raise ValueError(
                    f"Inconsistent number of fields between metadata ({len(self.fields) }) and dataset ({ self.data.shape[2]})."
                )

        shape = (
            (len(self.part_files),) + self.data.shape
            if self.partitioned
            else self.data.shape
        )
        self.num_sims, self.num_frames, self.num_fields, *_ = shape
        self.num_spatial_dim = len(shape) - 3  # subtract sim, step, and field dimension

        self.num_const = (
            self.const.shape[0] if self.partitioned else self.const.shape[1]
        )

        self.samples_per_sim = (
            self.num_frames - self.time_steps - self.start_offset - self.end_offset
        ) // self.step_size

        print(
            f"Successfully loaded { self.dataset } with { self.num_sims } simulations and {self.samples_per_sim} samples each."
        )

        # normalize data and constants
        if self.normalize:
            self.__prepare_norm_data__()

            # partitioned data is normalized lazy
            if not self.partitioned:
                self.__normalize__()

    def __load_partition__(self, partition, match_first_part=True):
        #print(f"DEBUG: loading partition {partition}")

        self.part_loaded = partition
        loaded = np.load(self.part_files[partition])
        self.data = loaded["data"]
        self.const = loaded["constants"]

        # valide shape
        if match_first_part:
            if self.data.shape != self.part_data_shape:
                raise ValueError(
                    f"The data shape of all partitions must be consistent. At least one does not match {self.part_data_shape}."
                )

            if self.const.shape != self.part_const_shape:
                raise ValueError(
                    f"The constant shape of all partitions must be consistent. At least for one partition the data does not match {self.part_const_shape}."
                )
        else:
            if len(self.data.shape) < 3:
                raise ValueError(
                    "Partition data must have shape (frames, fields, dim [, ...])."
                )

            if len(self.const.shape) != 1:
                raise ValueError(f"Partition constant data must have shape (n).")

            if len(self.fields) != self.data.shape[1]:
                raise ValueError(
                    f"Inconsistent number of fields between metadata ({len(self.fields) }) and dataset ({ self.data.shape[1]})."
                )

    def __prepare_norm_data__(self):
        field_groups = ["".join(g) for _, g in groupby(self.fields)]

        if self.partitioned:
            norm_data_file_path = DATASET_DIR + self.dataset + "/norm_data" + DATASET_EXT

            # normalization data cached
            if os.path.isfile(norm_data_file_path):

                loaded_norm_data = np.load(norm_data_file_path)
                self.data_std = loaded_norm_data["data_std"]
                self.const_mean = loaded_norm_data["const_mean"]
                self.const_std = loaded_norm_data["const_std"]

                if (
                    self.data_std.shape[1 if self.partitioned else 2]
                    != self.data.shape[1 if self.partitioned else 2]
                ):
                    raise ValueError(
                        "Inconsistent number of fields between normalization data and simulation data."
                    )

                if (
                    self.const_mean.shape[0 if self.partitioned else 1]
                    != self.const.shape[0 if self.partitioned else 1]
                ):
                    raise ValueError(
                        "Mean data of constants does not match shape of constants."
                    )

                if (
                    self.const_std.shape[0 if self.partitioned else 1]
                    != self.const.shape[0 if self.partitioned else 1]
                ):
                    raise ValueError(
                        "Std data of constants does not match shape of constants."
                    )

            # calculate normalization data
            else:
                print("No precomputed normalization data found. Calculating data...")
                # sequential loading of partitions for calculation

                groups_std = []
                groups_std_slim = [0] * len(field_groups)
                const_stacked = []
                idx = 0

                for part_file in self.part_files:
                    partition = np.load(part_file)
                    part_data = partition["data"]
                    part_const = partition["constants"]

                    # TODO do validation checks

                    # calculate normalization constants by field
                    for group_idx, group in enumerate(field_groups):
                        group_field = part_data[:, idx : (idx + len(group)), ...]

                        # vector norm
                        group_norm = np.linalg.norm(group_field, axis=1, keepdims=True)

                        # axes over which to compute the standard deviation (all axes except fields)
                        axes = (0, 1) + tuple(range(2, 2 + self.num_spatial_dim))

                        groups_std_slim[group_idx] += np.std(
                            group_norm, axis=axes, keepdims=True
                        )

                        idx += len(group)

                    const_stacked.append(part_const)

                # TODO overall std is calculated by averaging the std of all sims, efficient but mathematically not correct
                for group_idx, group in enumerate(field_groups):
                    groups_std.append(
                        np.broadcast_to(
                            groups_std_slim[group_idx] / self.num_sims,
                            (1, len(group)) + (1,) * self.num_spatial_dim,
                        )
                    )
                self.data_std = np.concatenate(groups_std, axis=1)
                self.const_mean = np.mean(const_stacked, axis=0, keepdims=False)
                self.const_std = np.std(const_stacked, axis=0, keepdims=False)

                # cache norm data
                np.savez(
                    DATASET_DIR + self.dataset + "/norm_data" + DATASET_EXT,
                    **{
                        "data_std": self.data_std,
                        "const_mean": self.const_mean,
                        "const_std": self.const_std,
                    },
                )

        else:
            groups_std = []
            idx = 0

            print("Calculating normalization data...")

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
                        group_std,
                        (1, 1, len(group)) + (1,) * self.num_spatial_dim,
                    )
                )
                idx += len(group)

            self.data_std = np.concatenate(groups_std, axis=2)
            self.const_mean = np.mean(self.const, axis=0, keepdims=True)
            self.const_std = np.std(self.const, axis=0, keepdims=True)

    def __normalize__(self):
        self.data /= self.data_std

        if abs(self.const_std) < 10e-10:
            self.const = np.zeros_like(self.const)
        else:
            self.const = (self.const - self.const_mean) / self.const_std

    def __len__(self):
        return self.num_sims * self.samples_per_sim

    def __getitem__(self, idx):
        """The data provided has the shape (channels [e.g. velocity_x, velocity_y, density, pressure], x-size, y-size [, z-size])."""

        # create input-target pairs with interval time_steps from simulation steps
        sim_idx = idx // self.samples_per_sim

        input_idx = self.start_offset + (idx % self.samples_per_sim) * self.step_size
        target_idx = input_idx + self.time_steps

        if self.partitioned:
            if self.part_loaded != sim_idx:
                self.__load_partition__(sim_idx)

                if self.normalize:
                    self.__normalize__()

            sim = self.data
        else:
            sim = self.data[sim_idx]

        input = sim[input_idx]

        if self.time_stepping:
            target = sim[input_idx + 1 : target_idx + 1]
        else:
            target = sim[target_idx]
        return (
            input,
            tuple(self.const if self.partitioned else self.const[sim_idx]),
            target,
        )

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
