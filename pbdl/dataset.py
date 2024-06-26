import os
import json
import urllib.request
import pkg_resources
from itertools import groupby

import numpy as np

config_path = pkg_resources.resource_filename(__name__, "config.json")

# load configuration
try:
    with open(config_path, "r") as f:
        config = json.load(f)
except json.JSONDecodeError:
    raise ValueError("Invalid configuration file.")

# load local dataset index
try:
    if os.path.isfile(config["local_index_path"]):
        with open(config["local_index_path"], "r") as f:
            local_index = json.load(f)
    else:
        local_index = {}
except json.JSONDecodeError:
    print(
        f"Error: {config['local_index_path']} has the wrong format. Ignoring local dataset index."
    )
    local_index = {}

# load global dataset index
global_index_path = pkg_resources.resource_filename(
    __name__, config["global_index_file"]
)
try:
    urllib.request.urlretrieve(config["global_index_url"], global_index_path)
except urllib.error.URLError:
    print("Error: Failed to download global dataset index.")

with open(global_index_path, "r") as f:
    global_index = json.load(f)


class Dataset:
    def __init__(
        self,
        dataset,
        time_steps,
        intermediate_time_steps=False,
        normalize=True,
        solver=False,
        simulations=[],  # if empty, all simulations are loaded
        trim_start=0,
        trim_end=0,
        step_size=1,
    ):

        self.dataset = dataset
        self.time_steps = time_steps
        self.trim_start = trim_start
        self.trim_end = trim_end
        self.step_size = step_size
        self.intermediate_time_steps = intermediate_time_steps
        self.normalize = normalize
        self.solver = solver
        self.partitioned = False

        # TODO get number of simulations from url

        # look for dataset in local index
        if dataset in local_index.keys():
            self.fields = local_index[dataset]["fields"]
            self.field_desc = local_index[dataset]["field_desc"]
            self.constant_desc = local_index[dataset]["constant_desc"]

            path = local_index[dataset]["path"]

            if not path.endswith(config["dataset_ext"]):
                self.partitioned = True

                if "num_part" in local_index[dataset]:
                    self.num_part = local_index[dataset]["num_part"]
                else:
                    raise ValueError(
                        f"No attribute 'num_part' for partitioned dataset '{self.dataset}' in dataset index."
                    )

            if self.partitioned:
                if not simulations:
                    # search for partition files
                    self.part_files = [
                        (path + "/" + f)
                        for f in os.listdir(path)
                        if os.path.isfile(path + "/" + f)
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
                        part_path = (
                            path
                            + "/"
                            + self.dataset
                            + "-"
                            + str(i)
                            + config["dataset_ext"]
                        )
                        if os.path.isfile(part_path):
                            self.part_files.append(part_path)
                        else:
                            raise ValueError(
                                f"Simulation {i} not found in local dataset '{self.dataset}'."
                            )

        # if not found in local index than look in global index
        elif dataset in global_index.keys():
            self.fields = global_index[dataset]["fields"]
            self.field_desc = global_index[dataset]["field_desc"]
            self.constant_desc = global_index[dataset]["constant_desc"]

            url = global_index[dataset]["url"]
            if not url.endswith(config["dataset_ext"]):
                self.partitioned = True

                if "num_part" in global_index[dataset]:
                    self.num_part = global_index[dataset]["num_part"]
                else:
                    raise ValueError(
                        f"No 'num_part' attribute for partitioned dataset '{self.dataset}' in dataset index."
                    )

            if self.partitioned:

                os.makedirs(
                    os.path.dirname(config["dataset_dir"] + self.dataset + "/"),
                    exist_ok=True,
                )

                self.part_files = []

                # check if simulations are cached
                for i in range(self.num_part):
                    part_file = self.dataset + "-" + str(i) + config["dataset_ext"]
                    part_path = config["dataset_dir"] + self.dataset + "/" + part_file
                    self.part_files.append(part_path)

                    if not os.path.isfile(part_path):
                        print(f"Simulation {i} not cached. Downloading...")

                        part_url = url + "/" + part_file

                        try:
                            urllib.request.urlretrieve(part_url, part_path)
                        except urllib.error.URLError:
                            raise FileNotFoundError(
                                f"Could not download {part_file} from server."
                            )

                # try to download normalization data
                try:
                    urllib.request.urlretrieve(
                        url + "/" + config["norm_data_file"] + config["dataset_ext"],
                        config["dataset_dir"] + self.dataset + config["norm_data_file"],
                    )
                except urllib.error.URLError:
                    print("No precomputed normalization data found on server.")

            else:

                path = config["dataset_dir"] + self.dataset + config["dataset_ext"]

                if not os.path.isfile(
                    config["dataset_dir"] + self.dataset + config["dataset_ext"]
                ):
                    print(f"Single-file dataset '{dataset}' not cached. Downloading...")
                    os.makedirs(os.path.dirname(config["dataset_dir"]), exist_ok=True)
                    urllib.request.urlretrieve(url, path)

        # neither found in local index nor in global index
        else:
            suggestions = ", ".join((global_index | local_index).keys())
            raise ValueError(
                f"Dataset '{dataset}' not found, datasets available are: {suggestions}."
            )

        print(f"Loading '{dataset}'...")

        if self.partitioned:
            # load only the first partition
            self.__load_partition__(0, assert_shape=False)

            # partition shape expected for all other partitions
            self.part_data_shape = self.data.shape
            self.part_const_shape = self.const.shape

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
            self.num_frames - self.time_steps - self.trim_start - self.trim_end
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

    def __load_partition__(self, partition, assert_shape=True):

        self.part_loaded = partition
        loaded = np.load(self.part_files[partition])
        self.data = loaded["data"]
        self.const = loaded["constants"]

        # valide shape
        if assert_shape:
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
            norm_data_file_path = (
                config["dataset_dir"] + self.dataset + "/" + config["norm_data_file"]
            )

            # check if normalization data is cached
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

                groups_std = []
                groups_std_slim = [0] * len(field_groups)
                const_stacked = []
                idx = 0

                # sequential loading of partitions
                for partition in range(self.num_part):
                    self.__load_partition__(partition)

                    # calculate normalization constants by field
                    for group_idx, group in enumerate(field_groups):
                        group_field = self.data[:, idx : (idx + len(group)), ...]

                        # vector norm
                        group_norm = np.linalg.norm(group_field, axis=1, keepdims=True)

                        # axes over which to compute the standard deviation (all axes except fields)
                        axes = (0, 1) + tuple(range(2, 2 + self.num_spatial_dim))

                        groups_std_slim[group_idx] += np.std(
                            group_norm, axis=axes, keepdims=True
                        )

                        idx += len(group)

                    const_stacked.append(self.const)

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
                    config["dataset_dir"]
                    + self.dataset
                    + "/"
                    + config["norm_data_file"],
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

        # solver needs non-normalized constants
        if self.solver:
            self.nnorm_const = self.const

        if abs(self.const_std) < 10e-10:
            self.const = np.zeros_like(self.const)
        else:
            self.const = (self.const - self.const_mean) / self.const_std

    def __len__(self):
        return self.num_sims * self.samples_per_sim

    def __getitem__(self, idx):
        """
        The data provided has the shape (channels, spatial dims...).

        Returns:
            numpy.ndarray: Input data (without constants)
            tuple: Constants
            numpy.ndarray: Target data
            tuple: Non-normalized constants (only if solver flag is set)
        """

        # create input-target pairs with interval time_steps from simulation steps
        sim_idx = idx // self.samples_per_sim

        input_idx = self.trim_start + (idx % self.samples_per_sim) * self.step_size
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

        if self.intermediate_time_steps:
            target = sim[input_idx + 1 : target_idx + 1]
        else:
            target = sim[target_idx]

        if self.solver:
            return (
                input,
                tuple(self.const if self.partitioned else self.const[sim_idx]),
                target,
                tuple(
                    self.nnorm_const if self.partitioned else self.nnorm_const[sim_idx]
                ),
            )
        else:
            return (
                input,
                tuple(self.const if self.partitioned else self.const[sim_idx]),
                target,
            )
