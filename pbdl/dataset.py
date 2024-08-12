import os
import io
import sys
import json
import urllib.request
import pkg_resources
import h5py
from itertools import groupby

import numpy as np
from pbdl.colors import colors

config_path = pkg_resources.resource_filename(__name__, "config.json")

# load configuration
try:
    with open(config_path, "r") as f:
        config = json.load(f)
except json.JSONDecodeError:
    raise ValueError("Invalid configuration file.")


class Dataset:
    def __init__(
        self,
        dset_name,
        time_steps,
        normalize=True,
        sel_sims=None,  # if None, all simulations are loaded
        trim_start=0,
        trim_end=0,
        step_size=1,
        **kwargs,
    ):

        self.time_steps = time_steps
        self.trim_start = trim_start
        self.trim_end = trim_end
        self.step_size = step_size
        # self.intermediate_time_steps = intermediate_time_steps
        self.normalize = normalize
        # self.solver = solver
        self.sel_sims = sel_sims

        config.update(kwargs)
        self.__load_indices__()

        # TODO get number of simulations from url

        if dset_name in self.local_index.keys():
            self.__load_dataset__(
                dset_name,
                dset_file=os.path.join(
                    config["local_datasets_dir"], self.local_index[dset_name]["path"]
                ),
            )
        elif dset_name in self.global_index.keys():
            self.__download_dataset__(dset_name, sel_sims)
            self.__load_dataset__(dset_name)
        else:
            suggestions = ", ".join((self.global_index | self.local_index).keys())
            print(
                colors.FAIL
                + f"Dataset '{dset_name}' not found, datasets available are: {suggestions}."
                + colors.ENDC
            )
            sys.exit(0)

        print(
            colors.OKCYAN
            + colors.BOLD
            + f"Successfully loaded { self.dset_name } with { self.num_sims } simulations "
            + (f"({len(self.sel_sims)} selected) " if self.sel_sims else "")
            + f"and {self.samples_per_sim} samples each."
            + colors.ENDC
        )

        if self.normalize:
            self.__prepare_norm_data__()

    def __load_indices__(self):
        # load local dataset index
        try:
            if os.path.isfile(config["local_index_path"]):
                with open(config["local_index_path"], "r") as f:
                    self.local_index = json.load(f)
            else:
                self.local_index = {}
        except json.JSONDecodeError:
            print(
                colors.WARNING
                + f"Warning: {config['local_index_path']} has the wrong format. Ignoring local dataset index."
                + colors.ENDC
            )
            self.local_index = {}

        # load global dataset index
        global_index_path = pkg_resources.resource_filename(
            __name__, config["global_index_file"]
        )
        try:
            urllib.request.urlretrieve(config["global_index_url"], global_index_path)
        except urllib.error.URLError:
            print(
                colors.WARNING
                + "Warning: Failed to download global dataset index. Check your internet connection."
                + colors.ENDC
            )

        with open(global_index_path, "r") as f:
            self.global_index = json.load(f)

    def __download_dataset__(self, dset_name, sel_sims=None):
        url = self.global_index[dset_name]["url"]

        # partitioned dataset
        if "num_part" in self.global_index[dset_name]:

            num_part = self.global_index[dset_name]["num_part"]

            dset_file = os.path.join(config["dataset_dir"], dset_name + config["dataset_ext"])

            if not sel_sims:
                sel_sims = range(num_part)

            with h5py.File(dset_file, "a") as f:

                added_sim = False
                # check if simulations are cached
                for i, s in enumerate(sel_sims):
                    print_download_progress(
                        i,
                        1,
                        len(sel_sims),
                        message=f"downloading sim {s}",
                    )

                    if "sims/sim" + str(s) not in f:
                        added_sim = True

                        sim_url = url + "/sim" + str(s) + config["dataset_ext"]

                        with urllib.request.urlopen(sim_url) as response:
                            with h5py.File(io.BytesIO(response.read()), "r") as f_sim:

                                if len(f_sim) != 1:
                                    raise ValueError(
                                        f"A partition file must contain exactly one simulation."
                                    )

                                sim = f.create_dataset(
                                    "sims/sim" + str(s), data=f_sim["sims/sim0"]
                                )
                                sim.attrs["const"] = f_sim["sims/sim0"].attrs["const"]

                print_download_progress(
                    len(sel_sims), 1, len(sel_sims), message="download completed"
                )

                # normalization data does not incorporate all sims
                if added_sim:
                    f.pop("norm_data_std", None)
                    f.pop("norm_const_mean", None)
                    f.pop("norm_const_std", None)

        # single-file dataset
        else:

            dset_file = os.path.join(config["dataset_dir"], dset_name + config["dataset_ext"])

            if not os.path.isfile(dset_file):
                # print(f"Dataset '{dset_name}' not cached. Downloading...")
                os.makedirs(os.path.dirname(config["dataset_dir"]), exist_ok=True)
                urllib.request.urlretrieve(
                    url, dset_file, reporthook=print_download_progress
                )

    def __load_dataset__(self, dset_name, dset_file=None):
        self.dset_name = dset_name

        if dset_name in self.local_index.keys():
            self.fields = self.local_index[dset_name]["fields"]
            self.field_desc = self.local_index[dset_name]["field_desc"]
            self.const_desc = self.local_index[dset_name]["const_desc"]
        elif dset_name in self.global_index.keys():
            self.fields = self.global_index[dset_name]["fields"]
            self.field_desc = self.global_index[dset_name]["field_desc"]
            self.const_desc = self.global_index[dset_name]["const_desc"]

        if not dset_file:
            dset_file = os.path.join(config["dataset_dir"],  dset_name + config["dataset_ext"])

        self.dset = h5py.File(dset_file, "r")

        first_sim_name = next(iter(self.dset["sims/"]))
        first_sim = self.dset["sims/" + first_sim_name]
        self.sim_shape = first_sim.shape
        self.const_shape = first_sim.attrs["const"].shape

        # basic validation checks on shape
        if len(self.sim_shape) < 3:
            raise ValueError(
                "Simulations data must have shape (frames, fields, spatial dim [...])."
            )

        if len(self.fields) != self.sim_shape[1]:
            raise ValueError(
                f"Inconsistent number of fields between metadata ({len(self.fields) }) and simulations ({ self.sim_shape[1]})."
            )

        if len(self.const_shape) != 1:
            raise ValueError(
                f"The shape of the constant array must have exactly one dimension."
            )

        # shape must be consistent through all sims
        for sim in self.dset["sims/"]:
            if (self.dset["sims/" + sim].shape) != self.sim_shape:
                raise ValueError(
                    f"The shape of all simulations must be consistent ({sim}.shape does not match {first_sim_name}.shape)."
                )
            if (self.dset["sims/" + sim].attrs["const"].shape) != self.const_shape:
                raise ValueError(
                    f"The shape of all constants must be consistent ({sim}.const.shape does not match {first_sim_name}.const.shape)."
                )

        # set attributes
        self.num_sims = len(self.dset["sims/"])
        self.num_frames, self.num_fields, *_ = self.sim_shape
        self.num_spatial_dim = (
            len(self.sim_shape) - 2
        )  # subtract 2 for frame and field dimension
        self.num_const = len(self.const_shape)

        self.samples_per_sim = (
            self.num_frames - self.time_steps - self.trim_start - self.trim_end
        ) // self.step_size

    def __prepare_norm_data__(self):

        if not all(
            key in self.dset
            for key in ["norm_data_std", "norm_const_mean", "norm_const_std"]
        ):
            # calculate norm
            print(
                "No precomputed normalization data found (or not complete). Calculating data..."
            )

            field_groups = ["".join(g) for _, g in groupby(self.fields)]

            groups_std = []
            groups_std_slim = [0] * len(field_groups)
            const_stacked = []
            idx = 0

            # sequential loading of simulations
            for sim in self.dset["sims/"]:

                # calculate normalization constants by field
                for group_idx, group in enumerate(field_groups):
                    group_field = self.dset["sims/" + sim][
                        :, idx : (idx + len(group)), ...
                    ]

                    # vector norm
                    group_norm = np.linalg.norm(group_field, axis=1, keepdims=True)

                    # axes over which to compute the standard deviation (all axes except fields)
                    axes = (0, 1) + tuple(range(2, 2 + self.num_spatial_dim))

                    groups_std_slim[group_idx] += np.std(
                        group_norm, axis=axes, keepdims=True
                    )[
                        0
                    ]  # drop frame dimension

                    idx += len(group)

                const_stacked.append(self.dset["sims/" + sim].attrs["const"])

            # TODO overall std is calculated by averaging the std of all sims, efficient but mathematically not correct
            for group_idx, group in enumerate(field_groups):
                groups_std.append(
                    np.broadcast_to(
                        groups_std_slim[group_idx] / self.num_sims,
                        (len(group),) + (1,) * self.num_spatial_dim,
                    )
                )

            dset_file = self.dset.filename
            self.dset.close()

            with h5py.File(dset_file, "r+") as f:
                f["norm_data_std"] = np.concatenate(groups_std, axis=0)
                f["norm_const_mean"] = np.mean(const_stacked, axis=0, keepdims=False)
                f["norm_const_std"] = np.std(const_stacked, axis=0, keepdims=False)

            self.dset = h5py.File(dset_file, "r")

        # load normalization data
        self.data_std = self.dset["norm_data_std"][
            ()
        ]  # [()] reads the entire array TODO
        self.const_mean = self.dset["norm_const_mean"][()]
        self.const_std = self.dset["norm_const_std"][()]

        # do basic checks on shape
        if self.data_std.shape[0] != self.sim_shape[1]:
            raise ValueError(
                "Inconsistent number of fields between normalization data and simulation data."
            )

        if self.const_mean.shape[0] != self.const_shape[0]:
            raise ValueError(
                "Mean data of constants does not match shape of constants."
            )

        if self.const_std.shape[0] != self.const_shape[0]:
            raise ValueError("Std data of constants does not match shape of constants.")

    def __len__(self):
        if self.sel_sims:
            return len(self.sel_sims) * self.samples_per_sim
        else:
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
        if idx >= len(self):
            raise IndexError

        # create input-target pairs with interval time_steps from simulation steps
        if self.sel_sims:
            sim_idx = self.sel_sims[idx // self.samples_per_sim]
        else:
            sim_idx = idx // self.samples_per_sim

        input_frame_idx = (
            self.trim_start + (idx % self.samples_per_sim) * self.step_size
        )
        target_frame_idx = input_frame_idx + self.time_steps

        sim = self.dset["sims/sim" + str(sim_idx)]
        const = sim.attrs["const"]

        input = sim[input_frame_idx]
        if self.normalize:
            input /= self.data_std

        target = sim[target_frame_idx]

        # normalize
        if self.normalize:
            input /= self.data_std
            target /= self.data_std

            if (abs(self.const_std) < 10e-10).any():  # TODO
                const = np.zeros_like(const)
            else:
                const = (const - self.const_mean) / self.const_std

        return (
                input,
                tuple(const), # required by loader
                target,
            )

def print_download_progress(count, block_size, total_size, message=None):
    progress = count * block_size
    percent = int(progress * 100 / total_size)
    bar_length = 50
    bar = (
        "━" * int(percent / 2)
        + colors.DARKGREY
        + "━" * (bar_length - int(percent / 2))
        + colors.OKBLUE
    )

    def format_size(size):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"

    downloaded_str = format_size(progress)
    total_str = format_size(total_size)

    sys.stdout.write(
        colors.OKBLUE
        + "\r\033[K"
        + (message if message else f"{downloaded_str} / {total_str}")
        + f"\t {bar} {percent}%"
        + colors.ENDC
    )
    sys.stdout.flush()

    if progress == total_size:
        sys.stdout.write("\n")
        sys.stdout.flush()
