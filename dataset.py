import os
import urllib.request
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np

# TODO load endpoints from server?
dataset_metadata = {
    "karman-2d-train": {
        "endpoint": "https://physicsbaseddeeplearning.org/data/sol-karman-2d-train.pickle",
        "sim_key_scheme": "karman-fdt-hires-set/sim_%06d",
    },
    "karman-2d-test": {
        "endpoint": "https://physicsbaseddeeplearning.org/data/sol-karman-2d-test.pickle",
        "sim_key_scheme": "karman-fdt-hires-testset/sim_%06d",
    },
}

# TODO to be loaded from dataset file
constant_data = [[160000.0, 320000.0, 640000.0, 1280000.0, 2560000.0, 5120000.0]]

DATASET_DIR = "datasets/"
DATASET_EXT = ".pickle"

class PBDLDataset(Dataset):
    def __init__(self, dataset, time_steps, normalized=True):
        # NOTE batching is handled by DataLoader

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
            self.data_preloaded = pickle.load(f)

        self.num_sims = len(self.data_preloaded)

        # no need to keep the simulation keys
        self.data_preloaded = [
            self.data_preloaded[dataset_metadata[dataset]["sim_key_scheme"] % i]
            for i in range(self.num_sims)
        ]

        # get the number of sim steps for the first sim (assuming they are all equal)
        self.sim_num_steps = len(self.data_preloaded[0])

        print(
            f"successfully loaded { self.dataset }, found { self.num_sims } simulations"
        )

        # trimming and padding is necessary because during the data generation process
        # the data was represented using a staggered grid where each cell face stores
        # data rather than each cell center (fencepost error)
        for sim in self.data_preloaded:
            for step in range(len(sim)):
                sim[step] = (
                    # density remains unchanged
                    sim[step][0],
                    # pad x velocity with zeros
                    np.pad(
                        sim[step][1],
                        [(0, 0), (0, 0), (0, 1)],
                    ),
                    # trim y velocity
                    sim[step][2][:, 1:],
                )

        # calculate the standard deviation for normalization
        vel_std = np.std(
            np.concatenate(
                [
                    # length of velocity vector
                    np.linalg.norm(
                        np.stack(
                            (
                                sim[step][1],
                                sim[step][2],
                            ),
                            axis=-1,
                        ),
                        axis=-1,
                    ).reshape(-1)
                    for sim in self.data_preloaded
                    for step in range(self.time_steps)
                ],
                axis=-1,
            )
        )

        den_std = np.std(
            np.concatenate(
                [
                    np.absolute(sim[step][0].reshape(-1))
                    for sim in self.data_preloaded
                    for step in range(self.time_steps)
                ],
                axis=-1,
            )
        )

        # TODO correct std calculation? (taken from solver-in-the-loop example)
        con_std = [
            np.std(list(map(np.absolute, constant))) for constant in constant_data
        ]

        self.data_stats = {
            "std": (
                den_std,
                vel_std,
                vel_std,
            ),
            "ext.std": con_std,
        }

        self.normalized = False
        if normalized:
            self.normalize()

    def __len__(self):
        return self.num_sims * (self.sim_num_steps - self.time_steps)

    def __getitem__(self, idx):
        """The data provided has the shape [channels (dens/vel/constants), y-size, x-size]."""

        # create input-target pairs with interval time_steps from simulation steps
        samples_per_sim = self.sim_num_steps - self.time_steps
        sim_idx = idx // samples_per_sim

        input_idx = idx % samples_per_sim
        target_idx = input_idx + self.time_steps

        input = self.data_preloaded[sim_idx][input_idx]
        target = self.data_preloaded[sim_idx][target_idx]

        # necessary because Conv2d does only accept 4 dims and batch dim is added by DataLoader
        input = np.squeeze(input, axis=1)
        target = np.squeeze(target, axis=1)

        # additional layers for constants
        input_ext = [
            np.full_like(
                input[0],
                constant[sim_idx] / std if self.normalized else constant[sim_idx],
            )
            for constant, std in zip(constant_data, self.data_stats["ext.std"])
        ]

        return (
            torch.tensor(np.concatenate((input, input_ext), axis=0)),
            torch.tensor(target),
        )

    def normalize(self):
        if self.normalized:
            return

        self.normalized = True

        for sim in self.data_preloaded:
            for step in range(len(sim)):
                sim[step] = [
                    field / std for field, std in zip(sim[step], self.data_stats["std"])
                ]

    def denormalize(self):
        if not self.normalized:
            return

        self.normalized = False

        for sim in self.data_preloaded:
            for step in range(len(sim)):
                sim[step] = [
                    field * std for field, std in zip(sim[step], self.data_stats["std"])
                ]