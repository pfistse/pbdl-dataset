# non-local package imports
import numpy as np
import torch
import torch.utils.data
import phi.torch.flow as pf

# local package imports
import pbdl.dataset

PHIFLOW_SPATIAL_DIM = ["x", "y", "z"]


class Dataset(pbdl.dataset.Dataset):

    def __init__(self, *args, intermediate_time_steps=False, **kwargs):
        self.intermediate_time_steps = intermediate_time_steps

        super().__init__(*args, **kwargs)

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

        if self.intermediate_time_steps:
            target = sim[input_frame_idx + 1 : target_frame_idx + 1]
        else:
            target = sim[target_frame_idx]

        # normalize
        if self.normalize:
            input /= self.data_std
            target /= (
                self.data_std[None, :, :]  # add frame dimension
                if self.intermediate_time_steps
                else self.data_std
            )

            if (abs(self.const_std) < 10e-10).any():  # TODO
                const = np.zeros_like(const)
            else:
                const = (const - self.const_mean) / self.const_std

        return (
                input,
                tuple(const), # required by loader
                target,
                tuple(sim.attrs["const"]),  # solver needs non-normalized constants
            )

    def to_phiflow(self, data):
        """Convert network input to solver input. Constant layers are ignored."""
        spatial_dim = ",".join(PHIFLOW_SPATIAL_DIM[0 : self.num_spatial_dim])

        # if necessary, cut off constant layers
        data = data[:, 0 : self.num_fields, ...]

        if self.normalize:
            data = data * torch.tensor(self.data_std)

        return pf.tensor(
            data,
            pf.batch("b"),
            pf.instance("time"),
            pf.spatial(spatial_dim),
        )

    def from_phiflow(self, data):
        """Convert solver output to a network output-like format."""
        spatial_dim = ",".join(PHIFLOW_SPATIAL_DIM[0 : self.num_spatial_dim])

        data = data.native(["b", "time", spatial_dim])

        if self.normalize:
            data = data / torch.tensor(self.data_std)

        return data

    def cat_constants(self, data, like):
        """Concatenate constants from `like` to `data`. Useful for mapping network outputs to network inputs of the next iteration."""
        return torch.cat(
            [data, like[:, self.num_fields : self.num_fields + self.num_const, ...]],
            axis=1,
        )  # dim 0 is batch dimension
