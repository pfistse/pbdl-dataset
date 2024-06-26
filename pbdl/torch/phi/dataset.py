# non-local package imports
import numpy as np
import torch
import torch.utils.data
import phi.torch.flow as pf

# local package imports
import pbdl.dataset

PHIFLOW_SPATIAL_DIM = ["x", "y", "z"]

class Dataset(pbdl.dataset.Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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