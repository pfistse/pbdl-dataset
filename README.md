The class `PBDLDataset` processes and prepares datasets produced by physics solvers, making them suitable for training convolutional neural networks.

# Usage
A `PBDLDataset` instance must be initialized with the dataset name and the time steps between input and target sample. Optionally, it is possible to specify whether the data should be normalized.

    data = PBDLDataset("transonic-cylinder-flow-tiny", time_steps=20, normalized=True)

There are optional arguments `start_offset`, `end_offset` and `step_size` with which you can specify which frames of the simulation should be used. The argument `simulations`allows to choose only specific simulations for the dataset.

The dataset class is usually passed to a DataLoader, which creates batches from the input-target array.

    loader = DataLoader(data, batch_size=5, shuffle=True)

# Local Datasets
If you want to load your own data set, you can add a `datasets.json` file with the following structure in the same directory:

    {
        "dataset-name": {
            "path": "./path/to/dataset.npz",
            "fields": "VVdp",
            "field_desc" : ["channel 1", "channel 2", "channel 3"],
            "constant_desc" : ["mach number", "reynolds number"]
        }
    }

- `field` contains information about the type of physical field in the form
of a string, e.g. `VVdp` (velocity x, velocity y, density, pressure). Consecutive identical letters indicate that a physical field consists of several indices (vector field). This information affects how normalization is performed: For vector fields, the vector norm is applied first before the standard deviation is calculated.
- `field_desc`/`constant_desc` ist meant to be a description for the fields/constants.