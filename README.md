The class `PBDLDataset` processes and prepares datasets produced by physics solvers, making them suitable for training convolutional neural networks.

# Usage
A `PBDLDataset` instance must be initialized with the dataset name and the time steps between input and target sample. Optionally, it is possible to specify whether the data should be normalized.

    data = PBDLDataset("transonic-cylinder-flow-tiny", time_steps=20, normalized=True)

There are optional arguments `start_offset`, `end_offset` and `step_size` with which you can specify which frames of the simulation should be used. The argument `simulations` allows to choose only specific simulations for the dataset.

The dataset is then passed to the PBDLDataLoader, which creates batches from the input-target array. If required, the batch sampler can be used to ensure that the constants of all samples in a batch match.

    batch_sampler = PBDLConstantBatchSampler(dataset, BATCH_SIZE, group_constants=[0]) # sampler is optional
    dataloader = PBDLDataLoader(dataset, batch_sampler=batch_sampler)

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

- `fields` contains information about the type of physical field in the form
of a string, e.g. `VVdp` (velocity x, velocity y, density, pressure). Consecutive identical letters indicate that a physical field consists of several indices (vector field). This information affects how normalization is applied: For vector fields, the vector norm is applied first before the standard deviation is calculated.
- `field_desc`/`constant_desc` ist meant to be a description for the fields/constants.

# Partitioned Datasets
If the JSON attribute `endpoint` (for global dataset metadata) or `path`(for local dataset metadata) specifies a path (no `.npz` extension), the dataset is interpreted as a partitioned dataset. This means that simulations are saved in separate .npz files and can be downloaded separately (via the argument `simulations` of `PBDLDataset`).