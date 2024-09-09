from itertools import groupby


def get_sel_const_sim(dset, sim, sel_const):
    attrs = dset["sims/sim" + str(sim)].attrs
    const = sel_const if sel_const else dset["sims/"].attrs["Constants"]
    return [attrs[key] for key in const]


def get_const_sim(dset, sim):
    attrs = dset["sims/sim" + str(sim)].attrs
    return [attrs[key] for key in dset["sims/"].attrs["Constants"]]


def get_meta_data(dset):
    # convert_key = lambda key: key.lower().replace(" ", "_")

    field_mapping = {
        "PDE": "pde",
        "Fields Scheme": "fields_scheme",
        "Fields": "fields",
        "Constants": "const",
        "Dt": "dt",
    }

    meta_attrs = dset["sims"].attrs

    meta = {field_mapping[field]: meta_attrs[field] for field in field_mapping.keys()}

    # retrieve remaining metadata from the first simulation
    first_sim = dset["sims"][next(iter(dset["sims"]))]
    sim_shape = first_sim.shape
    num_fields = len(list(groupby(meta["fields_scheme"])))  # TODO
    num_spatial_dim = len(sim_shape) - 2  # subtract 2 for frame and field dimension

    meta.update(
        {
            "num_sims": len(dset["sims"]),
            "num_const": len(meta["const"]),
            "sim_shape": sim_shape,
            "num_frames": sim_shape[0],
            "num_sca_fields": sim_shape[1],
            "num_fields": num_fields,
            "num_spatial_dim": num_spatial_dim,
        }
    )

    return meta
