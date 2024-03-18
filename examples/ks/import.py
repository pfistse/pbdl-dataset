import numpy as np

domain_size = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

data = np.load(f"ks-dataset-0.npz")["arr_0"]
constants = [[domain_size[0]]] * data.shape[0]

for i in range(1, 7):
    data_ext = np.load(f"ks-dataset-{i}.npz")["arr_0"]
    data = np.concatenate([data, data_ext], axis=0)
    constants.extend([[domain_size[i]]] * data_ext.shape[0])

data = np.expand_dims(data, axis=2)  # add field dimension
constants = np.array(constants)

print(f"data shape: {data.shape}")
print(f"constants shape: {constants.shape}")

new_npz = {"data": data, "constants": constants}
np.savez("ks-dataset-combined.npz", **new_npz)
