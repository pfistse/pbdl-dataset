import numpy as np

import torch
from dataset import PBDLDataset
from torch.utils.data import DataLoader

from net_small import NetworkSmall

import matplotlib.pyplot as plt

STEPS = 20
BATCH_SIZE = 3
MODEL_PATH = "model/small"

train_data = PBDLDataset("karman-2d-train", time_steps=STEPS, normalized=True)
test_data = PBDLDataset("karman-2d-test", time_steps=STEPS, normalized=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

net = NetworkSmall()
net.load_state_dict(torch.load(MODEL_PATH))
net.eval()

input, target = next(iter(test_dataloader))
output = net(input)

input = input.numpy()
target = target.numpy()
output = output.detach().numpy()

plt.subplot(1, 3, 1)
plt.imshow(np.flip(output[0, 2, ...], axis=-2), cmap="magma")
plt.title("Output")

plt.subplot(1, 3, 2)
plt.imshow(np.flip(target[0, 2, ...], axis=-2), cmap="magma")
plt.title("Target")


diff = target[0, 2, ...] - output[0, 2, ...]
plt.subplot(1, 3, 3)
plt.imshow(np.flip(diff, axis=-2), cmap="gray")
plt.title("Difference")

plt.show()