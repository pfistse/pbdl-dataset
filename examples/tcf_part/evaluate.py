import sys

sys.path.append("../../")

import numpy as np

import torch
from dataset import PBDLDataset
from dataloader import PBDLDataLoader
from torch.utils.data import DataLoader

from examples.tcf.net_small import NetworkSmall

import matplotlib.pyplot as plt

STEPS = 10
BATCH_SIZE = 3
MODEL_PATH = "model/small"

test_data = PBDLDataset(
    "transonic-cylinder-flow-tiny", time_steps=STEPS, normalize=True
)
test_dataloader = PBDLDataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

net = NetworkSmall()
net.load_state_dict(torch.load(MODEL_PATH))
net.eval()

input, target = next(iter(test_dataloader))
output = net(input)

input = input.numpy()
target = target.numpy()
output = output.detach().numpy()

plt.subplot(1, 4, 1)
plt.imshow(np.flip(input[0, 1, ...], axis=-2), cmap="magma")
plt.title("Input")

plt.subplot(1, 4, 2)
plt.imshow(np.flip(output[0, 1, ...], axis=-2), cmap="magma")
plt.title("Output")

plt.subplot(1, 4, 3)
plt.imshow(np.flip(target[0, 1, ...], axis=-2), cmap="magma")
plt.title("Target")

diff = target[0, 1, ...] - output[0, 1, ...]
plt.subplot(1, 4, 4)
plt.imshow(np.flip(diff, axis=-2), cmap="gray")
plt.title("Difference")

plt.show()
