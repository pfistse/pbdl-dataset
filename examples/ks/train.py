import sys
sys.path.append('../../')

import numpy as np

from dataset import PBDLDataset
from sampler import PBDLConstantBatchSampler
from dataloader import PBDLDataLoader

from ks_networks import ConvResNet1D
from ks_solver import DifferentiableKS

import torch
import matplotlib.pyplot as plt

# run configuration
CHANNELS = 16
DEPTH = 10
DATASET_FRACTION = 0.1

# default values
PREDHORI = 3  # nan issue for large values
DOMAIN_SIZE_BASE = 8
LR_FACTOR = 1.0
LR_GAMMA = 0.9
BATCH_SIZE = 16
EPOCHS = 4
DTPARTIAL = -1
TIMESTEP = 0.5
SN = 48

device = "cuda:0" if torch.cuda.is_available() else "cpu"

diff_ks = DifferentiableKS(resolution=SN, dt=TIMESTEP)

dataset = PBDLDataset(
    "ks-dataset-combined",
    time_steps=PREDHORI,
    step_size=50,
    time_stepping=True,
    normalize=False,
)

batch_sampler = PBDLConstantBatchSampler(dataset, BATCH_SIZE, group_constants=[0]) # group after first constant
dataloader = PBDLDataLoader(dataset, batch_sampler=batch_sampler)

net = ConvResNet1D(CHANNELS, DEPTH, device=device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4 * LR_FACTOR)
loss = torch.nn.MSELoss()

# print(dataset.__getitem__([0, 1, 2]))

for epoch in range(1):
    for i, (input, targets) in enumerate(dataloader):

        input = input.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # works only because domain size is not normalized (just a workaround)
        # sampler ensures that domain size is the same for the whole batch
        domain_size = input[0][1][0].item()

        inputs = [input]
        outputs = []

        for m in range(PREDHORI):
            output_solver = diff_ks.etd1(
                dataset.to_phiflow(inputs[-1]), DOMAIN_SIZE_BASE * domain_size
            )

            correction = diff_ks.dt * net(inputs[-1])

            output_combined = dataset.from_phiflow(output_solver) + correction

            # if torch.isnan(output_combined).any():
            #     print(f"NaN for m={m}")
            #     print(inputs[-1])
            #     print(dataset.from_phiflow(output_solver))
            #     print(correction)
            #     exit(1)

            outputs.append(output_combined)
            inputs.append(dataset.cat_constants(outputs[-1], inputs[0]))

        outputs = torch.stack(outputs, axis=1)

        loss_value = loss(outputs, targets)
        loss_value.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                f"[epoch: {epoch}, batch: {i}] loss: {(loss_value.item()*10000.) :.3f}"
            )

# training finished, evaluate net for one batch

input, targets = next(iter(dataloader))

domain_size = input[0][1][0].item()

inputs = [input]
outputs = []

for m in range(PREDHORI):
    output_solver = diff_ks.etd1(
        dataset.to_phiflow(inputs[-1]), DOMAIN_SIZE_BASE * domain_size
    )
    output_combined = dataset.from_phiflow(output_solver) + diff_ks.dt * net(
        inputs[-1]
    )

    outputs.append(output_combined)
    inputs.append(dataset.cat_constants(outputs[-1], inputs[0]))

outputs = torch.stack(outputs, axis=1)

input = inputs[0][0][0:1, ...].detach().numpy()
output = outputs[0][-1].detach().numpy()
target = targets[0][-1]

plt.subplot(4, 1, 1)
plt.imshow(input, cmap="magma", aspect=1)
plt.title("Input")

plt.subplot(4, 1, 2)
plt.imshow(output, cmap="magma", aspect=1)
plt.title("Output")

plt.subplot(4, 1, 3)
plt.imshow(target, cmap="magma", aspect=1)
plt.title("Target")

diff = target - output
plt.subplot(4, 1, 4)
plt.imshow(diff, cmap="gray", aspect=1)
plt.title("Difference Target Output")

plt.show()
