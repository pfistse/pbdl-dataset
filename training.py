import os
import torch
from dataset import PBDLDataset
from torch.utils.data import DataLoader
import net_small

STEPS = 20

BATCH_SIZE = 3
LR = 0.0001
EPOCHS = 5

MODEL_PATH = "model/small"

train_data = PBDLDataset("karman-2d-train", time_steps=STEPS, normalized=True)
train_data.normalize()

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

net = net_small.NetworkSmall()
criterionL2 = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.0)

print("start training...")

for epoch in range(EPOCHS):
    for i, (input, target) in enumerate(train_dataloader):

        net.zero_grad()
        output = net(input)

        loss = criterionL2(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"epoch { epoch }, batch { i }, loss { loss.item() }")

print("training finished")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(net.state_dict(), MODEL_PATH)
print("neural network saved")
