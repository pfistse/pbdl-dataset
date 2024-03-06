import os
import torch
from dataset import PBDLDataset
from torch.utils.data import DataLoader
import net_small

TIME_STEPS = 10

BATCH_SIZE = 3
LR = 0.0001
EPOCHS = 8

MODEL_PATH = "model/small"

train_data = PBDLDataset("transonic-cylinder-flow-tiny", time_steps=TIME_STEPS, normalize=True)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

net = net_small.NetworkSmall()
criterionL2 = torch.nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=0.0)

print("Start training...")

for epoch in range(EPOCHS):
    for i, (input, target) in enumerate(train_dataloader):

        net.zero_grad()
        output = net(input)

        loss = criterionL2(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"epoch { epoch }, batch { i }, loss { loss.item() }")

print("Training finished")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(net.state_dict(), MODEL_PATH)
print("Neural network saved")
