import matplotlib
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from SEP_AI_loss import Loss
from SEP_AI_model import Net
from SEP_AI_dataset import VOCDataset
from SEP_AI_train import (
    transform,
    IMG_DIR,
    LABEL_DIR,
)

custom_dataset = VOCDataset("archive/100examples.csv",
                            transform=transform,
                            img_dir=IMG_DIR,
                            label_dir=LABEL_DIR,
                            )
model = Net(split_size=7, num_boxes=2, num_classes=20)

loss = Loss()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

optimizer = optim.Adam(
    model.parameters(), lr=2e-5, weight_decay=0
)

dataloader = DataLoader(dataset=custom_dataset, batch_size=8, shuffle=True)

loop = tqdm(dataloader, leave=True)

model.to(device)
model.train()
epochs = 20
optimizer.zero_grad()
losses = 0
loss_list = []

for i in range(epochs):

    for batch_idx, (inp, target) in enumerate(loop):
        optimizer.zero_grad()
        inp = inp.to(device)
        target = target.to(device)

        out = model(inp)

        # print("Target shape is: ",target.shape)
        # print("output size is: ", out.shape)
        losses = loss(out, target)
        loss_list.append(losses.item())

        losses.backward()
        optimizer.step()
        loop.set_postfix(loss=losses.item())
        del inp
        del target
        del out
    print(f"Mean loss was {sum(loss_list) / len(loss_list)}")


plt.figure()
plt.plot(loss_list)
plt.show()
