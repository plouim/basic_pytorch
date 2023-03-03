import math
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch import nn
import torch
from torch import optim
from torchsummary import summary
#======================================================================================
train_ratio = 0.8
valid_ratio = 0.2
dataset = torchvision.datasets.EMNIST(
    root = './data/EMNIST',
    split = 'byclass',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])
)
train_size = math.ceil(len(dataset) * train_ratio)
valid_size = math.floor(len(dataset) * valid_ratio)
print(f"train size: {train_size}, valid size: {valid_size}")
train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
test_set = torchvision.datasets.EMNIST(
    root = './data/EMNIST',
    split = 'byclass',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])
)
test_size = len(test_set)
#======================================================================================
def fit(epochs, model, loss_func, opt, train_loader, test_loader):
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(dev)
            yb = yb.to(dev)
            out = model(xb)
            loss = loss_func(out, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(epoch, loss.item())
        with torch.no_grad():
            val_correct=0
            for xb, yb in valid_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                val_correct += sum(model(xb).argmax(dim=1) == yb)
            print("validation accuaray: {} / {} -> {:.1f}%".format(val_correct, valid_size, 100 * val_correct / valid_size))
        print()
    with torch.no_grad():
        correct = 0
        for xb, yb in test_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            correct += sum(model(xb).argmax(dim=1) == yb)
        print("test accuarcy: {} / {} -> {:.1f}%".format(correct, test_size, 100 * correct / test_size))
#======================================================================================
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
                nn.Conv2d(1, 20, 5),
                nn.ReLU(),
                nn.AvgPool2d(2, 2),
                nn.Conv2d(20, 50, 5),
                nn.ReLU(),
                nn.AvgPool2d(2, 2)
                )
        self.fc_layer = nn.Sequential(
                nn.Linear(4*4*50, 500),
                nn.ReLU(),
                nn.Linear(500, 62),
                )

    def forward(self, x):
        x = self.layer(x)
        x = x.view(len(x), -1)
        x = self.fc_layer(x)
        return x
#======================================================================================
print("GPU is available") if torch.cuda.is_available() else print("GPU is NOT available")

batch_size = 64
eta = 0.01

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = Model()
model.to(dev)
summary(model, (1, 28, 28))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size*2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size*2)

loss_func = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=eta)
fit(20, model, loss_func, opt, train_loader, test_loader)
summary(model, (1, 28, 28))
#======================================================================================

