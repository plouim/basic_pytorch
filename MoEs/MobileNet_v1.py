import math
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch import nn
import torch
from torch.utils.data import DataLoader
from torch import optim
from torchsummary import summary
import torch.nn.functional as F
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
def fit(epochs, model, loss_func, opt, train_loader, valid_loder, test_loader):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            xb, yb = xb.to(dev), yb.to(dev)
            loss = loss_func(model(xb), yb)

            loss.backward()
            opt.step()
        print("epoch {} loss:{}".format(epoch, loss.item()))
        model.eval()
        val_correct = 0
        for xb, yb in valid_loder:
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

class MobileNet_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(1, 32, 3, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU()
                )

        self.layer2 = nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1, groups=32), # DW
                nn.Conv2d(32, 64, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU()
                )

        self.layer_tmp = nn.Sequential(
                nn.Conv2d(64, 64, 3, 2, 1, groups=64), # DW
                nn.Conv2d(64, 128, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU()
                )

        self.layer3 = nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1, groups=128), # DW
                nn.Conv2d(128, 128, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, 2, 1, groups=128), # DW
                nn.Conv2d(128, 256, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU()
                )

        self.layer4 = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1, groups=256), # DW
                nn.Conv2d(256, 256, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1, groups=256), # DW
                nn.Conv2d(256, 256, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1, groups=256), # DW
                nn.Conv2d(256, 256, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1, groups=256), # DW
                nn.Conv2d(256, 256, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, 3, 1, 1, groups=256), # DW
                nn.Conv2d(256, 256, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AvgPool2d(3, 1)
                )

        self.fc_layer = nn.Sequential(
                nn.Linear(1024, 62),
                nn.ReLU()
                )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer_tmp(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(len(x), -1)
        x = self.fc_layer(x)
        return x
#======================================================================================
print("GPU is available") if torch.cuda.is_available() else print("GPU is NOT available")

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 32 
eta = 0.01

train_loader, valid_loader, test_loader = DataLoader(train_set, batch_size=batch_size), DataLoader(valid_set, batch_size=batch_size*2), DataLoader(test_set, batch_size)

loss_func = F.cross_entropy
model = MobileNet_v1()
model.to(dev)

summary(model, (1, 28, 28))
opt = optim.SGD(model.parameters(), lr=eta)
fit(20, model, loss_func, opt, train_loader, valid_loader, test_loader)
#======================================================================================
