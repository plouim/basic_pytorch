from pathlib import Path
import requests
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np

# ===================================================================================
PATH = Path('data')
DATA_PATH = PATH / 'mnist'
PATH.mkdir(parents=True, exist_ok=True)

# if file isn't existing, then get the file from github repo
URL = 'https://github.com/pytorch/tutorials/raw/main/_static/'
FILE_NAME = 'mnist.pkl.gz'
if not (DATA_PATH / FILE_NAME).exists():
    content = requests.get(URL+FILE_NAME).content
    (DATA_PATH / FILE_NAME).open('wb').write(content)

# unzip mnist file
with gzip.open((DATA_PATH / FILE_NAME).as_posix(), 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
# ===================================================================================

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=eta)

def get_data(train_ds, valid_ds, batch_size):
    return(
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(valid_ds, batch_size=batch_size*2),
            )
# define accuracy
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)
        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                    *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
                    )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print(epoch, val_loss)

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784,10)
    
    def forward(self, xb):
        return self.lin(xb)

class Mnist_CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
            self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)
            
        def forward(self, xb):
            xb = xb.view(-1, 1, 28, 28)
            xb = F.relu(self.conv1(xb))
            xb = F.relu(self.conv2(xb))
            xb = F.relu(self.conv3(xb))
            xb = F.avg_pool2d(xb, 4)
            return xb.view(-1, xb.size(1))

# convert data to tensor type
(x_train, y_train, x_valid, y_valid) = map(torch.tensor,
                                            (x_train, y_train, x_valid, y_valid))
# define constant
num_data, _ = x_train.shape
bs = 64     # batch size
eta = 0.1   # learning rate
epochs = 2  # epochs

train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)

loss_func = F.cross_entropy

# train
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
#model, opt = get_model()
model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=eta, momentum=0.9)
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
