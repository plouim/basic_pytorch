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

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784,10)
    
    def forward(self, xb):
        return self.lin(xb)

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    
    def forward(self, x):
        return self.func(x)

class WrappedDataLoader():
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)
#model = Mnist_CNN()
# model = nn.Sequential(
# #            Lambda(preprocess),
            # nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
# #            nn.AvgPool2d(4),
            # nn.AdaptiveAvgPool2d(1),
            # Lambda(lambda x: x.view(x.size(0), -1)),
        # )
#model.to(dev)
#opt = optim.SGD(model.parameters(), lr=eta, momentum=0.9)
#fit(epochs, model, loss_func, opt, train_dl, valid_dl)
# model = nn.Sequential(
        # nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5),
        # nn.Tanh(),
        # nn.AvgPool2d(kernel_size=2, stride=2),
        # nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5),
        # nn.Tanh(),
        # nn.AvgPool2d(kernel_size=2, stride=2),
        # nn.Linear(in_features=50*4*4, out_features=500)
        # )
class my_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.
model = my_model()
print(model(x_train[0].reshape(1,28,28)))
print(model(x_train[0].reshape(1,28,28)).shape)




