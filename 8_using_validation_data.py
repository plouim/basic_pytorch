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

# define accuracy
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

# convert data to tensor type
(x_train, y_train, x_valid, y_valid) = map(torch.tensor,
                                            (x_train, y_train, x_valid, y_valid))
# define constant
num_data, _ = x_train.shape
bs = 64     # batch size
eta = 0.1   # learning rate
epochs = 2  # epochs

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs*2)

class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784,10)
    
    def forward(self, xb):
        return self.lin(xb)

loss_func = F.cross_entropy

# batch data
xb = x_train[0:bs]
yb = y_train[0:bs]

# define model
model, opt = get_model()

def fit():
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()
        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)
        print(epoch, valid_loss/len(valid_dl))

fit()
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
