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

def nll(pred, target):
    return -pred[range(target.shape[0]), target].mean()

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

# initialize weights and biases
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
#        self.weights = nn.Parameter(torch.randn(784,10) / math.sqrt(784))
#        self.bias = nn.Parameter(torch.zeros(10))
        self.lin = nn.Linear(784,10)
    
    def forward(self, xb):
#        return xb @ self.weights + self.bias
        return self.lin(xb)

# define loss function
loss_func = F.cross_entropy

# batch data
xb = x_train[0:bs]
yb = y_train[0:bs]

# define model
model = Mnist_Logistic()

def fit():
    for epoch in range(epochs):
        for i in range( (num_data-1)//bs+1 ):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]

            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= eta * p.grad
                model.zero_grad()
fit()
print(loss_func(model(xb), yb), accuracy(model(xb), yb))
