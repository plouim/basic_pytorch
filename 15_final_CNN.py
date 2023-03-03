import pickle, gzip
from pathlib import Path
from mnist_sample import *
     

PATH = Path('data')/'mnist'

with gzip.open(PATH/'mnist.pkl.gz', 'rb') as f:
    ((train_x, train_y), (valid_x, valid_y), _) = pickle.load(f, encoding='latin-1')
     

bs=64
lr=0.1
epochs=20

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
     

def preprocess(x,y): return x.view(-1,1,28,28).to(dev),y.to(dev)

def get_dataloader(x,y,bs,shuffle):
    ds = TensorDataset(*map(tensor, (x,y)))
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle)
    return WrappedDataLoader(dl, preprocess)
     

train_dl = get_dataloader(train_x, train_y, bs,   shuffle=False)
valid_dl = get_dataloader(valid_x, valid_y, bs*2, shuffle=True )
     

model = nn.Sequential(
    nn.Conv2d(1,  16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0),-1))
).to(dev)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
     

fit(epochs, model, F.cross_entropy, opt, train_dl, valid_dl)
