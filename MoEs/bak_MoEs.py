import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
#======================================================================================
train_set = torchvision.datasets.EMNIST(
    root = './data/EMNIST',
    split = 'byclass',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])
)
test_set = torchvision.datasets.EMNIST(
    root = './data/EMNIST',
    split = 'byclass',
    train = False,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

print(train_set.shape)
#======================================================================================
