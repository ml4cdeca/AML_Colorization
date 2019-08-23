import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torchvision.datasets as datasets
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

def load_trainset(data_path):
    if data_path == './cifar-10':
        trainset = datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transforms.ToTensor())
        print('cifar loaded')
    elif 'places' in data_path:
        trainset = PlacesDataset(data_path)
        print('places loaded')    
    return trainset

class PlacesDataset(Dataset):
    def __init__(self, path, transform=True):
        self.path = path
        self.file_list = sorted(list(set(os.listdir(path))))
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, item):
        img_name = os.path.join(self.path,
                                self.file_list[item])
        image = plt.imread(img_name)/255
        if self.transform:
            image = torch.tensor(np.transpose(image, (2,0,1))).type(torch.FloatTensor)
        return image