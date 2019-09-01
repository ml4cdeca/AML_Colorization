import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torchvision.datasets as datasets
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from skimage import color

def load_trainset(data_path,lab=False):
    if data_path == './cifar-10':
        trainset = datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transforms.ToTensor())
        print('cifar loaded')
    elif 'places' in data_path:
        trainset = PlacesDataset(data_path,lab=lab)
        print('places loaded')    
    return trainset

class PlacesDataset(Dataset):
    def __init__(self, path, transform=True, lab=False):
        self.path = path
        self.file_list = sorted(list(set(os.listdir(path))))
        self.transform = transform
        self.lab=lab
        #need to use transforms.Normalize in future but currently broken
        self.offset=np.array([50,0,0])
        self.range=np.array([50,128,128])
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, item):
        img_name = os.path.join(self.path,
                                self.file_list[item])
        image = plt.imread(img_name)/255
        if self.lab:
            image = (color.rgb2lab(image)-self.offset[None,None,:])/self.range[None,None,:]

        if self.transform:
            image = torch.tensor(np.transpose(image, (2,0,1))).type(torch.FloatTensor)
        return image