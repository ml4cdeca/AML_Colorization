import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def load_places(folder):
    images = []
    for filename in os.listdir(folder):
        img = plt.imread(os.path.join(folder, filename))/255
        if img is not None and img.T.shape == (3, 256, 256):
            images.append(img.T)
    return torch.tensor(images).type(torch.FloatTensor)

def load_trainset(data_path):
    if data_path == './cifar-10':
        trainset = datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transforms.ToTensor())
        print('cifar loaded')
    elif 'places' in data_path:
        trainset = load_places(data_path)
        print('places loaded')    
    return trainset