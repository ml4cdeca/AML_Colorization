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
    def __init__(self, path, transform=True, lab=False, bins=False):
        self.path = path
        self.file_list = sorted(list(set(os.listdir(path))))
        self.transform = transform
        self.lab=lab
        #need to use transforms.Normalize in future but currently broken
        self.offset=-np.array([0,128,128])
        self.range=np.array([100,255,255])
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, item):
        img_name = os.path.join(self.path,
                                self.file_list[item])
        image = plt.imread(img_name)/255
        if self.lab:
            image = (color.rgb2lab(image)-self.offset[None,None,:])/self.range[None,None,:]
            if bins:
                image[:,:,1:]=ab2bins(image[:,:,1:])
        if self.transform:
            image = torch.tensor(np.transpose(image, (2,0,1))).type(torch.FloatTensor)
        return image



#image preprocessing

#distance matrix
def dist_mat(X,Y):
    return -2 * X@Y.T + np.sum(Y**2, axis=1) + np.sum(X**2, axis=1)[:, None]
   
bins=np.load('resources/norm_bins.npy')
def ab2bins(image):
    #takes image with only ab channels and returns the 
    shape=image.shape
    mbsize = shape[0] if len(shape)==4 else 1
    bin_rep = dist_mat(bins,image.reshape(-1,2)).argmin(0).reshape(mbsize,shape[2 if len(shape)==4 else 1],-1,1)
    if len(shape)==4:
        return bin_rep
    else:
        return bin_rep[0]
        
def bins2lab(bin_rep,L=None):
    #takes bins representation of an image and returns rgb if Lightness is provided. Else only ab channel
    mbsize=1 if len(bin_rep.shape)==3 else bin_rep.shape[0]
    size=bin_rep.shape[2] if len(bin_rep.shape)==4 else bin_rep.shape[1]
    ab=bins[bin_rep.flatten()].reshape(mbsize,size,-1,2)
    if not L is None:
        ab=np.concatenate((L.reshape(mbsize,size,-1,1),ab),3)
    
    if len(bin_rep.shape)==3:
        ab=ab[0]
    return ab