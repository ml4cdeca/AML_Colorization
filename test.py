import torch
import os
from unet.model import unet
from torchvision import transforms
from settings import s
import torchvision.datasets as datasets
from torch.utils.data import dataloader
import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
from show import show_colorization
def main(argv):
    
    data_path = './cifar-10'
    weight_path = s.weights_path
    batch_norm=s.batch_norm
    try:
        opts, args = getopt.getopt(argv,"h:w:p:b:",["help", "weight-path", "datapath"])
    except getopt.GetoptError as error:
        print(error)
        print( 'test.py -i <Boolean> -s <Boolean>')
        sys.exit(2)
    print("opts", opts)
    for opt, arg in opts:
        if opt == '-h':
            print( 'test.py -i <Boolean> -s <Boolean>')
            sys.exit()
        elif opt in ("-w", "--weight-path"):
            weight_path = arg
        elif opt in ("--datapath", "-p"):
            data_path = arg
        elif opt in ("--batchnorm", "-b"):
            batch_norm = arg in ["True", "true", "1"]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_shape=(3,32,32)
    out_shape=(s.classes,32,32)
    trainset = datasets.CIFAR10(root='./cifar-10', train=False,
                                        download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=3,
                                        shuffle=True, num_workers=2)
    print("Loaded dataset from", data_path)
    
    #define model
    UNet=unet().to(device)
    #load weights
    try:
        UNet.load_state_dict(torch.load(weight_path))
        print("Loaded network weights from", weight_path)
    except FileNotFoundError:
        print("Did not find weight files.")
        sys.exit(2)
    UNet.eval()
    gray = torch.tensor([0.2989 ,0.5870, 0.1140])[:,None,None].float()
    with torch.no_grad():
        for i,(image,c) in enumerate(trainloader):
            #convert to grayscale image
            
            #using the matlab formula: 0.2989 * R + 0.5870 * G + 0.1140 * B and load data to gpu
            X=(image.clone()*gray).sum(1).to(device).view(-1,1,*in_shape[1:])
            image=image.float().to(device)
            #generate colorized version with unet
            unet_col=UNet(X)
            show_colorization(unet_col,image,X)
if __name__ == '__main__':
    main(sys.argv[1:])