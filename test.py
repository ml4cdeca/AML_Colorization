import torch
import os
from models.unet import unet
from models.model import model
from torchvision import transforms
from settings import s
import torchvision.datasets as datasets
from torch.utils.data import dataloader
import sys, getopt
import numpy as np
import matplotlib.pyplot as plt
from show import show_colorization
from functions import load_trainset
def main(argv):
    
    data_path = s.data_path
    weight_path = s.weights_path
    mode=0
    try:
        opts, args = getopt.getopt(argv,"h:w:p:b:m:",["help", "weight-path", "datapath",'model'])
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
        elif opt in ('-m','--model'):
            if arg in ('u'):
                mode=1
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if data_path == './cifar-10':
        in_size = 32
    elif 'places' in data_path:
        in_size = 256
    in_shape=(3,in_size,in_size)
    #out_shape=(s.classes,32,32)

    trainset = load_trainset(data_path)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=3,
                                        shuffle=True, num_workers=2)
    print("Loaded dataset from", data_path)
    
    try:
        UNet=model() if mode==0 else unet()
        #load weights
        try:
            UNet.load_state_dict(torch.load(weight_path, map_location='cpu'))
            print("Loaded network weights from", weight_path)
        except FileNotFoundError:
            print("Did not find weight files.")
            sys.exit(2)
    except RuntimeError:
        #if the wrong mode was chosen: try the other one
        UNet=model() if mode==1 else unet()
        #load weights
        try:
            UNet.load_state_dict(torch.load(weight_path, map_location='cpu'))
            print("Loaded network weights from", weight_path)
        except FileNotFoundError:
            print("Did not find weight files.")
            sys.exit(2)  
    UNet.eval()
    gray = torch.tensor([0.2989 ,0.5870, 0.1140])[:,None,None].float()
    with torch.no_grad():
        for i,batch in enumerate(trainloader):
            if data_path == './cifar-10':
                (image,_) = batch
            elif 'places' in data_path:
                image = batch
            #convert to grayscale image
            
            #using the matlab formula: 0.2989 * R + 0.5870 * G + 0.1140 * B and load data to gpu
            X=(image.clone()*gray).sum(1).to(device).view(-1,1,*in_shape[1:])
            image=image.float().to(device)
            #print(X.min(),X.max())
            #generate colorized version with unet
            try:
                unet_col=UNet(X)
            except:
                unet_col=UNet(torch.stack((X,X,X),1)[:,:,0,:,:])
            show_colorization(unet_col,image,X)
if __name__ == '__main__':
    main(sys.argv[1:])