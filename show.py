import torch
import matplotlib.pyplot as plt
import numpy as np
from unet.dataset import gta_dataset, city_dataset
from settings import s
from unet.unet import unet

#not really beautiful 
def show_colorization(pred,truth=None,original=None):
    N = 1
    if len(pred.shape)==4:
         N = pred.shape[0]
    M = 1+(1 if not truth is None else 0)+(1 if not original is None else 0)
    plt.figure(figsize=(5, N*5/M))
    counter=np.arange(1,1+N*M).reshape(M,N)
    for i in range(N):
        if truth is None and original is None:
            plt.imshow(pred[i].detach().cpu().numpy())
        elif original is None:
            #print(truth.shape,pred.shape)
            plt.subplot(N,2,counter[i,0])
            if i==0:
                plt.title('colorization')
            plt.axis('off')
            plt.imshow(np.transpose(pred[i].detach().cpu().numpy(),(1,2,0)))
            plt.subplot(N,2,counter[i,1])
            if i==0:
                plt.title('ground truth')
            plt.axis('off')
            plt.imshow(np.transpose(truth[i].detach().cpu().numpy(),(1,2,0)))
        else:
            #print(N,truth.shape,pred.shape,original.shape)
            plt.subplot(N,3,counter[i,0])
            if i==0:
                plt.title('Input image')
            plt.axis('off')
            plt.imshow(.5*(1+original[i].detach().cpu().numpy()[0]),cmap='gray')
            plt.subplot(N,3,counter[i,1])
            if i==0:
                plt.title('Ground truth')
            plt.axis('off')
            plt.imshow(np.transpose(truth[i].detach().cpu().numpy(),(1,2,0)))
            plt.subplot(N,3,counter[i,2])
            if i==0:
                plt.title('colorization')
            plt.imshow(np.transpose(pred[i].detach().cpu().numpy(),(1,2,0)))
            plt.axis('off')
    plt.show()