import torch
import matplotlib.pyplot as plt
import numpy as np
from unet.dataset import gta_dataset, city_dataset
from settings import s
from unet.model import unet

#not really beautiful 
def show_colorization(pred,truth=None,original=None):
    N = 1
    if len(pred.shape)==4:
         N = pred.shape[0]
    for i in range(N):
        plt.figure(figsize=(10, 2))
        if truth is None and original is None:
            plt.imshow(pred[i].detach().cpu().numpy())
            plt.show()
        elif original is None:
            #print(truth.shape,pred.shape)
            plt.subplot(121)
            plt.title('colorization')
            plt.axis('off')
            plt.imshow(np.transpose(pred[i].detach().cpu().numpy(),(1,2,0)))
            plt.subplot(122)
            plt.title('ground truth')
            plt.axis('off')
            plt.imshow(np.transpose(truth[i].detach().cpu().numpy(),(1,2,0)))
            plt.show()
        else:
            #print(original.shape,original.max(),original.min())
            plt.subplot(131)
            plt.title('Input image')
            plt.axis('off')
            plt.imshow(.5*(1+np.transpose(original[i].detach().cpu().numpy(),(1,2,0))))
            plt.subplot(132)
            plt.title('Ground truth')
            plt.axis('off')
            plt.imshow(truth[0].detach().cpu().numpy())
            plt.subplot(133)
            plt.title('colorization')
            plt.imshow(pred[0].detach().cpu().numpy())
            plt.axis('off')
            plt.show()