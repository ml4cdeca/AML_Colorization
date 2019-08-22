import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def load_places(folder):
    images = []
    for filename in os.listdir(folder):
        img = plt.imread(os.path.join(folder, filename))
        if img is not None and img.T.shape == (3, 256, 256):
            images.append(img.T)
    return torch.tensor(images).type(torch.FloatTensor)