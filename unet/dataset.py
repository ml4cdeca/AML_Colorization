from torch.utils.data import DataLoader, Dataset
import torch
import torchvision.transforms as trafo
from torch import utils
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from settings import s


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

class gta_dataset(Dataset):
    def __init__(self, path="data_red", scale=1., crop_size=(0, 0), debug=False, labels_only=False):
        super().__init__()
        #path = os.path.join(os.getcwd(), path)

        self.labels_only = labels_only
        self.debug = debug
        self.path = path
        self.file_list = sorted(list(set(os.listdir(path + "/labels")) & set(os.listdir(path + "/images"))))

        self.crop_size = crop_size
        self.scale = scale!=1

        self.label2train = np.array(s.label2train)[:,1]
        self.labelnames = s.label
        # get image dimensions:
        filename = os.listdir(path + "/labels/")[0]
        W, H = Image.open(path + "/labels/" + filename).size
        if scale != 1:
            scale_trafo = trafo.Resize(size=(int(H * scale), int(W * scale)), interpolation=Image.NEAREST)
            self.im_trafo = trafo.Compose(
                [scale_trafo, trafo.ToTensor(), trafo.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.label_trafo = scale_trafo
        else:
            self.im_trafo = trafo.Compose(
                [trafo.ToTensor(), trafo.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        filename = self.file_list[item]
        if not self.labels_only:
            im = self.im_trafo(Image.open(self.path + "/images/" + filename))
        if self.scale:
            lbl = np.array(self.label_trafo(Image.open(self.path + "/labels/" + filename)))
        else:
            lbl = np.array((Image.open(self.path + "/labels/" + filename)))


        # cropping of image: cannot use torchvision.transforms.RandomCrop as it would not apply same trafo to
        # label and image
        if self.crop_size != (0, 0) :
            x = (im.size(2) - self.crop_size[1] + 1)//2#np.random.randint(0, im.size(2) - self.crop_size[1] + 1)  # W
            y = (im.size(1) - self.crop_size[0] + 1)//2#np.random.randint(0, im.size(1) - self.crop_size[0] + 1)  # H
            if not self.labels_only:
                im = im[:, y: y + self.crop_size[0]:, x:x+self.crop_size[1]:]
            # attention: at this stage: lbl has shape(HxC)
            lbl = lbl[y: y + self.crop_size[0]:, x:x+self.crop_size[1]]
        # replacing class indices by train indices (see https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py)
        lbl = self.label2train[lbl.flatten()].reshape(lbl.shape)
        lbl = torch.tensor(lbl, dtype=torch.long)
        if self.debug:
            print("labels", lbl)
            print("len labels", lbl.size())
            print("type labels", lbl.dtype)
            print("original label type", lbl.dtype)

        if self.labels_only:
            return lbl
        else:
            return im, lbl

    def plot_instance(self, i, plot=True):
        image, label = self.__getitem__(i)
        image = image/2. + 0.5
        image = image.numpy().swapaxes(0, 2).swapaxes(0, 1)
        if plot:
            plt.subplot(211)
            plt.imshow(image)
            plt.subplot(212)
            plt.imshow(label)
            plt.show()
        return image, label


class city_dataset(Dataset):
    def __init__(self, path="cityscapes_data",
                 scale=1, crop_size=(0, 0), debug=False, labels_only=False):
        super().__init__()
        # path = os.path.join(os.getcwd(), path)

        self.labels_only = labels_only
        self.debug = debug
        self.path = path
        self.file_list = sorted(list(set(os.listdir(path + "/labels")) & set(os.listdir(path + "/images"))))

        self.crop_size = crop_size
        self.scale = scale != 1

        self.label2train = np.array(s.label2train)[:, 1]
        self.labelnames = s.label
        # get image dimensions:
        filename = os.listdir(path + "/labels/")[0]
        W, H = Image.open(path + "/labels/" + filename).size
        if scale != 1:
            scale_trafo = trafo.Resize(size=(int(H * scale), int(W * scale)), interpolation=Image.NEAREST)
            self.im_trafo = trafo.Compose(
                [scale_trafo, trafo.ToTensor(), trafo.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.label_trafo = scale_trafo
        else:
            self.im_trafo = trafo.Compose(
                [trafo.ToTensor(), trafo.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        filename = self.file_list[item]
        if not self.labels_only:
            im = self.im_trafo(Image.open(self.path + "/images/" + filename))
        if self.scale:
            lbl = np.array(self.label_trafo(Image.open(self.path + "/labels/" + filename)))[:,:,:3:]
        else:
            lbl = np.array((Image.open(self.path + "/labels/" + filename)))[:,:,:3:]
        # cropping of image: cannot use torchvision.transforms.RandomCrop as it would not apply same trafo to
        # label and image
        if self.crop_size != (0, 0):
            x = (im.size(2) - self.crop_size[1] + 1)//2#np.random.randint(0, im.size(2) - self.crop_size[1] + 1)  # W
            y = (im.size(1) - self.crop_size[0] + 1)//2#np.random.randint(0, im.size(1) - self.crop_size[0] + 1)  # H
            if not self.labels_only:
                im = im[:, y: y + self.crop_size[0]:, x:x+self.crop_size[1]:]
            # attention: lbl is in numpy image format (shape: (HxWxC)) insteat of torch format (shape: (CxHxW))
            lbl = lbl[y: y + self.crop_size[0]:, x:x+self.crop_size[1]:, :]

        # replacing colours in label image by class indices (attention: only train on 19 out of the given 33 classes!)
        lbl_class_idx = np.zeros(lbl.shape[:2]) + 255   # per default non-important class, now fill up classes which are
                                                        # desired to be detected
        for i in range(len(s.palette)):
            lbl_class_idx[np.where(np.all(lbl == s.palette[i], axis=-1))] = i if i != 19 else 255
        lbl = torch.tensor(lbl_class_idx, dtype=torch.long)
        if self.debug:
            print("labels", lbl)
            print("len labels", lbl.size())
            print("type labels", lbl.dtype)
            print("original label type", lbl.dtype)

        if self.labels_only:
            return lbl
        else:
            return im, lbl

    def plot_instance(self, i, plot=True):
        image, label = self.__getitem__(i)
        image = image/2. + 0.5
        image = image.numpy().swapaxes(0, 2).swapaxes(0, 1)
        from matplotlib import cm
        cmap=cm.get_cmap('nipy_spectral', 27)
        if plot:
            plt.subplot(121)
            plt.imshow(image)
            plt.subplot(122)
            plt.imshow(label,vmin=0,vmax=27,cmap=cmap)
            plt.colorbar(ticks=range(27))
            plt.show()
        return image, label

class mixed_dataset(Dataset):
    def __init__(self, gta_path, city_path, scale, crop_size):
        super(mixed_dataset, self).__init__()
        self.gta_dataset = gta_dataset(gta_path, scale, crop_size)
        self.city_dataset = city_dataset(city_path, scale, crop_size)
        self.len = min([len(self.gta_dataset), len(self.city_dataset)])*2
    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if item >= self.len:
            raise IndexError("Index", item, "out of bounds.")
        if item % 2 == 0:
            return self.gta_dataset[int(item/2)]
        else:
            return self.city_dataset[int((item-1)/2)]

    def plot_instance(self, i, plot=True):
        image, label = self.__getitem__(i)
        image = image/2. + 0.5
        image = image.numpy().swapaxes(0, 2).swapaxes(0, 1)
        if plot:
            plt.subplot(211)
            plt.imshow(image)
            plt.subplot(212)
            plt.imshow(label)
            plt.show()
        return image, label

def main():
    # gta = gta_dataset("data")
    # gta[0]

    city = city_dataset(path="city_data_red", scale=0.5, debug=True)
    # need to test scale != 1 as well
    #city[np.random.randint(len(city))]
    city[np.random.randint(len(city))]

    #print(gta.path)
    #city.plot_instance(np.random.randint(2000))



if __name__ == '__main__':
    main()