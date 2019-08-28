import torch
import torch.nn as nn

class critic(nn.Module):
    def __init__(self,im_size):
        super(critic,self).__init__()

        self.cnn=nn.Sequential(convBlock(3,16),
                               convBlock(16,32),
                               convBlock(32,64),
                               convBlock(64,128))
        proc_im_size=im_size//(2**4)
        self.fc=nn.Linear(128*proc_im_size**2,1)
        self.sig=nn.Sigmoid()

    def forward(self,x):
        x=self.cnn(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        return self.sig(x)

class markov_critic(nn.Module):
    '''
    input: grayscale (first 3 channels) and colored (last 3 channels) image

    return: real/fake classification of "patches" of input as tensor
    (generator loss will be cronstructed form mean)
    '''
    def __init__(self):
        super(critic,self).__init__()
        
        self.cnn = nn.Sequential(convBlock(6,16,kernel_size=2),
                                convBlock(16,32,kernel_size=2),
                                convBlock(32,64,kernel_size=2),
                                convBlock(64,128,kernel_size=2),
                                convBlock(128,1,kernel_size=1))
        self.sig=nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        return self.sig(x)

class convBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=2,padding=1):
        super(convBlock,self).__init__()

        self.block=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding),
                                 nn.LeakyReLU(.2,True),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout2d(.25))
    def forward(self,x):
        return self.block(x)