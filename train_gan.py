import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from itertools import count
import sys, getopt
from models.discriminator import critic
from models.model import model
from models.unet import unet
from settings import s
import time
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import dataloader
import json

def main(argv):
    # setting argument defaults
    mbsize = s.batch_size
    data_path = s.data_path
    report_freq=s.report_freq
    weight_path=s.weights_path
    weights_name=s.weights_name
    lr=s.learning_rate
    loss_norm = s.loss_norm
    batch_norm = s.batch_norm
    save_freq = s.save_freq
    mode=0
    time_namer = time.strftime("%y%m%d%H%M%S")
    load_specific = True
    parent_name = None
    image_loss_weight=s.image_loss_weight
    epochs = s.epochs
    path_gta = path_city = None
    help='test.py -b <int> -p <string> -r <int> -w <string>'
    dataset_type = s.dataset_type
    try:
        opts, args = getopt.getopt(argv,"h:e:b:p:r:w:l:s:t:n:m:",["mbsize=","data-path=","report-freq=",'weight-path=','parent-name=', 'lr=', 'loss-norm=', 'batch-norm=', 'epochs=','save-freq=','timer-name=', 'dataset-type=', 'datapath-gta=', 'datapath-city='])
    except getopt.GetoptError:
        print(help)
        sys.exit(2)
    print("opts" ,opts)
    for opt, arg in opts:
        if opt == '-h':
            print(help)
            sys.exit()
        elif opt in ("-b", "--mbsize"):
            mbsize = int(arg)
        elif opt in ("-p", "--data-path"):
            data_path = arg
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ('-r','--report-freq'):
            report_freq = int(arg)
        elif opt in ("-w", "--weight-path"):
            weight_path = arg
        elif opt in ("-n", "--weight-name"):
            weights_name = arg            
        elif opt in ("-s", "--save-freq"):
            save_freq=int(arg)
        elif opt in ("-t", "--timer-name"):
            time_namer=arg
            load_specific = True
        elif opt in ("-l", "--lr"):
            lr = float(arg)
        elif opt in ("--loss-norm"):
            load_specific=arg in ["true", "True", "1"]
        elif opt in ("--batch-norm"):
            batch_norm = arg in ["true", "True", "1"]
        elif opt == "--parent-name":
            parent_name = arg
        elif opt == "--dataset-type":
            dataset_type = arg  # supported values: "gta", "city", "mixed"
        elif opt == "--datapath-gta":
            path_gta = arg
        elif opt == "--datapath-city":
            path_city = arg
        elif opt=='-m':
            mode = arg in ('u','1')

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_shape=(3,32,32)
    out_shape=(s.classes,32,32)

    weight_path_ending=os.path.join(weight_path,weights_name+'.pth')
    print("NETWORK PATH:", weight_path_ending)

    loss_path_ending = os.path.join(weight_path, time_namer + "_" + s.loss_name)
    model_description_path_ending = os.path.join(weight_path,s.model_description_name)
    
    trainset = datasets.CIFAR10(root='./cifar-10', train=True,
                                        download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=mbsize,
                                        shuffle=True, num_workers=2)
   
    print("NETWORK PATH:", weight_path_ending)
    '''
    if parent_name is None:
        parent_path_ending = None
    else:
        if parent_name == "latest":
            i = -1
            while not s.weights_name in parent_name:  # filtering out logs
                parent_name = sorted(os.listdir(weight_path))[i]
                i -= 1
        parent_path_ending=os.path.join(weight_path, parent_name)  # assuming parent and children weights to be in same directory
    loss_path_ending = os.path.join(weight_path, time_namer + "_" + s.loss_name)
    model_description_path_ending = os.path.join(weight_path,s.model_description_name)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10(root='./cifar-10', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=mbsize,
                                          shuffle=True, num_workers=2)
    '''
    #print("dataset_type",dataset_type)
    #dataloader
    #trainloader=torch.utils.data.DataLoader(dataset,batch_size=mbsize,shuffle=True, num_workers=1)

    #define model
    UNet=model().to(device) if mode==0 else unet().to(device)
    #load weights
    if load_specific:
        try:
            UNet.load_state_dict(torch.load(weight_path_ending))
            print("Loaded weights from", weight_path_ending)
        except FileNotFoundError:
            print("Unable to find weight path "+ weight_path_ending + ". Initializing new weights.")
            # weights already initialized in unet.__init__()
    else:
        print("No parent network specified. Initialized new weights.")
        #TODO: fix this 
        '''
        # change model overview list:
        if not time_namer in model_dict.keys():
            model_dict[time_namer] = {
                "data_set_path": data_path,
                "loaded_weights": weights_name,
                "epochs": e,
                "batch_size": mbsize,
                "scale": s.scale,
                "crop": s.crop_size,
                "lr": lr,
                "betas": s.betas
            }
        else:
            model_dict[time_namer]["epochs"] = e  # only epochs change when model is already mentioned in dict
        with open(model_description_path_ending, "w") as file:
            json.dump(model_dict, file, sort_keys=True, indent=4)'''


    UNet.train()
    #fix resnet layers

    resnet_layers=torch.load('models/resnet_weight_names.pt')
    for name,param in UNet.state_dict().items():
        if name in resnet_layers:
            param.requires_grad=False
    #define critic as custom loss function for unet
    crit=critic(trainset.data.shape[1]).to(device)
    #load discriminator weights
    crit_path=weight_path+'/'+weights_name+'_crit.pth'
    if load_specific:
        try:
            crit.load_state_dict(torch.load(crit_path))
            print('Loaded weights for discriminator from %s'%crit_path)
        except FileNotFoundError:
            print('Initialize new weights for discriminator')
            crit.apply(weights_init_normal)
    #optimizer
    optimizer_g=optim.Adam(filter(lambda p: p.requires_grad, UNet.parameters()),lr=lr,betas=s.betas)
    optimizer_c=optim.Adam(crit.parameters(),lr=lr,betas=s.betas)
    criterion = nn.BCELoss().to(device)
    #additional gan loss: l1 loss
    l1loss = nn.L1Loss().to(device)
    loss_hist=[]

    # initialize model dict
    try:
        with open(model_description_path_ending, "r") as file:
            model_dict = json.load(file)
    except FileNotFoundError:
        model_dict = {}

    #convert to black and white image using following weights
    gray = torch.tensor([0.2989 ,0.5870, 0.1140 ])[:,None,None].float()
    ones=torch.ones(mbsize,device=device)
    zeros=torch.zeros(mbsize,device=device)
    # run over epochs
    for e in (range(epochs) if not s.infinite_loop else count()):
        g_running,c_running=0,0
        #load batches
        for i,(image,_) in enumerate(trainloader):
            batch_size=image.shape[0]
            #create ones and zeros tensors
            
            #convert to grayscale image
            
            #using the matlab formula: 0.2989 * R + 0.5870 * G + 0.1140 * B and load data to gpu
            X=(image.clone()*gray).sum(1).to(device).view(-1,1,*in_shape[1:])
            image=image.float().to(device)
            #----------------------------------------------------------------------------------------
            ################################### Unet optimization ###################################
            #----------------------------------------------------------------------------------------
            #clear gradients
            optimizer_g.zero_grad()
            #generate colorized version with unet
            unet_col=None
            if mode==0:
                unet_col=UNet(torch.stack((X,X,X),1)[:,:,0,:,:])
            else:
                unet_col=UNet(X)
            #calculate loss as a function of how good the unet can fool the critic
            fooling_loss=criterion(crit(unet_col)[:,0], ones[:batch_size])
            #calculate how close the generated pictures are to the ground truth
            image_loss=l1loss(unet_col,image)
            #combine both losses and weight them
            loss_g=fooling_loss+image_loss_weight*image_loss
            #backpropagation
            loss_g.backward()
            optimizer_g.step()

            #----------------------------------------------------------------------------------------
            ################################## Critic optimization ##################################
            #----------------------------------------------------------------------------------------
            optimizer_c.zero_grad()
            real_loss=criterion(crit(image)[:,0],ones[:batch_size])
            #requires no gradient in unet col
            fake_loss=criterion(crit(unet_col.detach())[:,0],zeros[:batch_size])
            loss_c=.5*(real_loss+fake_loss)
            loss_c.backward()
            optimizer_c.step()

            g_running+=loss_g.item()
            c_running+=loss_c.item()
            loss_hist.append([e,i,loss_g.item(),loss_c.item()])

            #report loss
            if (i+len(trainloader)*e)%report_freq==report_freq-1:
                print('Epoch %i, batch %i: \tunet loss=%.2e, \tcritic loss=%.2e'%(e+1,i+1,g_running/report_freq,c_running/report_freq))
                g_running=0
                c_running=0
            if s.save_weights and (i+len(trainloader)*e)%save_freq==save_freq-1:
                #save parameters
                try:
                    torch.save(UNet.state_dict(),weight_path_ending)
                    torch.save(crit.state_dict(),crit_path)
                except FileNotFoundError:
                    os.makedirs(weight_path)
                    torch.save(UNet.state_dict(),weight_path_ending)
                    torch.save(crit.state_dict(),crit_path)
                print("saved parameters")
                if s.save_loss:
                    #save loss history to file
                    try:
                        f=open(loss_path_ending,'a')
                        np.savetxt(f,loss_hist,'%e')
                        f.close()
                    except FileNotFoundError:
                        os.makedirs(s.loss_path)
                        np.savetxt(loss_path_ending,loss_hist,'%e')
                    loss_hist=[]
            

        

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

if __name__ == '__main__':
    main(sys.argv[1:])