
from collections import namedtuple
settings = {
    'epochs': 20,
    'infinite_loop':True,
    'classes': 3,

    # Hyperparameters
    'learning_rate':.001,
    'batch_size':64,
    'betas':(0.9, 0.999),
    'loss_norm': False,
    'batch_norm': True,
    'image_loss_weight':50,

    # Other
    'save_freq': 500,
    'save_weights':True,
    'weights_path':'weights',
    'weights_name':'unet',
    'save_loss':True,
    'loss_path':'weights',
    'loss_name':'loss.txt',
    'model_description_name': 'model_description.json',
    # data paht
    'data_path':'places-test/',
    # report loss every x batch
    'report_freq':10,

    
}
s = namedtuple("Settings", settings.keys())(*settings.values())
