import os
import torch
import torch.nn as nn
import rtdl
import typing as ty
import yaml
import numpy as np
from dataset import load_dataset
from utils import * 
from model import load_model
from sklearn.model_selection import KFold, StratifiedKFold, 
import wandb

def model_train(args : ty.Any , config : ty.Union[ty.Dict, ty.List[str]] ,common : ty.Dict) -> None:

    """
    if you, chnage any thing, check main.py and run.yaml file.
    args get action and cuda using class,
    config config have model parameter (lr, weigt decay, epochs etc..)
    common common have path, choose model architecture.
    except args, there are made by run.yaml file.
    """

    common_path = os.listdir(common["path"])
    data_folders = os.listdir("data")
    
    for data_folder in data_folders:
        data_path = os.path.join(common_path, data_folders)
        train_dict, val_dict, test_dict, info_dict = load_dataset(data_path)
        info_dict["shape"] = np.shape(train_dict)[1]
        
        model = load_model(common, info_dict)
        optimizer = get_optimizer(model, config)
        loss_fn = get_loss(info_dict)

        if config["fold"] > 0:
            for epoch in range(config["epochs"]):
                pass



        model.train()
        
def model_run(model, optimizer, loss) -> ty.List[float]:
    
    pass