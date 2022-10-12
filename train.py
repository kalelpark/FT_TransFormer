import os
import torch
import torch.nn as nn
import rtdl
import typing as ty
import yaml
import numpy as np
from dataset import *
from utils import *
from metrics import *
from model import load_model
from sklearn.model_selection import KFold, StratifiedKFold
import wandb
from collections import OrderedDict

def model_train(args : ty.Any , config : ty.Union[ty.Dict, ty.List[str]] ,common : ty.Dict[str, ty.Union(str, int)]) -> None:

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
        print("loaded Model..")
        
        optimizer = get_optimizer(model, config)
        loss_fn = get_loss(info_dict)

        if config["fold"] > 0:
            """
            Kfold training
            """
            kf = KFold(get_splits = 15)
            for idxx, (train_idx, temp_idx) in enumerate(kf.split(train_dict["N_train"], train_dict["y_train"])):  # X_train, X_temp, y_train, y_temp = 
                fold_train_dict = {}
                fold_train_dict["N_train"] = train_dict["N_train"][train_idx]
                fold_train_dict["y_train"] = train_dict["y_train"][train_idx]
                train_dataloader, valid_dataloader = get_DataLoader(fold_train_dict, val_dict)
        else:
            """
            Default : Singe Training
            """
            train_dataloader, valid_dataloader = get_DataLoader(train_dict, val_dict)

                
def model_run(model, optimizer, loss_fn, train_dataloader, valid_dataloader, common, args) -> ty.List[float]:
    json_info = OrderedDict()

    for epoch in range(common["epochs"]):
        train_loss_score, valid_loss_score = 0, 0
        best_valid = 0
        train_prediction, valid_prediction = list(), list()

        for X_data, y_label in train_dataloader:
            model.train()
            optimizer.zero_grad()
            X_data, y_label = X_data.to(args.device), y_label.to(args.device)
            y_pred = model(X_data)
            loss = loss_fn(y_pred, y_label)
            loss.backward()
            optimizer.step()
            train_loss_score += loss.item()
            train_prediction.extend(y_pred)
        train_accuracy_score = get_accuracy_score(y_pred, y_label)
        
        model.eval()
        valid_prediction = list()
        for X_data, y_label in valid_dataloader:
            X_data, y_label = X_data.to(args.device), y_label.to(args.device)
            y_pred = model(X_data)
            valid_prediction.extend(y_pred)
        
        valid_accuracy_score = get_accuracy_score(y_pred, y_label)

        if best_valid < valid_accuracy_score:
            best_valid = valid_accuracy_score
            save_model(model, common)
            
            json_info["model"] = common["model"]
            json_info["epochs"] = common["epochs"]
            json_info["valid_accuracy"] = valid_accuracy_score
            json_info["task_name"] = common["task_name"]
            

            

            





    
    pass