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

def model_train(args : ty.Any, config : ty.Dict[str, ty.Union(str, int, float)]) -> None:

    """
    args have device info (CPU, GPU, etc..) if you modfiy info, check main.py
    config have train & test info (lr, optim, model) in the form Dictionary.
    check run.yaml File.  - If you question or Error, leave an Issue.
    """

    data_folders = os.listdir(config["data_path"])    
    for data_folder in data_folders:
        data_path = os.path.join(config["data_path"], data_folder)
        train_dict, val_dict, test_dict, info_dict = load_dataset(data_path)
        print("looaded Dataset..")

        model = load_model(config, info_dict)

        print("loaded Model..")
        
        optimizer = get_optimizer(model, config)
        loss_fn = get_loss(info_dict)
        print("loaded optimizer and loss_fn..")

        if config["fold"] > 0:
            print("KFOLD Training..")
            kf = KFold(get_splits = 15)
            for idxx, (train_idx, temp_idx) in enumerate(kf.split(train_dict["N_train"], train_dict["y_train"])):  # X_train, X_temp, y_train, y_temp = 
                fold_train_dict = {}
                fold_train_dict["N_train"] = train_dict["N_train"][train_idx]
                fold_train_dict["y_train"] = train_dict["y_train"][train_idx]
                train_dataloader, valid_dataloader = get_DataLoader(fold_train_dict, val_dict, config)
        else:
            """
            Default : Singe Training
            """
            train_dataloader, valid_dataloader = get_DataLoader(train_dict, val_dict, config)
            model_run(model, optimizer, loss_fn, train_dataloader, valid_dataloader, args, config)
                
def model_run(  model : torch.Module , optimizer : torch.optim, loss_fn : torch.functional, 
                train_dataloader : DataLoader, valid_dataloader : DataLoader, args : ty.Any, 
                config : ty.Dict[str, ty.Union(str, int)]) -> ty.List[float]:

    """
    if you change any options, you will see model_train function, and run.yaml File
    this function only using things to learn the model.
    and make json_file to see info trained_model score. score get train, valid accuracy
    - If you question or Error, leave an Issue.
    """

    json_info = OrderedDict()
    model.to(args.device)
    for epoch in range(config["epochs"]):
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
            save_model(model, config)
            
            json_info["model"] = config["model"]
            json_info["epochs"] = config["epochs"]
            json_info["valid_accuracy"] = valid_accuracy_score
            json_info["task_name"] = config["task_name"]
    
    pass