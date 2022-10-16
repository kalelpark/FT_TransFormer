import os
import torch
import torch.nn as nn
from torch import LongTensor
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

def model_train(args : ty.Any, config: ty.Dict[str, ty.List[str]]) -> None:

    """
    args have device info (CPU, GPU, etc..), data, path [check main.py File].
    (Default Model using torcn.nn.parallel.) 
    config have model parameters and model info. check model.yaml.  
    - If you question or Error, leave an Issue.
    """

    wandb.init( name = config["model"], 
                project = config["model"] + '_' + args.data)
    wandb.config = config
    
    train_dict, val_dict, test_dict, info_dict = load_dataset(args.data_path)
    print("loaded Dataset..")

    model = load_model(config, info_dict)           # Update!! 
    print("loaded Model..")
      
    optimizer = (get_optimizer(model, config))
    loss_fn = (get_loss(info_dict))
    loss_fn.to(args.device)

    print("loaded optimizer and loss..")

        if int(config["fold"]) > 0:
            print("Fold Training..")
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
            print("Single[default] Training..")
            train_dataloader, valid_dataloader = get_DataLoader(train_dict, val_dict, config)
            model_run(model, optimizer, loss_fn, data_folder, train_dataloader, valid_dataloader, info_dict, args, config)
                
def model_run(  model, optimizer, 
                loss_fn, data_name : str,
                train_dataloader, valid_dataloader, info_dict,
                 args : ty.Any, config : ty.Dict[str, ty.List[str]]) ->ty.List[float]:

    """
    if you change any options, you will see model_train function, and run.yaml File
    this function only using things to learn the model.
    and make json_file to see info trained_model score. score get train, valid accuracy
    - If you question or Error, leave an Issue.
    """
    
    json_info = OrderedDict()
    model.to(args.device)
    # wandb.watch(model)
    method =  "ensemble" if config["fold"] > 0 else "default"
    info_save_path = os.path.join(config["output_path"], config["model"], data_name, method)
    print("start_training..")
    loss_fn
    for epoch in range(config["epochs"]):
        train_loss_score, valid_loss_score = 0, 0
        if info_dict["task_type"] == "regression":
            best_valid = 1e10
        else:
            best_valid = 0
        
        train_pred, valid_pred = list(), list()
        train_label, valid_label = list(), list()
        
        for X_data, y_label in train_dataloader:
            model.train()
            optimizer.zero_grad()
            X_data, y_label = X_data.to(args.device), y_label.to(args.device)
            y_pred = model(X_data, x_cat=None)
            loss = loss_fn(y_pred, y_label)
            loss.backward()
            optimizer.step()
            train_loss_score += loss.item() 

            train_pred.extend(y_pred)
            train_label.extend(y_label)
        
        if info_dict["task_type"] == "regression":
            train_score = get_rmse_score(train_pred, train_label)
        else:
            train_score = get_accuracy_score(train_pred, train_pred)
        
        model.eval()
        for X_data, y_label in valid_dataloader:
            X_data, y_label = X_data.to(args.device), y_label.to(args.device)
            y_pred = model(X_data)

            valid_pred.extend(y_pred)
            valid_label.extend(y_label)
        
        if info_dict["task_type"] == "regression":
            valid_score = get_rmse_score(valid_pred, valid_label)
        else:
            valid_score = get_accuracy_score(valid_pred, valid_label)
        

        if info_dict["task_type"] == "regression":
            if best_valid > valid_score:
                best_valid = valid_score

                json_info["model"] = config["model"]
                json_info["epochs"] = config["epochs"]
                json_info["valid_accuracy"] = valid_score
                json_info["task_name"] = config["task_name"]
                save_mode_with_json(model, json_info,config, info_save_path)
        
        else:
            if best_valid < valid_score:
                best_valid = valid_score
                
                json_info["model"] = config["model"]
                json_info["epochs"] = config["epochs"]
                json_info["valid_accuracy"] = valid_score
                json_info["task_name"] = config["task_name"]
                save_mode_with_json(model, json_info,config, info_save_path)

        wandb.log({
            "train_score" : train_score,
            "valid_score" : valid_score,
            "train_loss" : train_loss_score
        })