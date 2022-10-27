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
from model import common
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

    train_dict, val_dict, test_dict, dataset_info_dict, d_out = load_dataset(args.data_path)
    print("loaded Dataset..")

    model = common.load_model(config, dataset_info_dict)           
    print("loaded Model..")

    wandb.init( name = config["model"] + "_" + str(config["count"]), 
                project = config["model"] + '_' + args.data)
    wandb.config = config
      
    optimizer = get_optimizer(model, config)
    loss_fn = get_loss(dataset_info_dict)
    print("loaded optimizer and loss..")

    if int(config["fold"]) > 2:     # fold
        print("Fold Training..")
        kf = KFold(get_splits = 15)
        for idxx, (train_idx, temp_idx) in enumerate(kf.split(train_dict["N_train"], train_dict["y_train"])):  # X_train, X_temp, y_train, y_temp = 
            fold_train_dict = {}
            fold_train_dict["N_train"] = train_dict["N_train"][train_idx]
            fold_train_dict["y_train"] = train_dict["y_train"][train_idx]
            train_dataloader, valid_dataloader, test_dataloader = get_DataLoader(fold_train_dict, val_dict, test_dict, config)

    else:   # Default 
        print("Single[default] Training..")
        train_dataloader, valid_dataloader, test_dataloader = get_DataLoader(train_dict, val_dict, test_dict, config)
        model_run(model, optimizer, loss_fn, train_dataloader, valid_dataloader, test_dataloader, dataset_info_dict, args, config)

# 
def model_run(model, optimizer, loss_fn, train_dataloader, valid_dataloader, test_dataloader, dataset_info_dict, args, config):
    model.to(args.device)
    model = nn.DataParallel(model)
    method = "ensemble" if config["fold"] > 0 else "default"
    json_info_output_path = os.path.join(str(args.savepath), config["model"], str(args.data), method)
    print("Ready to run model...")

    # Best Score
    if dataset_info_dict["task_type"] == "regression":  # RMSE
        best_valid = 1e10
    else:       # Accuracy
        best_valid = 0

    for epoch in range(int(config["epochs"])):
        train_loss_score, valid_loss_score = 0, 0

        train_pred, valid_pred, test_pred = np.array([]), np.array([]), np.array([])
        train_label, valid_label, test_label = np.array([]), np.array([]), np.array([])

        model.train()       # train about ResNet
        for X_data, y_label in train_dataloader:        # Train
            optimizer.zero_grad()

            X_data, y_label = X_data.to(args.device), y_label.to(args.device)
            y_pred = model(X_data) if config["model"] == "!!" else model(x_num = X_data, x_cat = None)
            loss = loss_fn(y_pred.to(torch.float64).squeeze(1), y_label.to(torch.float64))
            
            loss.backward()
            optimizer.step()
            train_loss_score += loss.item()
            
            train_pred = np.append(train_pred, y_pred.cpu().detach().numpy())
            train_label = np.append(train_label, y_label.cpu().detach().numpy())
        
        model.eval()        # Valid about ResNet
        for X_data, y_label in valid_dataloader:        # Valid
            
            X_data, y_label = X_data.to(args.device), y_label.to(args.device) 
            y_pred = model(X_data) if config["model"] == "!!" else model(x_num = X_data, x_cat = None)

            valid_pred = np.append(valid_pred, y_pred.cpu().detach().numpy())
            valid_label = np.append(valid_label, y_label.cpu().detach().numpy())

        model.eval()
        for X_data, y_label in test_dataloader:         # Test

            X_data, y_label = X_data.to(args.device), y_label.to(args.device)
            y_pred = model(X_data) if config["model"] == "!!" else model(x_num = X_data, x_cat = None)
            
            test_pred = np.append(test_pred, y_pred.cpu().detach().numpy())
            test_label = np.append(test_label, y_label.cpu().detach().numpy())            
        
        if dataset_info_dict["task_type"] == "regression":
            train_score, valid_score = get_rmse_score(train_pred, train_label), get_rmse_score(valid_pred, valid_label)

            if best_valid > valid_score:
                best_valid = valid_score
                test_score = get_rmse_score(test_pred, test_label)
                config["valid_rmse"] = valid_score
                config["test_rmse"] = test_score
                save_mode_with_json(model, config, json_info_output_path)
        else:
            train_score, valid_score = get_accuracy_score(train_pred, train_pred), get_accuracy_score(valid_pred, valid_label)

            if best_valid < valid_score:
                best_valid = valid_score
                test_score = get_accuracy_score(test_pred, test_label)

                config["valid_accuracy"] = valid_score
                config["test_accuracy"] = test_score
                save_mode_with_json(model, config, json_info_output_path)
        
        wandb.log({
            "train_score" : train_score,
            "train_loss" : train_loss_score,
            "valid_score" : valid_score,
            "test_score" : test_score
        })
