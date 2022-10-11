import os
import torch
import torch.nn as nn
import rtdl
import typing as ty
from dataset import load_dataset

def model_train(config, args, common) -> None:
    common_path = os.listdir(common["path"])
    data_folders = os.listdir("data")

    for data_folder in data_folders:
        data_path = os.path.join(common_path, data_folders)
        train_dict, val_dict, test_dict, info_dict = load_dataset(data_path)