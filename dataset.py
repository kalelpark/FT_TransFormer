import os
import json
import torch
import numpy as np
import typing as ty
from torch import Tensor
from torch import FloatTensor
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler


class npy_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __getitem__(self, idx):
        x_data = torch.from_numpy(self.data)[idx]
        y_label = torch.from_numpy(self.label)[idx]
        
        return x_data, y_label
    
    def __len__(self):
        return len(self.data)

def get_DataLoader( train_data : ty.Dict[str, np.ndarray], 
                    valid_data : ty.Dict[str, np.ndarray],
                    test_data : ty.Dict[str, np.ndarray],
                    config):

    """
    train_data, valid_data,test_data you can change batch_size.
    checking about run.yaml file. config is default.
    - If you question or Error, leave an Issue.
    """ 

    train_dataset = npy_dataset(train_data["N_train"], train_data["y_train"])
    valid_dataset = npy_dataset(valid_data["N_val"], valid_data["y_val"])
    test_dataset = npy_dataset(test_data["N_test"], test_data["y_test"])
    
    train_dataloader = DataLoader(train_dataset, batch_size = int(config["batchsize"]), pin_memory = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = int(config["batchsize"]), pin_memory = True)
    test_dataloader = DataLoader(test_dataset, batch_size = int(config["batchsize"]), pin_memory = True)

    return train_dataloader, valid_dataloader, test_dataloader


def load_dataset(data_path : str) -> ty.List[ty.Dict[str, str]]:
    """
    load data and json info and return train_dict, val_dict, test_dict
    - If you question or Error, leave an Issue.
    """

    info_json = "info.json"
    N_train, y_train = "N_train.npy", "y_train.npy"
    N_test, y_test = "N_test.npy", "y_test.npy"
    N_val, y_val = "N_val.npy", "y_val.npy"

    json_path = os.path.join(data_path, info_json)
    
    with open(json_path, "r") as f:
        info_dict = json.load(f)

    train_dict = {}
    val_dict = {}
    test_dict = {}

    # load_dataset
    N_train_data, y_train_data = np.load(os.path.join(data_path, N_train)), np.load(os.path.join(data_path, y_train))    
    N_val_data, y_val_data = np.load(os.path.join(data_path, N_val)), np.load(os.path.join(data_path, y_val))
    N_test_data, y_test_data = np.load(os.path.join(data_path, N_test)), np.load(os.path.join(data_path, y_test))

    if info_dict["task_type"] != "regression":
        y_train_data = LabelEncoder().fit_transform(y_train_data).astype("int64")
        y_val_data = LabelEncoder().fit_transform(y_val_data).astype("int64")
        y_test_data = LabelEncoder().fit_transform(y_test_data).astype("int64")

    n_classes = int(max(y_train_data)) + 1 if info_dict["task_type"] == "multiclass" else None

    preprocess = StandardScaler().fit(N_train_data)
    train_dict["N_train"] = preprocess.transform(N_train_data)
    val_dict["N_val"] = preprocess.transform(N_val_data)
    test_dict["N_test"] = preprocess.transform(N_test_data)
    
    y_std = None

    if info_dict["task_type"] == "regression":
        y_mean = y_train_data.mean()
        y_std = y_train_data.std()
        y_train_data = (y_train_data - y_mean) / y_std
        y_val_data = (y_val_data - y_mean) / y_std
        y_test_data = (y_test_data - y_mean) / y_std

    if info_dict["task_type"] != 'multiclass':
        y_train_data = np.float64(y_train_data)
        y_val_data = np.float64(y_val_data)
        y_test_data = np.float64(y_test_data)

    train_dict["y_train"] = y_train_data
    val_dict["y_val"] = y_val_data
    test_dict["y_test"] = y_test_data

    d_out = n_classes or 1

    return train_dict, val_dict, test_dict, info_dict, d_out, y_std