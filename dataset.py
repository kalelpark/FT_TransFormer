import os
import json
import numpy as np
import typing as ty
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

class npy_dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __getitem__(self, idx):
        x_data = Tensor(self.data[idx])
        y_label = Tensor(self.label[idx])

        return x_data, y_label
    
    def __len__(self):
        return len(self.data)

def get_DataLoader(train_data : ty.Dict[np.ndarray], valid_data : ty.Dict[np.ndarray]) -> DataLoader:
    """
    Making CustomDataLoader
    """ 
    train_dataset = npy_dataset(train_data["N_train"], train_data["y_train"])
    valid_dataset = npy_dataset(valid_data["N_val"], valid_data["y_val"])
    
    train_dataloader, valid_dataloader = DataLoader(train_dataset, batch_size = 32), DataLoader(valid_dataset, batch_size = 32)
    
    return train_dataloader, valid_dataloader



def load_dataset(data_path : ty.Optional[str]) -> ty.Dict[str, ty.Union(np.ndarray, str)]:
    """
    load data and json info
    return train_dict, val_dict, test_dict
    """
    info_json = "info.json"
    N_train, y_train = "N_train.npy", "y_train.npy"
    N_test, y_test = "N_test.npy", "y_test.npy"
    N_val, y_val = "N_val.npy", "y_val.npy"
    json_path = os.path.join(data_path, info_json)
    info_dict = {}
    
    with open(json_path, "r") as f:
        json_dict = json.load(f)
        json_data = json.dumps(json_dict)
        info_dict["task_type"] = json_data["task_type"] 
        info_dict["n_classes"] = json_data["n_classes"]

    train_dict = {}
    val_dict = {}
    test_dict = {}

    N_train_data, y_train_data = np.load(os.path.join(data_path, N_train)), np.load(os.path.join(data_path, y_train))
    train_dict["N_train"] = N_train_data
    train_dict["y_train"] = y_train_data

    N_val_data, y_val_data = np.load(os.path.join(data_path, N_val)), np.load(os.path.join(data_path, y_val))
    val_dict["N_val"] = N_val_data
    val_dict["y_val"] = y_val_data

    N_test_data, y_test_data = np.load(os.path.join(data_path, N_test)), np.load(os.path.join(data_path, y_test))
    test_dict["N_test"] = N_test_data
    test_dict["y_test"] = y_test_data

    return train_dict, val_dict, test_dict, info_dict
    

