import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import typing as ty
from torch import Tensor
import random
import numpy as np

def seed_everything(seed):     # set seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def save_mode_with_json(model,config : ty.Dict[int ,ty.List[str]], save_path : str) -> None:
    
    """
    Your Experiment save model and info in file.
    ex> output/ft-transformer/data_name/default/model_path   
        output/ft-transformer/data_name/default/info_dict
    - If you question or Error, leave an Issue.
    """

    with open(os.path.join(save_path,"info_dict" + str(config["count"]) + ".json"), "w", encoding = "utf-8") as make_file:
        json.dump(config, make_file, ensure_ascii = False, indent = "\t")

    torch.save(model.module.state_dict(), os.path.join(save_path, config["model"]) + "_" + str(config["count"]) + ".pt")

def save_model_info(info_data : ty.Dict, common : ty.List[int]) -> None:
    
    """
    save model info abouy trained model.
    json_file get model, epochs, valid_accuracy, task_name.. etc..
    """
    # assert path "It inapporate file_path. check file_path"
    with open("path", "w", encoding = "utf-8") as make_file:
        json.dump(info_data, make_file, ensure_ascii = False, indent = "\t")


def save_model(model, common : ty.List[int]) -> None:

    """
    save model input, model, parallel, parallel using input type
    """
    
    assert isinstance(common["parallel"], int) , "use Boolean in common[parallel]"
    
    if common["parallel"]:
        torch.save(model.module.sate_dict(), common["mode_save_path"])
    else:
        torch.save(model.state_dict(), common["model_save_path"])


def get_optimizer(model, config : ty.Dict[str, str]) -> optim:
    
    """
    rtdl using default optim AdamW
    if you want change, see run yaml
    """

    if config["optim"] == "AdamW":
        return torch.optim.Adam(
            model.parameters(),
            lr = float(config["lr"]),
            weight_decay = float(config["weight_decay"]),
            eps = 1e-8
        )
    elif config["optim"] == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr = float(config["lr"]),
            momentum=0.9,
            weight_decay = float(config["weight_decay"]),
        )
    else:
        pass

def get_loss(info_dict : ty.Dict[str, str]) -> Tensor:
    
    """
    The loss function used varies depending on the type of task.
    Binaryclass using binary_crossentropy
    but, multicass using cross_entropy
    """
    
    if info_dict["task_type"] == "binclass":
        return F.binary_cross_entropy_with_logits
    elif info_dict["task_type"] == "multiclass":
        return F.cross_entropy
    else:
        return F.mse_loss

