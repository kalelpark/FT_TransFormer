import enum
import math
import time
from copy import deepcopy
import warnings
import typing as ty
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from .fttransformer import Transformer
from .resnet import ResNet

ModuleType = Union[str, Callable[..., nn.Module]]

def reglu(x : Tensor) -> Tensor:
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim = -1)
    return a * F.relu(b)

def geglu(x : Tensor) -> Tensor:
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim = -1)
    return a * F.gelu(b)

class ReGLU(nn.Module):
    def forward(self, x : Tensor) -> Tensor:
        return reglu(x)

class GEGLU(nn.Module):
    def forward(self, x : Tensor) -> Tensor:
        return geglu(x)

def load_model(config: ty.Dict , info_dict : ty.Dict):
    if config["model"] == "ft-transformer":
        return Transformer(
                        d_numerical = int(info_dict["n_num_features"]),
                        categories = None,

                        # Model Architecture
                        n_layers = int(config["n_layers"]),
                        n_heads = int(config["n_heads"]),
                        d_token = int(config["d_token"]),
                        d_ffn_factor = float(config["d_ffn_factor"]),
                        attention_dropout = float(config["attention_dropout"]),
                        ffn_dropout = float(config["attention_dropout"]),
                        residual_dropout = float(config["residual_dropout"]),
                        activation = config["activation"],
                        prenormalization = True,
                        initialization = config["initialization"],
                        
                        # default_Setting
                        token_bias = True,
                        kv_compression = None,
                        kv_compression_sharing= None,
                        d_out = 1 if info_dict["task_type"] == "regression" else int(info_dict["n_classes"]) if info_dict["task_type"] == "multiclass" else 2
        )

    elif config["model"] == "resnet":
        return ResNet(
                    d_numerical= int(info_dict["n_num_features"]),
                    categories = None,

                    # ModelA Architecture
                    activation = "relu",
                    d = int(info_dict["d"]),
                    d_embedding = int(info_dict["d_embedding"]),
                    d_hidden_factor = float(info_dict["d_hidden_factor"]), 
                    hidden_dropout = float(info_dict["hidden_dropout"]),
                    n_layers = int(info_dict["n_layers"]),
                    normalization = info_dict["batchnorm"],
                    residual_dropout = float(info_dict["residual_dropout"]),
                    
                    # default_Setting
                    d_out = 1 if info_dict["task_type"] == "regression" else int(info_dict["n_classes"]) if info_dict["task_type"] == "multiclass" else 2
        )
    else:
        pass
