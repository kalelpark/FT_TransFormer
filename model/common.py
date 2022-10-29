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
                        kv_compression = None if int(config["kv_compression"]) == 0 else int(config["kv_compression"]),
                        kv_compression_sharing= None if int(config["kv_compression"]) == 0 else float(config["kv_compression"]),
                        d_out = int(info_dict["n_classes"]) if info_dict["task_type"] == "multiclass" else 1
        )

    elif config["model"] == "resnet":
        return ResNet(
                    d_numerical= int(info_dict["n_num_features"]),
                    categories = None,

                    # ModelA Architecture
                    activation = "relu",
                    d = int(config["d"]),
                    d_embedding = int(config["d_embedding"]),
                    d_hidden_factor = float(config["d_hidden_factor"]), 
                    hidden_dropout = float(config["hidden_dropout"]),
                    n_layers = int(config["n_layers"]),
                    normalization = config["normalization"],
                    residual_dropout = float(config["residual_dropout"]),

                    # default_Setting
                    d_out = int(info_dict["n_classes"]) if info_dict["task_type"] == "multiclass" else 1
        )
    else:
        pass
