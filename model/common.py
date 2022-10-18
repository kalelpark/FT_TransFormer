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
from fttransformer import FTTransformer
from resnet import ResNet

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
    # 저장한 정보도 반환하게 끔 만들기
    if config["model"] == "ft-transformer":
        return FTTransformer.make_baseline(
                            # Example
                            n_num_features= int(info_dict["n_num_features"]),
                            cat_cardinalities= None,
                            d_token = int(config["d_token"]),
                            n_blocks = int(config["n_blocks"]),
                            attention_dropout = float(config["attention_dropout"]),
                            ffn_d_hidden = int(config["ffn_d_hidden"]),
                            ffn_dropout = float(config["ffn_d_dropout"]),
                            residual_dropout= float(config["residual_dropout"]),
                            d_out= 1 if info_dict["task_type"] == "regression" else int(info_dict["n_classes"]),
                        )
    elif config["resnet"] == "resnet":
        return ResNet.make_baseline(
                        d_in = int(info_dict["n_num_features"]),
                        d_main = int(config["d_main"]),
                        d_hidden = int(config["d_hidden"]),
                        dropout_first = float(config["d_hidden"]),
                        dropout_second = float(config["dropout_second"]),
                        n_blocks = int(config["n_blocks"]),
                        d_out= 1 if info_dict["task_type"] == "regression" else int(info_dict["n_classes"]),
                    )
    else:
        pass
