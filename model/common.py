import enum
import math
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch import Tensor

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

def _make_nn_module(module_type : ModuleType, *args) -> nn.Module:  ## Active Function을 전달합니다.
    if isinstance(module_type, str):
        if module_type == 'ReGLU':
            return ReGLU()
        elif module_type == 'GeGLU':
            return GEGLU()
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(
                    f'Failed to construct the module {module_type} with the arguments {args}'
                ) from err
            return cls(*args)
    else:
        return module_type(*args)