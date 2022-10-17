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

