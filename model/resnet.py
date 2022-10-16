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
from common import *

ModuleType = Union[str, Callable[..., nn.Module]]

class ResNet(nn.Module):

    """
    The ResNet Model composed FCN and Blocks.

    ResNet: (in) -> Linear -> Block -> ... -> Block -> Head -> (out)

        |-> Norm -> Linear -> Activation -> Dropout -> Linear -> Dropout ->|
        |                                                                  |
    Block: (in) --------------------------------------------------------> Add -> (out)

    Head: (in) -> Norm -> Activation -> Linear -> (out)
    """

    """
    Examples of Test CODE:
        >> Test CODE
            x = torch.randn(4, 2)
            module = ResNet.make_baseline(
                d_in=x.shape[1],
                n_blocks=2,
                d_main=3,
                d_hidden=4,
                dropout_first=0.25,
                dropout_second=0.0,
                d_out=1
            )

            print(module(x))
    """

    class Block(nn.Module):
        def __init__(
            self,
            *,
            d_main : int,
            d_hidden : int,
            bias_first : bool,
            bias_second : bool,
            dropout_first : float,
            dropout_second : float,
            normalization : ModuleType,
            activation : ModuleType,
            skip_connection : bool
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_main)
            self.linear_first = nn.Linear(d_main, d_hidden, bias_first)
            self.activation = _make_nn_module(activation)
            self.dropout_first = nn.Dropout(dropout_first)
            self.linear_second = nn.Linear(d_hidden, d_main, bias_second)
            self.dropout_second = nn.Dropout(dropout_second)
            self.skip_connection = skip_connection

        def forward(self, x : Tensor) -> Tensor:
            x_input = x
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout_first(x)
            x = self.linear_second(x)
            x = self.dropout_second(x)
            
            if self.skip_connection:
                x = x_input + x

            return x

    class Head(nn.Module):
        def __init__(
            self,
            *,
            d_in : int,
            d_out : int,
            bias : bool,
            normalization : ModuleType,
            activation : ModuleType
        ) -> None:
            super().__init__()
            self.normalization = _make_nn_module(normalization, d_in)
            self.activation = _make_nn_module(activation)
            self.linear = nn.Linear(d_in, d_out, bias)
        
        def forward(self, x : Tensor) -> Tensor:
            if self.normalization is not None:
                x = self.normalization(x)
            x = self.activation(x)
            x = self.linear(x)

            return x
        
    ## ResNet __init__
    def __init__(
        self,
        *,
        d_in : int,
        n_blocks : int,
        d_main : int,
        d_hidden : int,
        dropout_first : float,
        dropout_second : float,
        normalization : ModuleType,
        activation : ModuleType,
        d_out : int) -> None:

        super().__init__()
        self.first_layer = nn.Linear(d_in, d_main)

        if d_main is None:
            d_main = d_in
        self.blocks = nn.Sequential(
            *[
                ResNet.Block(
                    d_main = d_main,
                    d_hidden = d_hidden,
                    bias_first = True,
                    bias_second = True,
                    dropout_first = dropout_first,
                    dropout_second = dropout_second,
                    activation = activation,
                    skip_connection = True,
                )
                for _ in range(n_blocks)
            ]
        )

        self.head = ResNet.head(
            d_in = d_main,
            d_out = d_out,
            bias = True,
            normalization = normalization,
            activation = activation
        )
            
    @classmethod
    def make_baseline(
        cls : Type['ResNet'],
        *,
        d_in : int,
        n_blocks : int,
        d_main : int,
        d_hidden : int,
        dropout_first : float,
        dropout_second : float,
        d_out : int
    ) -> 'ResNet':

        return cls(
            d_in = d_in,
            n_blocks = n_blocks,
            d_main = d_main,
            d_hidden = d_hidden,
            dropout_first = dropout_first,
            dropout_second = dropout_second,
            normalization = 'BatchNorm1d',
            activation = 'ReLU',
            d_out = d_out,
        )
    
    def forward(self, x : Tensor) -> Tensor:
        x = self.first_layer(x)
        x = self.blocks(x)
        x = self.head(x)

        return x