import os
from torch import Tensor 
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
import json
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import typing as ty
import torch.nn as nn
import math
import torch.nn.init as nn_init
import torch
import torch.nn.functional as F

with open(f"yaml/aloi.yaml") as f:
    config = yaml.load(f, Loader = yaml.FullLoader)["fttransformer"]

print(config)