import typing as ty
import numpy as np
from sklearn.metrics  import accuracy_score, mean_squared_error
import scipy
import torch
# torch.metrics
def get_accuracy_score(y_pred : ty.List[np.ndarray], y_label : ty.List[np.ndarray], dataset_info_dict) -> float:

    if dataset_info_dict["task_type"] == "multiclass":
        pass
    else:
        y_pred = np.round(scipy.special.expit(y_pred))

    return accuracy_score(y_pred, y_label)

def get_rmse_score(y_pred : ty.List[np.ndarray], y_label : ty.List[np.ndarray], y_std) -> float:
    return (mean_squared_error(y_pred, y_label) ** 0.5) * y_std
    