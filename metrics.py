import typing as ty
import numpy as np
from sklearn.metrics  import accuracy_score, mean_squared_error

def get_accuracy_score(y_pred : ty.List[np.ndarray], y_label : ty.List[np.ndarray]) -> float:
    return accuracy_score(y_pred, y_label)

def get_rmse_score(y_pred : ty.List[np.ndarray], y_label : ty.List[np.ndarray]) -> float:
    return mean_squared_error(y_pred, y_label) ** 0.5
    