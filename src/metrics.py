from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true, y_pred) -> float:
    # compatible with sklearn versions where squared= may not exist
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_balanced_accuracy(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))
