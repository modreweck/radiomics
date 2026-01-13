from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    cohen_kappa_score,
)


def rmse(y_true, y_pred) -> float:
    """RMSE compatible with sklearn versions where squared= may not exist."""
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Standard regression metrics used in the protocol."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_balanced_accuracy(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def classification_weighted_kappa_quadratic(y_true, y_pred) -> float:
    """Quadratic weighted Cohen's kappa (ordinal agreement)."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def mae_days_from_groups(
    y_true_group,
    y_pred_group,
    group_to_days: dict[str, int] | None = None,
) -> float:
    """
    Convert group labels to days and compute MAE in days.
    """
    if group_to_days is None:
        group_to_days = {"GC": 0, "G2": 2, "G5": 5, "G7": 7, "G14": 14}

    y_true_days = pd.Series(y_true_group).map(group_to_days).astype(float).to_numpy()
    y_pred_days = pd.Series(y_pred_group).map(group_to_days).astype(float).to_numpy()

    return float(mean_absolute_error(y_true_days, y_pred_days))


def classification_metrics_multiclass(
    y_true,
    y_pred,
    group_to_days: dict[str, int] | None = None,
) -> dict[str, float]:
    """Standard classification metrics used in the protocol."""
    return {
        "balanced_accuracy": classification_balanced_accuracy(y_true, y_pred),
        "weighted_kappa_quadratic": classification_weighted_kappa_quadratic(y_true, y_pred),
        "mae_days": mae_days_from_groups(y_true, y_pred, group_to_days=group_to_days),
    }

