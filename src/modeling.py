from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import confusion_matrix

import lightgbm as lgb

from src.metrics import (
    classification_metrics_multiclass,
    regression_metrics,
)


def _make_groups_default(n: int) -> np.ndarray:
    """
    Placeholder groups when no grouping variable exists.
    NOTE: This behaves like no grouping (each sample is its own group).
    Prefer providing real groups (e.g., animal_id) whenever possible.
    """
    return np.arange(n)


def eval_groupkfold_multiclass_lr(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray | None = None,
    labels_order: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if labels_order is None:
        labels_order = ["GC", "G2", "G5", "G7", "G14"]
    if groups is None:
        groups = _make_groups_default(len(X))

    cv = GroupKFold(n_splits=n_splits)

    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=5000,
            random_state=seed
        ))
    ])

    rows = []
    cm_total = np.zeros((len(labels_order), len(labels_order)), dtype=int)

    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups), start=1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])

        m = classification_metrics_multiclass(y.iloc[te], pred)
        cm = confusion_matrix(y.iloc[te], pred, labels=labels_order)
        cm_total += cm

        rows.append({"fold": fold, "n_test": len(te), **m})

    df_folds = pd.DataFrame(rows)
    summary = df_folds.drop(columns=["fold", "n_test"]).agg(["mean", "std"]).T.reset_index()
    summary = summary.rename(columns={"index": "metric"})

    cm_df = pd.DataFrame(cm_total, index=labels_order, columns=labels_order)
    return df_folds, summary, cm_df


def eval_groupkfold_multiclass_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray | None = None,
    labels_order: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if labels_order is None:
        labels_order = ["GC", "G2", "G5", "G7", "G14"]
    if groups is None:
        groups = _make_groups_default(len(X))

    cv = GroupKFold(n_splits=n_splits)

    # LightGBM does not strictly require scaling, but keep it for protocol consistency
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", lgb.LGBMClassifier(
            random_state=seed,
            n_estimators=500,
        ))
    ])

    rows = []
    cm_total = np.zeros((len(labels_order), len(labels_order)), dtype=int)

    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups), start=1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])

        m = classification_metrics_multiclass(y.iloc[te], pred)
        cm = confusion_matrix(y.iloc[te], pred, labels=labels_order)
        cm_total += cm

        rows.append({"fold": fold, "n_test": len(te), **m})

    df_folds = pd.DataFrame(rows)
    summary = df_folds.drop(columns=["fold", "n_test"]).agg(["mean", "std"]).T.reset_index()
    summary = summary.rename(columns={"index": "metric"})

    cm_df = pd.DataFrame(cm_total, index=labels_order, columns=labels_order)
    return df_folds, summary, cm_df


def eval_groupkfold_regression_elasticnet(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if groups is None:
        groups = _make_groups_default(len(X))

    cv = GroupKFold(n_splits=n_splits)

    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("reg", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed))
    ])

    rows = []
    oof = np.full(len(X), np.nan, dtype=float)

    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups), start=1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        oof[te] = pred

        m = regression_metrics(y.iloc[te], pred)
        rows.append({"fold": fold, "n_test": len(te), **m})

    df_folds = pd.DataFrame(rows)
    summary = df_folds.drop(columns=["fold", "n_test"]).agg(["mean", "std"]).T.reset_index()
    summary = summary.rename(columns={"index": "metric"})

    oof_df = pd.DataFrame({"y_true": y.values, "y_pred": oof})
    return df_folds, summary, oof_df


def eval_groupkfold_regression_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if groups is None:
        groups = _make_groups_default(len(X))

    cv = GroupKFold(n_splits=n_splits)

    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("reg", lgb.LGBMRegressor(
            random_state=seed,
            n_estimators=1000,
        ))
    ])

    rows = []
    oof = np.full(len(X), np.nan, dtype=float)

    for fold, (tr, te) in enumerate(cv.split(X, y, groups=groups), start=1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        oof[te] = pred

        m = regression_metrics(y.iloc[te], pred)
        rows.append({"fold": fold, "n_test": len(te), **m})

    df_folds = pd.DataFrame(rows)
    summary = df_folds.drop(columns=["fold", "n_test"]).agg(["mean", "std"]).T.reset_index()
    summary = summary.rename(columns={"index": "metric"})

    oof_df = pd.DataFrame({"y_true": y.values, "y_pred": oof})
    return df_folds, summary, oof_df