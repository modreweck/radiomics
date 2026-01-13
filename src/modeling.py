from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import confusion_matrix

import lightgbm as lgb

from src.metrics import classification_metrics_multiclass, regression_metrics


def eval_stratifiedkfold_multiclass_lr(
    X: pd.DataFrame,
    y: pd.Series,
    labels_order: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 42,
    max_iter: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Multiclass Logistic Regression with MinMax scaling.
    Unit of analysis: image.
    Returns: folds, summary, confusion matrix, OOF predictions.
    """
    if labels_order is None:
        labels_order = ["GC", "G2", "G5", "G7", "G14"]

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=max_iter,
            random_state=seed
        ))
    ])

    rows = []
    cm_total = np.zeros((len(labels_order), len(labels_order)), dtype=int)
    oof_pred = np.array([None] * len(X), dtype=object)

    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        oof_pred[te] = pred

        m = classification_metrics_multiclass(y.iloc[te], pred)
        cm_total += confusion_matrix(y.iloc[te], pred, labels=labels_order)
        rows.append({"fold": fold, "n_test": len(te), **m})

    df_folds = pd.DataFrame(rows)

    summary = (
        df_folds.drop(columns=["fold", "n_test"])
        .agg(["mean", "std"])
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )

    cm_df = pd.DataFrame(cm_total, index=labels_order, columns=labels_order)
    oof_df = pd.DataFrame({"y_true": y.values, "y_pred": oof_pred})

    return df_folds, summary, cm_df, oof_df


def eval_stratifiedkfold_multiclass_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    labels_order: list[str] | None = None,
    n_splits: int = 5,
    seed: int = 42,
    n_estimators: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Multiclass LightGBM with MinMax scaling (protocol consistency).
    Unit of analysis: image.
    """
    if labels_order is None:
        labels_order = ["GC", "G2", "G5", "G7", "G14"]

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", lgb.LGBMClassifier(
            random_state=seed,
            n_estimators=n_estimators,
        ))
    ])

    rows = []
    cm_total = np.zeros((len(labels_order), len(labels_order)), dtype=int)

    for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])

        m = classification_metrics_multiclass(y.iloc[te], pred)
        cm_total += confusion_matrix(y.iloc[te], pred, labels=labels_order)

        rows.append({"fold": fold, "n_test": len(te), **m})

    df_folds = pd.DataFrame(rows)
    summary = df_folds.drop(columns=["fold", "n_test"]).agg(["mean", "std"]).T.reset_index()
    summary = summary.rename(columns={"index": "metric"})

    cm_df = pd.DataFrame(cm_total, index=labels_order, columns=labels_order)
    return df_folds, summary, cm_df


def eval_kfold_regression_elasticnet(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ElasticNet regression with MinMax scaling.
    Unit of analysis: image.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("reg", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed))
    ])

    rows = []
    oof = np.full(len(X), np.nan, dtype=float)

    for fold, (tr, te) in enumerate(cv.split(X), start=1):
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


def eval_kfold_regression_lgbm(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
    n_estimators: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    LightGBM regression with MinMax scaling.
    Unit of analysis: image.
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("reg", lgb.LGBMRegressor(
            random_state=seed,
            n_estimators=n_estimators,
        ))
    ])

    rows = []
    oof = np.full(len(X), np.nan, dtype=float)

    for fold, (tr, te) in enumerate(cv.split(X), start=1):
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
