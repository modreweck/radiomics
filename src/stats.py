from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal
import scikit_posthocs as sp
import pingouin as pg


def spearman_with_bootstrap_ci(x, y, n_boot: int = 2000, ci: float = 0.95, seed: int = 42):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    rho, p = spearmanr(x, y)

    rng = np.random.default_rng(seed)
    n = len(x)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r, _ = spearmanr(x[idx], y[idx])
        boot.append(r)

    boot = np.array(boot, dtype=float)
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1 - alpha))

    return {"rho": float(rho), "p_value": float(p), "ci_low": lo, "ci_high": hi, "n": int(n)}


def icc2_absolute(df_long: pd.DataFrame, targets: str, raters: str, ratings: str) -> pd.DataFrame:
    """
    ICC(2,1): two-way random effects, absolute agreement, single measurement.
    Expects long format with columns: targets, raters, ratings.
    """
    icc = pg.intraclass_corr(
        data=df_long,
        targets=targets,
        raters=raters,
        ratings=ratings
    )
    return icc


def epsilon_squared(H: float, k: int, n: int) -> float:
    return float((H - k + 1) / (n - k))


def kruskal_dunn_holm(df: pd.DataFrame, value_col: str, group_col: str = "group", group_order=None):
    if group_order is None:
        group_order = list(pd.unique(df[group_col]))

    groups = [df.loc[df[group_col] == g, value_col].dropna() for g in group_order]
    H, p = kruskal(*groups)
    n = int(sum(len(g) for g in groups))
    k = int(len(groups))
    eps2 = epsilon_squared(H, k, n)

    dunn = sp.posthoc_dunn(
        df,
        val_col=value_col,
        group_col=group_col,
        p_adjust="holm"
    ).loc[group_order, group_order]

    summary = pd.DataFrame({
        "metric": ["H", "p_value", "epsilon_squared", "n_samples"],
        "value": [float(H), float(p), float(eps2), n]
    })

    return summary, dunn
