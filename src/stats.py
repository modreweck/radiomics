from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal
import scikit_posthocs as sp
import pingouin as pg


# ---------------------------
# Correlation
# ---------------------------
def spearman_with_bootstrap_ci(
    x,
    y,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
):
    """
    Spearman correlation with bootstrap CI.

    Notes
    -----
    - NaNs are removed pairwise.
    - If a bootstrap resample is constant (spearman undefined), it is skipped.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n < 3:
        raise ValueError("Spearman requires at least 3 paired observations after NaN filtering.")

    rho, p = spearmanr(x, y)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]

        # Skip degenerate samples where correlation is undefined
        if np.nanstd(xb) == 0 or np.nanstd(yb) == 0:
            continue

        r, _ = spearmanr(xb, yb)
        if np.isfinite(r):
            boot.append(r)

    boot = np.asarray(boot, dtype=float)
    if boot.size == 0:
        # If all resamples degenerated, return NaN CI but keep point estimate
        return {"rho": float(rho), "p_value": float(p), "ci_low": np.nan, "ci_high": np.nan, "n": int(n)}

    alpha = (1 - ci) / 2
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1 - alpha))

    return {"rho": float(rho), "p_value": float(p), "ci_low": lo, "ci_high": hi, "n": int(n)}


# ---------------------------
# ICC
# ---------------------------
def icc2_absolute(df_long: pd.DataFrame, targets: str, raters: str, ratings: str) -> pd.DataFrame:
    """
    ICC table from pingouin.intraclass_corr.

    ICC(2,1) corresponds to:
    - two-way random effects
    - absolute agreement
    - single measurement

    Expects long format with columns: targets, raters, ratings.
    """
    icc = pg.intraclass_corr(
        data=df_long,
        targets=targets,
        raters=raters,
        ratings=ratings,
    )
    return icc


# ---------------------------
# Non-parametric group comparison
# ---------------------------
def epsilon_squared(H: float, k: int, n: int) -> float:
    """
    Epsilon-squared effect size for Kruskal–Wallis.
    eps^2 = (H - k + 1) / (n - k)
    """
    if (n - k) <= 0:
        return float("nan")
    eps2 = (H - k + 1) / (n - k)
    # clamp to 0 to avoid negative values due to sampling noise
    return float(max(eps2, 0.0))


def kruskal_wallis_epsilon2(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "group",
    order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Kruskal–Wallis test with epsilon².

    Returns a one-row DataFrame:
    - test, value_col, group_col
    - H, p_value, epsilon2
    - k_groups, n_total
    - group_order
    """
    if order is None:
        order = sorted(df[group_col].dropna().unique().tolist())

    groups = []
    nonempty_order = []
    for g in order:
        vals = df.loc[df[group_col] == g, value_col].astype(float).dropna()
        if len(vals) > 0:
            groups.append(vals)
            nonempty_order.append(g)

    if len(groups) < 2:
        raise ValueError("Need at least two non-empty groups for Kruskal–Wallis.")

    H, p = kruskal(*groups)
    n = int(sum(len(v) for v in groups))
    k = int(len(groups))
    eps2 = epsilon_squared(float(H), k, n)

    return pd.DataFrame([{
        "test": "Kruskal–Wallis",
        "value_col": value_col,
        "group_col": group_col,
        "H": float(H),
        "p_value": float(p),
        "epsilon2": float(eps2),
        "k_groups": k,
        "n_total": n,
        "group_order": nonempty_order,
    }])


def dunn_posthoc_holm(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "group",
    order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Dunn post-hoc test with Holm adjustment.
    Returns a square DataFrame (groups x groups) of adjusted p-values.
    """
    if order is None:
        order = sorted(df[group_col].dropna().unique().tolist())

    tmp = df[[value_col, group_col]].copy()
    tmp = tmp.dropna()
    tmp[value_col] = tmp[value_col].astype(float)
    tmp[group_col] = tmp[group_col].astype(str)

    pvals = sp.posthoc_dunn(
        tmp,
        val_col=value_col,
        group_col=group_col,
        p_adjust="holm",
    )

    # Ensure requested order (may introduce NaN if a group is absent)
    return pvals.reindex(index=order, columns=order)


# ---------------------------
# Backwards-compatible wrapper (keeps your previous API)
# ---------------------------
def kruskal_dunn_holm(
    df: pd.DataFrame,
    value_col: str,
    group_col: str = "group",
    group_order=None,
):
    """
    Backwards-compatible wrapper.
    Returns:
      - summary: long DataFrame (metric/value)
      - dunn: square DataFrame of Holm-adjusted p-values
    """
    if group_order is None:
        group_order = list(pd.unique(df[group_col].dropna()))

    kw = kruskal_wallis_epsilon2(df, value_col=value_col, group_col=group_col, order=list(group_order))
    dunn = dunn_posthoc_holm(df, value_col=value_col, group_col=group_col, order=list(group_order))

    summary = pd.DataFrame({
        "metric": ["H", "p_value", "epsilon2", "n_total", "k_groups"],
        "value": [
            float(kw.loc[0, "H"]),
            float(kw.loc[0, "p_value"]),
            float(kw.loc[0, "epsilon2"]),
            int(kw.loc[0, "n_total"]),
            int(kw.loc[0, "k_groups"]),
        ],
    })

    return summary, dunn