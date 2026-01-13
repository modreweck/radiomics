from __future__ import annotations

import numpy as np
import pandas as pd

def minmax_scale_1d(x: np.ndarray) -> np.ndarray:
    """
    Min–max normalization for 1D arrays.
    Returns values scaled to [0, 1]. NaNs are ignored in min/max.
    If all values are equal (or all NaN), returns zeros.
    """
    x = np.asarray(x, dtype=float)

    if np.all(np.isnan(x)):
        return np.zeros_like(x)

    xmin = np.nanmin(x)
    xmax = np.nanmax(x)

    if xmax == xmin:
        return np.zeros_like(x)

    return (x - xmin) / (xmax - xmin)


def minmax_series(s: pd.Series) -> pd.Series:
    """Min–max normalization for a pandas Series, returns a Series aligned to the same index."""
    scaled = minmax_scale_1d(s.to_numpy(dtype=float))
    return pd.Series(scaled, index=s.index, name=s.name)


def minmax_df(df: pd.DataFrame) -> pd.DataFrame:
    """Column-wise min–max normalization for a DataFrame."""
    return df.apply(minmax_series, axis=0)
