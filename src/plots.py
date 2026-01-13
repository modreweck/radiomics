from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_figure(fig, path: Path, dpi: int = 300) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def bland_altman_plot(x, y, ax=None, title=None):
    """
    Bland–Altman: difference vs mean.
    Returns bias and limits of agreement.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mean = (x + y) / 2
    diff = x - y
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    ax.scatter(mean, diff, s=12)
    ax.axhline(bias, linestyle="--")
    ax.axhline(loa_low, linestyle="--")
    ax.axhline(loa_high, linestyle="--")

    ax.set_xlabel("Mean of methods")
    ax.set_ylabel("Difference (ERAT_norm - EHAT_norm)")
    if title:
        ax.set_title(title)

    return {"bias": bias, "loa_low": loa_low, "loa_high": loa_high, "sd_diff": sd}, fig, ax
