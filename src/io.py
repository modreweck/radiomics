from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from matplotlib.figure import Figure


def ensure_dir(path: Path) -> Path:
    """Create directory if it does not exist and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix} ({path})")


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=index)
    return path


def save_parquet(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """
    Tries Parquet first; raises a clear error if parquet engine is unavailable.
    """
    path = Path(path)
    ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=index)
    except Exception as e:
        raise RuntimeError(
            f"Failed to write parquet: {path}\n"
            "Ensure pyarrow is installed and compatible.\n"
            f"Original error: {e}"
        ) from e
    return path


def save_dataset(df: pd.DataFrame, base_path: Path, index: bool = False) -> dict[str, Path]:
    """
    Saves both CSV and Parquet using a single base path without suffix.
    Example: base_path='data/processed/X_radiomics' produces X_radiomics.csv and .parquet
    """
    base_path = Path(base_path)
    csv_path = base_path.with_suffix(".csv")
    pqt_path = base_path.with_suffix(".parquet")

    saved = {"csv": save_csv(df, csv_path, index=index)}
    try:
        saved["parquet"] = save_parquet(df, pqt_path, index=index)
    except RuntimeError:
        # keep CSV as guaranteed format
        saved["parquet"] = pqt_path  # path reserved, may not exist
    return saved


# ---------------------------
# Standardized outputs for results/
# ---------------------------
def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """
    Standard helper for saving result tables.
    Uses CSV for maximum portability.
    """
    return save_csv(df, path, index=index)


def save_json(obj: dict, path: Path, indent: int = 2) -> Path:
    """Save a dict-like object as JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    return path


def save_figure(fig: Figure, path: Path, dpi: int = 300) -> Path:
    """
    Save a Matplotlib figure (PNG recommended).
    """
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
