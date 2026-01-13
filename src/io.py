from __future__ import annotations

from pathlib import Path
import pandas as pd


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
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def save_parquet(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """
    Tries Parquet first; raises a clear error if parquet engine is unavailable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
