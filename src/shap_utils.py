from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_shap_importance(importance_df: pd.DataFrame, out_csv: Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(out_csv, index=False)
    return out_csv
