from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    raw: Path
    processed: Path
    results: Path
    figures: Path
    tables: Path
    text: Path
    supplementary: Path
    supp_shap: Path
    supp_tables: Path
    supp_text: Path


def find_project_root(start: Path | None = None) -> Path:
    """
    Finds the repository root by searching for key files/folders.
    Works when executed from notebooks/ or root.
    """
    start = start or Path.cwd()
    candidates = [start] + list(start.parents)

    for p in candidates:
        if (p / "data").exists() and (p / "requirements.txt").exists():
            return p
    # fallback
    return Path.cwd()


def get_paths() -> ProjectPaths:
    root = find_project_root()
    data = root / "data"
    raw = data / "raw"
    processed = data / "processed"

    results = root / "results"
    figures = results / "figures"
    tables = results / "tables"
    text = results / "text"

    supplementary = root / "supplementary"
    supp_shap = supplementary / "shap"
    supp_tables = supplementary / "tables"
    supp_text = supplementary / "text"

    for d in [raw, processed, figures, tables, text, supp_shap, supp_tables, supp_text]:
        d.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        root=root,
        data=data,
        raw=raw,
        processed=processed,
        results=results,
        figures=figures,
        tables=tables,
        text=text,
        supplementary=supplementary,
        supp_shap=supp_shap,
        supp_tables=supp_tables,
        supp_text=supp_text,
    )


def set_seed(seed: int = 42) -> int:
    np.random.seed(seed)
    return seed
