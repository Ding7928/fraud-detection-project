from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = {"Time", "Amount", "Class"}

@dataclass
class LoadedData:
    df: pd.DataFrame

def load_csv(path: str | Path) -> LoadedData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)[:10]}...")

    # Basic cleanup
    df = df.dropna().copy()
    df["Class"] = df["Class"].astype(int)

    return LoadedData(df=df)

def basic_sanity_checks(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Dataset is empty after loading/cleanup.")
    if df["Class"].nunique() != 2:
        raise ValueError(f"Expected binary Class, got: {df['Class'].unique()}")