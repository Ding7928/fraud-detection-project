from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from config import Config

def risk_score_from_probability(p: np.ndarray) -> np.ndarray:
    """Risk score in [0, 100]."""
    p = np.clip(p, 0.0, 1.0)
    return (p * 100.0).round(2)

def score_dataframe(df: pd.DataFrame, feature_cols: list[str], model, threshold: float) -> pd.DataFrame:
    X = df[feature_cols].copy()
    probs = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["fraud_probability"] = probs
    out["risk_score"] = risk_score_from_probability(probs)
    out["is_high_risk"] = (probs >= threshold).astype(int)
    return out

def export_outputs(
    config: Config,
    scored_df: pd.DataFrame,
) -> None:
    config.outputs_dir.mkdir(parents=True, exist_ok=True)

    scored_df.to_csv(config.scored_csv_path, index=False)

    high_risk = scored_df[scored_df["is_high_risk"] == 1].copy()
    high_risk = high_risk.sort_values(["risk_score"], ascending=False)
    high_risk.to_csv(config.high_risk_csv_path, index=False)