from __future__ import annotations

import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lightweight behavioral/time features from Time (seconds)."""
    out = df.copy()

    # hour-of-day proxy using modulo 24h
    seconds_in_day = 24 * 3600
    out["time_in_day"] = out["Time"] % seconds_in_day
    out["hour_bucket"] = (out["time_in_day"] // 3600).astype(int)

    # rolling "burstiness" per time order (global, not per user since dataset lacks user_id)
    # Use differences between consecutive times
    out = out.sort_values("Time").reset_index(drop=True)
    out["delta_time"] = out["Time"].diff().fillna(0.0)
    out["is_burst"] = (out["delta_time"] <= 1.0).astype(int)  # very tight bursts

    return out

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Use all numeric columns except target. Keep original PCA features (V1..V28) + engineered."""
    cols = []
    for c in df.columns:
        if c == "Class":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols