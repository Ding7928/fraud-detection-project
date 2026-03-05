from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import Config

@dataclass
class TrainArtifacts:
    model_name: str
    pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

def build_model(config: Config, model_name: str) -> Pipeline:
    """
    Two model options:
      - 'logreg' : strong baseline with class_weight
      - 'rf'     : random forest with class_weight
    """
    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if config.use_class_weight else None,
            n_jobs=None,
        )
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", clf),
        ])
        return pipe

    if model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=config.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample" if config.use_class_weight else None,
            max_depth=None,
            min_samples_leaf=1,
        )
        pipe = Pipeline([
            ("clf", clf),
        ])
        return pipe

    raise ValueError("model_name must be one of: 'logreg', 'rf'")

def train(
    df: pd.DataFrame,
    feature_cols: list[str],
    config: Config,
    model_name: str = "rf",
) -> TrainArtifacts:
    X = df[feature_cols].copy()
    y = df["Class"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )

    pipe = build_model(config, model_name=model_name)
    pipe.fit(X_train, y_train)

    return TrainArtifacts(
        model_name=model_name,
        pipeline=pipe,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )