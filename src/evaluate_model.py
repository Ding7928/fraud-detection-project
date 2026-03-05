from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import Config

@dataclass
class EvalResults:
    metrics: Dict[str, Any]
    threshold: float

def choose_threshold_for_target_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float
) -> float:
    """Pick the smallest threshold that achieves >= target_recall (prioritizes recall)."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # Note: thresholds length = len(precisions)-1
    # recalls aligns with precisions
    candidate_thresholds = np.concatenate([thresholds, [1.0]])

    ok = np.where(recalls >= target_recall)[0]
    if len(ok) == 0:
        # If cannot achieve, choose threshold that maximizes recall (lowest threshold)
        return 0.0

    # choose threshold that yields target recall with best precision among those
    best_idx = ok[np.argmax(precisions[ok])]
    return float(candidate_thresholds[best_idx])

def evaluate(config: Config, y_true, y_prob) -> EvalResults:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    thr = choose_threshold_for_target_recall(y_true, y_prob, config.target_recall)
    y_pred = (y_prob >= thr).astype(int)

    metrics: Dict[str, Any] = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "threshold": float(thr),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
    }

    return EvalResults(metrics=metrics, threshold=thr)

def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))