from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    random_state: int = 42
    test_size: float = 0.2

    # For imbalance handling
    use_class_weight: bool = True

    # Risk scoring threshold strategy:
    # choose threshold that achieves target recall on the test set.
    target_recall: float = 0.90

    # Output paths
    outputs_dir: Path = Path("outputs")
    model_path: Path = outputs_dir / "model.joblib"
    metrics_path: Path = outputs_dir / "metrics.json"
    threshold_path: Path = outputs_dir / "threshold.json"
    scored_csv_path: Path = outputs_dir / "scored_transactions.csv"
    high_risk_csv_path: Path = outputs_dir / "high_risk_transactions.csv"
    sqlite_db_path: Path = outputs_dir / "transactions.db"