from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from config import Config
from data_processing import load_csv, basic_sanity_checks
from feature_engineering import add_time_features, get_feature_columns
from train_model import train
from evaluate_model import evaluate, save_json
from score_and_export import score_dataframe, export_outputs
from sqlite_loader import to_sqlite, add_indexes

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to creditcard.csv")
    p.add_argument("--model", type=str, default="rf", choices=["rf", "logreg"])
    return p.parse_args()

def main() -> None:
    args = parse_args()
    config = Config()

    loaded = load_csv(args.data)
    df = loaded.df
    basic_sanity_checks(df)

    # Feature engineering
    df_fe = add_time_features(df)
    feature_cols = get_feature_columns(df_fe)

    # Train
    artifacts = train(df_fe, feature_cols, config, model_name=args.model)

    # Evaluate
    y_prob_test = artifacts.pipeline.predict_proba(artifacts.X_test)[:, 1]
    eval_res = evaluate(config, artifacts.y_test.values, y_prob_test)

    # Save model + metrics + threshold
    config.outputs_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.pipeline, config.model_path)
    save_json(config.metrics_path, {"model": artifacts.model_name, **eval_res.metrics})
    save_json(config.threshold_path, {"threshold": eval_res.threshold, "target_recall": config.target_recall})

    # Score full dataset and export
    scored = score_dataframe(df_fe, feature_cols, artifacts.pipeline, eval_res.threshold)
    export_outputs(config, scored)

    # Load raw transactions into SQLite for SQL analysis
    to_sqlite(df_fe, config.sqlite_db_path, "transactions")
    add_indexes(config.sqlite_db_path)

    print("Done.")
    print(f"Model saved: {config.model_path}")
    print(f"Metrics: {config.metrics_path}")
    print(f"Scored CSV: {config.scored_csv_path}")
    print(f"High risk CSV: {config.high_risk_csv_path}")
    print(f"SQLite DB: {config.sqlite_db_path}")

if __name__ == "__main__":
    main()