# E-commerce Fraud Detection System (Python, SQL, Scikit-learn)

## Dataset

Credit Card Fraud Detection Dataset  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

- 284,807 transactions
- 492 fraud cases
- Highly imbalanced dataset

## Project Goals

- Identify behavioral patterns associated with fraudulent activity
- Train machine learning models for fraud classification
- Generate fraud risk scores to prioritize investigation

## Tech Stack

Python  
SQL  
Scikit-learn  
Random Forest  
Isolation Forest  
SQLite

## Pipeline

1. Data preprocessing
2. Feature engineering
3. Model training
4. Fraud risk scoring
5. SQL analysis of transaction patterns

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

python src/run_pipeline.py --data data/creditcard.csv
```
