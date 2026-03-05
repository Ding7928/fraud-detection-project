# E-commerce Fraud Detection System (Python, SQL, Scikit-learn)

## Dataset

Download "Credit Card Fraud Detection" dataset from Kaggle and place `creditcard.csv` in `data/`.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt

python src/run_pipeline.py --data data/creditcard.csv
```
