from __future__ import annotations

import sqlite3
from pathlib import Path
import pandas as pd

def to_sqlite(df: pd.DataFrame, db_path: Path, table_name: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)

def add_indexes(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        # Indexes for typical analysis
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_time ON transactions(Time);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_class ON transactions(Class);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_amount ON transactions(Amount);")
        finally:
            conn.commit()