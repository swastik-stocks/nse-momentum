"""
Quick, read-only inspection of a SQLite db: lists every table and its row
count. Doesn't modify anything.

Usage:
    python check_db.py data/momentum_v5.db
    python check_db.py data/momentum_v4.db
"""
import sqlite3
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "data/momentum_v5.db"

conn = sqlite3.connect(db_path)
tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()

print(f"\nTables in {db_path}:")
for (t,) in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t:<35} {count:>8,} rows")

conn.close()
