"""
NSE Momentum — Hourly History Loader
Loads nifty500_1h_built.csv (built via build_hourly_from_1min.py, resampled
from 1-minute data, 2015-2026) into its OWN table, price_history_hourly --
kept separate from price_history_deep (daily) for the same reason
load_deep_history.py keeps price_history_deep separate from price_history:
different resolution, different use case, zero risk of silently changing
what the live scanner or the daily evidence chain reads.

Usage:
    python load_deep_history_hourly.py --csv nifty500_1h_built.csv
    python load_deep_history_hourly.py --stats
"""
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from database.schema import get_connection

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

_HOURLY_DDL = """
CREATE TABLE IF NOT EXISTS price_history_hourly (
    ticker      TEXT NOT NULL,
    datetime    TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      INTEGER,
    PRIMARY KEY (ticker, datetime)
)
"""
_HOURLY_IDX = [
    "CREATE INDEX IF NOT EXISTS idx_phh_ticker ON price_history_hourly(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_phh_datetime ON price_history_hourly(datetime)",
]


def _ensure_table():
    conn = get_connection()
    conn.execute(_HOURLY_DDL)
    for stmt in _HOURLY_IDX:
        conn.execute(stmt)
    conn.commit()
    conn.close()


def load_hourly(csv_path: str, chunksize: int = 500_000) -> int:
    """
    Loads in chunks -- the built CSV is ~7.8M rows, too large to comfortably
    hold as one giant executemany() batch. Adds .NS suffix to match the
    convention used everywhere else (price_history_deep, NSE_UNIVERSE).
    """
    _ensure_table()
    conn = get_connection()
    total = 0

    log.info(f"Loading {csv_path} in chunks of {chunksize:,} rows...")
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        required = {"Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"}
        missing = required - set(chunk.columns)
        if missing:
            raise ValueError(f"CSV missing expected columns: {missing}. Got: {list(chunk.columns)}")

        chunk["ticker"] = chunk["Ticker"].astype(str).str.strip() + ".NS"
        rows = list(chunk[["ticker", "Datetime", "Open", "High", "Low", "Close", "Volume"]]
                    .itertuples(index=False, name=None))

        conn.executemany("""
            INSERT OR IGNORE INTO price_history_hourly
            (ticker, datetime, open, high, low, close, volume)
            VALUES (?,?,?,?,?,?,?)
        """, rows)
        conn.commit()
        total += len(rows)
        log.info(f"  ...{total:,} rows loaded so far")

    conn.close()
    return total


def print_stats() -> None:
    _ensure_table()
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM price_history_hourly").fetchone()[0]
    tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM price_history_hourly").fetchone()[0]
    rng = conn.execute("SELECT MIN(datetime), MAX(datetime) FROM price_history_hourly").fetchone()
    conn.close()

    print("\n" + "=" * 60)
    print("  price_history_hourly -- CURRENT STATE")
    print("=" * 60)
    print(f"  Rows            : {count:,}")
    print(f"  Distinct tickers: {tickers}")
    print(f"  Date range      : {rng[0]} to {rng[1]}" if rng[0] else "  (empty)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to nifty500_1h_built.csv")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    elif args.csv:
        n = load_hourly(args.csv)
        log.info(f"price_history_hourly now has {n:,} total rows inserted this run")
        print_stats()
    else:
        log.warning("No --csv given. Use --stats to inspect current table, "
                     "or --csv <path> to load.")
