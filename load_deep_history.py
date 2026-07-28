"""
NSE Momentum v6 — Deep History Loader
Ingests two new data sources into dedicated tables, kept SEPARATE from the
existing price_history / NSE_UNIVERSE that your live scanner depends on:

  1. Multi-year per-stock OHLCV (Kaggle nifty500_1d.csv, ~1999-2026,
     501 tickers) -> price_history_deep table
  2. Point-in-time NIFTY 500 constituent snapshots (your compiled CSV,
     2016-2026, 47 clean snapshots, always exactly 500 members) ->
     universe_snapshots table

WHY SEPARATE TABLES, NOT REPLACING price_history / NSE_UNIVERSE:
  - price_history is actively maintained by collectors/price_collector.py
    and read by your live scanner every trading day. Overwriting it with
    a different data source (different adjustment methodology, different
    date range) risks silently changing live scan behaviour.
  - NSE_UNIVERSE is a single CURRENT snapshot (used for today's scan).
    Backtesting needs the OPPOSITE — the correct universe AS OF each
    historical test date, which is what get_point_in_time_universe()
    below provides. These are different use cases, not two versions of
    the same thing.

  validation/pipeline_replay.py and validation/backtest.py can be updated
  to read from price_history_deep + get_point_in_time_universe() instead
  of price_history + NSE_UNIVERSE for a proper multi-year, survivorship-
  bias-free backtest — that update is a separate next step, not done by
  this loader.

Usage:
    python load_deep_history.py --price-csv nifty500_1d.csv --universe-csv nifty500_2016-01-01_to_2026-12-31.csv
    python load_deep_history.py --stats                       # print current table contents only

CHANGELOG (v6.1 — perf fix):
    get_point_in_time_universe() previously opened a brand-new SQLite
    connection on EVERY call. Called ~250,000 times across a full deep
    replay (501 tickers x ~500 test dates each), against only 47 distinct
    snapshot rows that ever change. This caused severe cumulative slowdown
    and, after an interrupted run left a stale WAL lock, a silent
    multi-hour hang with no timeout and no error.

    Fixed by loading all 47 snapshots into an in-memory list ONCE
    (_load_snapshot_cache) and answering all subsequent lookups via
    bisect against that list, wrapped in functools.lru_cache so repeated
    lookups for the same date_str (common — many tickers share test
    dates) are free. Net effect: ~250,000 DB round-trips collapse to 1.
"""

import sys
import csv
import bisect
import logging
import argparse
from pathlib import Path
from datetime import datetime
from functools import lru_cache

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

_DEEP_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS price_history_deep (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      INTEGER,
    PRIMARY KEY (ticker, date)
)
"""
_DEEP_HISTORY_IDX = [
    "CREATE INDEX IF NOT EXISTS idx_phd_ticker ON price_history_deep(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_phd_date   ON price_history_deep(date)",
]

_UNIVERSE_SNAPSHOT_DDL = """
CREATE TABLE IF NOT EXISTS universe_snapshots (
    effective_date  TEXT PRIMARY KEY,
    symbols         TEXT NOT NULL,
    n_constituents  INTEGER
)
"""


def _ensure_tables():
    conn = get_connection()
    conn.execute(_DEEP_HISTORY_DDL)
    for stmt in _DEEP_HISTORY_IDX:
        conn.execute(stmt)
    conn.execute(_UNIVERSE_SNAPSHOT_DDL)
    conn.commit()
    conn.close()


def load_price_history_deep(csv_path: str) -> int:
    """
    Load the Kaggle nifty500_1d.csv into price_history_deep.
    Adds the .NS suffix to raw tickers (e.g. 'RELIANCE' -> 'RELIANCE.NS')
    to match the convention used everywhere else in this codebase.
    Returns number of rows inserted.
    """
    log.info(f"Reading {csv_path} (this file is large — may take a minute)...")
    df = pd.read_csv(csv_path)

    required = {"Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}. Got: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["Datetime"]).dt.strftime("%Y-%m-%d")
    df["ticker"] = df["Ticker"].astype(str).str.strip() + ".NS"

    rows = list(df[["ticker", "date", "Open", "High", "Low", "Close", "Volume"]].itertuples(
        index=False, name=None
    ))

    log.info(f"Inserting {len(rows):,} rows into price_history_deep...")
    conn = get_connection()
    conn.executemany("""
        INSERT OR IGNORE INTO price_history_deep
        (ticker, date, open, high, low, close, volume)
        VALUES (?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    inserted = conn.execute("SELECT COUNT(*) FROM price_history_deep").fetchone()[0]
    conn.close()
    return inserted


def load_universe_snapshots(csv_path: str) -> int:
    """
    Load the point-in-time constituent snapshot CSV into universe_snapshots.
    Adds .NS suffix to each symbol for consistency with the rest of the
    codebase. Returns number of snapshot dates loaded.
    """
    log.info(f"Reading {csv_path}...")
    rows_written = 0
    conn = get_connection()

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            effective_date = row["effective_date"].strip()
            raw_symbols = [s.strip() for s in row["symbols"].split(",") if s.strip()]
            symbols_ns = [s + ".NS" for s in raw_symbols]
            symbols_str = ",".join(symbols_ns)

            conn.execute("""
                INSERT OR REPLACE INTO universe_snapshots
                (effective_date, symbols, n_constituents)
                VALUES (?,?,?)
            """, (effective_date, symbols_str, len(symbols_ns)))
            rows_written += 1

    conn.commit()
    conn.close()

    # Invalidate in-memory caches — the on-disk table just changed underneath them.
    invalidate_snapshot_cache()

    return rows_written


# ---------------------------------------------------------------------------
# In-memory point-in-time universe cache (perf fix — see module changelog)
# ---------------------------------------------------------------------------

_SNAPSHOT_CACHE = None  # list[(effective_date: str, tickers: list[str])], sorted ascending


def _load_snapshot_cache():
    global _SNAPSHOT_CACHE
    if _SNAPSHOT_CACHE is None:
        conn = get_connection()
        rows = conn.execute(
            "SELECT effective_date, symbols FROM universe_snapshots ORDER BY effective_date ASC"
        ).fetchall()
        conn.close()
        _SNAPSHOT_CACHE = [(r["effective_date"], r["symbols"].split(",")) for r in rows]
        log.info(f"  [cache] Loaded {len(_SNAPSHOT_CACHE)} universe snapshots into memory "
                 f"(point-in-time lookups now in-memory, no per-call DB hit).")
    return _SNAPSHOT_CACHE


def invalidate_snapshot_cache():
    """Call this after writing new rows to universe_snapshots so a long-running
    process (or the next call in the same process) picks up fresh data."""
    global _SNAPSHOT_CACHE
    _SNAPSHOT_CACHE = None
    get_point_in_time_universe.cache_clear()


@lru_cache(maxsize=4096)
def get_point_in_time_universe(date_str: str) -> list:
    """
    Returns the list of tickers (with .NS suffix) that were actual NIFTY
    500 constituents as of the most recent snapshot at or before date_str.
    This is the function validation/backtest.py and pipeline_replay.py
    should call per test-date instead of using the static NSE_UNIVERSE,
    to eliminate survivorship bias in the historical replay.

    Perf note: answered via in-memory bisect against a snapshot list
    loaded once per process (see _load_snapshot_cache), wrapped in
    lru_cache so repeated lookups for the same date_str are free. There
    are only ~47 distinct snapshot dates, so this cache is tiny and never
    needs eviction in practice.

    Returns [] if date_str is before the earliest available snapshot.
    """
    snapshots = _load_snapshot_cache()
    if not snapshots:
        return []
    dates = [s[0] for s in snapshots]
    idx = bisect.bisect_right(dates, date_str) - 1
    if idx < 0:
        return []
    return snapshots[idx][1]


def print_stats() -> None:
    _ensure_tables()
    conn = get_connection()

    ph_count = conn.execute("SELECT COUNT(*) FROM price_history_deep").fetchone()[0]
    ph_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM price_history_deep").fetchone()[0]
    ph_range = conn.execute("SELECT MIN(date), MAX(date) FROM price_history_deep").fetchone()

    us_count = conn.execute("SELECT COUNT(*) FROM universe_snapshots").fetchone()[0]
    us_range = conn.execute("SELECT MIN(effective_date), MAX(effective_date) FROM universe_snapshots").fetchone()

    conn.close()

    print("\n" + "=" * 60)
    print("  DEEP HISTORY TABLES — CURRENT STATE")
    print("=" * 60)
    print(f"  price_history_deep")
    print(f"    Rows           : {ph_count:,}")
    print(f"    Distinct tickers: {ph_tickers}")
    print(f"    Date range     : {ph_range[0]} to {ph_range[1]}" if ph_range[0] else "    (empty)")
    print()
    print(f"  universe_snapshots")
    print(f"    Snapshot dates : {us_count}")
    print(f"    Date range     : {us_range[0]} to {us_range[1]}" if us_range[0] else "    (empty)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--price-csv", default=None, help="Path to nifty500_1d.csv")
    parser.add_argument("--universe-csv", default=None, help="Path to the point-in-time constituent CSV")
    parser.add_argument("--stats", action="store_true", help="Print current table contents only")
    args = parser.parse_args()

    _ensure_tables()

    if args.stats:
        print_stats()
    else:
        if args.price_csv:
            n = load_price_history_deep(args.price_csv)
            log.info(f"price_history_deep now has {n:,} total rows")
        if args.universe_csv:
            n = load_universe_snapshots(args.universe_csv)
            log.info(f"Loaded {n} universe snapshot dates")
        if not args.price_csv and not args.universe_csv:
            log.warning("No --price-csv or --universe-csv given — nothing to do. Use --stats to inspect current tables.")
        print_stats()
