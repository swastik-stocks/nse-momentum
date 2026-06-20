"""
NSE Momentum v4.0 — Price History Collector
Builds and maintains the local price_history SQLite table.
Run once to seed 2yr history, then daily to update.

Usage:
    python collectors/price_collector.py          # seed/update all 504 stocks
    python collectors/price_collector.py RELIANCE # update single stock
"""

import sys
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.schema import get_connection, DB_PATH, init_all_tables
from nse_universe import NSE_UNIVERSE

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

HISTORY_YEARS = 2


def _last_date_in_db(ticker: str) -> str | None:
    """Return the most recent date for a ticker in price_history."""
    conn = get_connection()
    row = conn.execute(
        "SELECT MAX(date) FROM price_history WHERE ticker=?", (ticker,)
    ).fetchone()
    conn.close()
    return row[0] if row and row[0] else None


def fetch_and_store(ticker: str, full_reload: bool = False) -> int:
    """
    Download OHLCV for ticker and store in price_history.
    Returns number of new rows inserted.
    """
    yf_ticker = ticker if ticker.endswith(".NS") else f"{ticker}.NS"
    last_date = _last_date_in_db(ticker) if not full_reload else None

    if last_date:
        # Incremental: fetch from last_date onward
        start = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # Full: fetch 2yr history
        start = (datetime.today() - timedelta(days=365 * HISTORY_YEARS)).strftime("%Y-%m-%d")

    try:
        df = yf.download(yf_ticker, start=start, progress=False, auto_adjust=True)
    except Exception as e:
        log.warning(f"{ticker}: download failed — {e}")
        return 0

    if df is None or df.empty:
        return 0

    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df["ticker"] = ticker
    df["date"]   = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    rows = []
    for _, row in df.iterrows():
        rows.append((
            ticker,
            row["date"],
            float(row.get("Open",  0) or 0),
            float(row.get("High",  0) or 0),
            float(row.get("Low",   0) or 0),
            float(row.get("Close", 0) or 0),
            int(row.get("Volume",  0) or 0),
            float(row.get("Close", 0) or 0),  # adj_close = close (already adjusted)
        ))

    if not rows:
        return 0

    conn = get_connection()
    conn.executemany("""
        INSERT OR IGNORE INTO price_history
        (ticker, date, open, high, low, close, volume, adj_close)
        VALUES (?,?,?,?,?,?,?,?)
    """, rows)
    inserted = conn.total_changes
    conn.commit()
    conn.close()
    return len(rows)


def seed_all(max_stocks: int = 504) -> None:
    """
    Download 2yr history for all universe stocks.
    Takes ~30-45 minutes for 504 stocks. Run once.
    """
    init_all_tables()
    tickers = list(dict.fromkeys(s[0] for s in NSE_UNIVERSE))[:max_stocks]
    total   = len(tickers)

    log.info(f"Seeding price_history for {total} stocks (2yr history)...")
    log.info("This will take ~30-45 minutes. Progress shown every 50 stocks.")

    inserted_total = 0
    failed         = []

    for i, ticker in enumerate(tickers, 1):
        n = fetch_and_store(ticker, full_reload=True)
        inserted_total += n
        if n == 0:
            failed.append(ticker)
        if i % 50 == 0:
            log.info(f"  {i}/{total} done | {inserted_total:,} rows so far | {len(failed)} failed")
        time.sleep(0.1)  # rate limit courtesy

    log.info(f"\nSeed complete: {inserted_total:,} rows | {len(failed)} tickers failed")
    if failed:
        log.warning(f"Failed: {failed[:20]}")


def update_all() -> None:
    """
    Daily incremental update — fetch only new bars since last stored date.
    Run after 3:30 PM IST every trading day.
    """
    tickers = list(dict.fromkeys(s[0] for s in NSE_UNIVERSE))
    new_rows = 0
    for ticker in tickers:
        n = fetch_and_store(ticker)
        new_rows += n
        time.sleep(0.05)
    log.info(f"Daily update complete: {new_rows} new rows added")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        t = sys.argv[1].upper()
        n = fetch_and_store(t, full_reload=True)
        print(f"Fetched {n} rows for {t}")
    else:
        seed_all()
