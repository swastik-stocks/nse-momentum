"""
industry_breadth.py — P3-04

Computes count-based industry breadth (% of stocks above SMA20/50/100, %
above RSI(14)=50, % with positive 55-day return) from price_history, at
138-industry granularity, and publishes the result to Turso's
industry_breadth table for Portfolio Dashboard to render as a rotation
grid — the direct payoff of P3-03's Screener.in-scraped ticker_industry_map.

Structurally this is sector_breadth.py (P3-01/P3-02) with two swaps:
  1. Ticker->industry mapping comes from Turso's ticker_industry_map table
     (published by P3-03), not a local CSV + nse_universe.py fallback.
     There is no local/offline fallback here -- if Turso is unreachable or
     the table is empty, this script logs clearly and exits rather than
     silently computing nothing or guessing at a substitute taxonomy.
  2. Aggregation key is the 'industry' column (138-level, most granular of
     the four levels P3-03 published: broad_sector/sector/broad_industry/
     industry) instead of NSE's 20-sector official list.

Everything else -- flag definitions, price_history as the source (not the
Kaggle-capped price_history_deep), RSI(14) convention, RS55 absolute
55-day return, snapshot-not-timeseries semantics -- is identical to
sector_breadth.py and intentionally not re-derived here. See that file's
docstring for the full reasoning.

Usage:
    python industry_breadth.py             # compute + publish to Turso
    python industry_breadth.py --dry-run   # compute + print, skip Turso publish
"""

import argparse
import sys
import logging
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from database.schema import get_connection

# Reuse sector_breadth's per-stock flag computation verbatim -- same
# definitions (SMA20/50/100, RSI14, RS55), no reason to duplicate or drift.
from sector_breadth import _compute_stock_flags, MIN_HISTORY_DAYS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def load_ticker_industry_map() -> dict:
    """
    ticker -> industry (138-level), read from Turso's ticker_industry_map
    table (published by P3-03's Screener.in scrape). No local fallback --
    unlike sector_breadth.py's NSE-CSV-with-nse_universe.py-fallback
    pattern, there is no second source for industry-level classification
    in this codebase. If the table is missing or empty, that's a hard stop
    with a clear log message, not a silent partial run.

    Returns {} if the read fails or the table is empty -- caller decides
    whether that's fatal.
    """
    from turso_sync import get_client

    try:
        client = get_client()
    except SystemExit as e:
        log.error(f"Cannot reach Turso to load ticker_industry_map — {e}")
        return {}

    mapping = {}
    try:
        result = client.execute("SELECT ticker, industry FROM ticker_industry_map")
        rows = result.rows if hasattr(result, "rows") else result
        for row in rows:
            ticker, industry = row[0], row[1]
            if ticker and industry:
                mapping[ticker] = industry
    except Exception as e:
        log.error(f"Failed to read ticker_industry_map from Turso — {e}")
        return {}
    finally:
        client.close()

    return mapping


def compute_industry_breadth(as_of_date: str = None) -> list:
    """
    Returns a list of dicts, one per industry (138-level):
    {industry, breadth_date, pct_above_sma20, pct_above_sma50,
     pct_above_sma100, pct_above_rsi50, pct_above_rs55, stock_count}

    Reads price_history for every ticker present in Turso's
    ticker_industry_map, computes per-stock flags as of the latest close
    on or before as_of_date, and aggregates to %-of-stocks-meeting-each-
    condition per industry. Mirrors compute_sector_breadth() exactly except
    for the mapping source and aggregation key.
    """
    ticker_to_industry = load_ticker_industry_map()
    if not ticker_to_industry:
        log.error("ticker_industry_map is empty or unreachable — aborting. "
                   "Run sector_breadth.py's publish_ticker_sector_map()-equivalent "
                   "for industries (P3-03) before retrying.")
        return []

    tickers = list(ticker_to_industry.keys())
    log.info(f"  Loaded {len(tickers)} ticker->industry mappings from Turso")

    conn = get_connection()
    if as_of_date is None:
        row = conn.execute("SELECT MAX(date) FROM price_history").fetchone()
        as_of_date = row[0] if row and row[0] else date.today().isoformat()

    log.info(f"Computing industry breadth as of {as_of_date} for {len(tickers)} tickers...")

    industry_flags = defaultdict(lambda: {
        "above_sma20": 0, "above_sma50": 0, "above_sma100": 0,
        "above_rsi50": 0, "above_rs55": 0, "count": 0,
    })

    n_skipped_no_data = 0
    n_skipped_short_history = 0

    for i, ticker in enumerate(tickers, 1):
        industry = ticker_to_industry[ticker]
        rows = conn.execute(
            "SELECT close FROM price_history WHERE ticker = ? AND date <= ? ORDER BY date",
            (ticker, as_of_date),
        ).fetchall()
        if not rows:
            n_skipped_no_data += 1
            continue

        close = np.array([r[0] for r in rows], dtype=float)
        flags = _compute_stock_flags(close)
        if flags is None:
            n_skipped_short_history += 1
            continue

        s = industry_flags[industry]
        s["count"] += 1
        for key, val in flags.items():
            if val:
                s[key] += 1

        if i % 100 == 0:
            log.info(f"  {i}/{len(tickers)} tickers processed...")

    conn.close()

    if n_skipped_no_data or n_skipped_short_history:
        log.warning(f"  Skipped {n_skipped_no_data} tickers with no price_history rows, "
                    f"{n_skipped_short_history} with < {MIN_HISTORY_DAYS} days of history")

    results = []
    for industry, s in sorted(industry_flags.items()):
        count = s["count"]
        if count == 0:
            continue
        results.append({
            "industry":          industry,
            "breadth_date":      as_of_date,
            "pct_above_sma20":   round(s["above_sma20"]  / count * 100, 1),
            "pct_above_sma50":   round(s["above_sma50"]  / count * 100, 1),
            "pct_above_sma100":  round(s["above_sma100"] / count * 100, 1),
            "pct_above_rsi50":   round(s["above_rsi50"]  / count * 100, 1),
            "pct_above_rs55":    round(s["above_rs55"]   / count * 100, 1),
            "stock_count":       count,
        })
    return results


def print_breadth_table(results: list):
    print(f"\n{'INDUSTRY':<45}{'COUNT':>6}{'SMA20':>8}{'SMA50':>8}{'SMA100':>8}{'RSI50':>8}{'RS55':>8}")
    print("-" * 97)
    for r in sorted(results, key=lambda x: -x["pct_above_sma50"]):
        print(f"{r['industry']:<45}{r['stock_count']:>6}"
              f"{r['pct_above_sma20']:>7.1f}%{r['pct_above_sma50']:>7.1f}%"
              f"{r['pct_above_sma100']:>7.1f}%{r['pct_above_rsi50']:>7.1f}%"
              f"{r['pct_above_rs55']:>7.1f}%")


def publish_to_turso(results: list) -> int:
    from turso_sync import get_client
    now = datetime.now().isoformat()
    try:
        client = get_client()
    except SystemExit as e:
        log.warning(f"Publish skipped — {e}")
        return 0
    published = 0
    try:
        client.execute("""
            CREATE TABLE IF NOT EXISTS industry_breadth (
                industry          TEXT NOT NULL,
                breadth_date      TEXT NOT NULL,
                pct_above_sma20   REAL,
                pct_above_sma50   REAL,
                pct_above_sma100  REAL,
                pct_above_rsi50   REAL,
                pct_above_rs55    REAL,
                stock_count       INTEGER,
                published_at      TEXT,
                PRIMARY KEY (industry, breadth_date)
            )
        """)
        for r in results:
            try:
                client.execute(
                    """
                    INSERT INTO industry_breadth (
                        industry, breadth_date, pct_above_sma20, pct_above_sma50,
                        pct_above_sma100, pct_above_rsi50, pct_above_rs55,
                        stock_count, published_at
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(industry, breadth_date) DO UPDATE SET
                        pct_above_sma20=excluded.pct_above_sma20,
                        pct_above_sma50=excluded.pct_above_sma50,
                        pct_above_sma100=excluded.pct_above_sma100,
                        pct_above_rsi50=excluded.pct_above_rsi50,
                        pct_above_rs55=excluded.pct_above_rs55,
                        stock_count=excluded.stock_count,
                        published_at=excluded.published_at
                    """,
                    [r["industry"], r["breadth_date"], r["pct_above_sma20"], r["pct_above_sma50"],
                     r["pct_above_sma100"], r["pct_above_rsi50"], r["pct_above_rs55"],
                     r["stock_count"], now],
                )
                published += 1
            except Exception as e:
                log.warning(f"  Failed to publish {r['industry']}: {e}")
    finally:
        client.close()
    return published


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Compute and print only, skip Turso publish")
    ap.add_argument("--date", default=None, help="Compute as of this date (YYYY-MM-DD); defaults to latest available")
    args = ap.parse_args()

    results = compute_industry_breadth(as_of_date=args.date)

    if not results:
        print("\nNo results — see error log above (likely ticker_industry_map unreachable/empty).")
        sys.exit(1)

    print_breadth_table(results)

    if args.dry_run:
        print(f"\n[DRY RUN] Would publish {len(results)} industry rows to Turso — skipped.")
    else:
        n = publish_to_turso(results)
        print(f"\nPublished {n}/{len(results)} industry rows to Turso industry_breadth table.")
