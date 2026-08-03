"""
sector_breadth.py — P3-01 / P3-02

Computes count-based sector breadth (% of stocks above SMA20/50/100, %
above RSI(14)=50, % with positive 55-day return) from price_history_deep,
and publishes the result to Turso's sector_breadth table (schema from
P1-01) for Portfolio Dashboard to render as a rotation grid.

Deliberately count-based, not market-cap-weighted (P3-05, separate item):
per the backlog's own reasoning, count-based is "arguably more robust,
since MCap weighting lets 2-3 megacaps dominate a sector reading" — and
it's fully buildable today with data already on hand, unlike MCap
weighting which needs a market-cap source not yet wired in.

Definitions (stated explicitly since the backlog doesn't fully spell
these out numerically):
  - Source: price_history (the daily-refreshed 2yr table maintained by
    price_collector.py), NOT price_history_deep. The deep table is capped
    at the Kaggle source file's own ceiling (23 Feb 2026 as of this
    writing) -- correct for backtesting, wrong for "what's the market
    doing right now." price_history is refreshed every trading day and
    comfortably covers the ~150-day lookback SMA100/RSI/RS55 need.
  - RSI: 14-day, matching the RSI convention already used elsewhere in
    this codebase (orchestrator.py's own RSI(14) calc).
  - RS55: 55-trading-day price return (close_today / close_55d_ago - 1).
    This is a per-stock ABSOLUTE momentum measure, distinct from the RS
    PERCENTILE used elsewhere in the scanner (that's a cross-sectional
    rank against the universe; this is StockEdge's own "RS55" window-
    return convention).
  - Sector taxonomy: NSE's own official Nifty 500 constituent list
    (data/ind_nifty500list.csv, https://nsearchives.nseindia.com/content/
    indices/ind_nifty500list.csv), NOT nse_universe.py's own sector field.
    Confirmed 500/500 ticker match, 20 clean non-duplicated sectors --
    nse_universe.py's sector field has real duplication (e.g. "Financials"
    vs "Financial Services", "Auto" vs "Automobile and Auto Components",
    "Metals" vs "Metals & Mining") from the 401->500 universe expansion
    merging two different naming conventions. Falls back to
    nse_universe.py's own sector field only if a ticker isn't found in the
    official list (logged clearly, not silent) -- should be rare given
    today's confirmed 100% match rate, but a ticker added to the universe
    after this CSV was downloaded wouldn't yet be in it.
  - Snapshot as of the latest available date per ticker, not a time
    series -- re-runnable daily, same as everything else in this pipeline.

Usage:
    python sector_breadth.py            # compute + publish to Turso
    python sector_breadth.py --dry-run   # compute + print, skip Turso publish
"""

import argparse
import sys
import logging
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from database.schema import get_connection
from nse_universe import NSE_UNIVERSE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

MIN_HISTORY_DAYS = 100   # need at least 100 daily closes for SMA100

OFFICIAL_SECTOR_CSV = Path(__file__).parent / "data" / "ind_nifty500list.csv"


def _load_official_sectors() -> dict:
    """
    ticker (with .NS) -> official NSE sector, from data/ind_nifty500list.csv
    (NSE's own live-maintained Nifty 500 constituent list -- confirmed
    500/500 ticker match, 20 clean non-duplicated sectors, vs.
    nse_universe.py's own sector field which has real duplication from the
    401->500 universe expansion merging two naming conventions).

    Returns {} if the file is missing -- callers fall back to
    nse_universe.py's own sector field entirely in that case, logged
    clearly, never silently.
    """
    if not OFFICIAL_SECTOR_CSV.exists():
        return {}
    import csv as _csv
    mapping = {}
    with open(OFFICIAL_SECTOR_CSV, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            symbol = row.get("Symbol", "").strip()
            sector = row.get("Industry", "").strip()
            if symbol and sector:
                mapping[f"{symbol}.NS"] = sector
    return mapping


def _compute_stock_flags(close: np.ndarray) -> dict:
    """
    Per-stock breadth flags as of the LAST value in `close`. Returns None
    if there isn't enough history to compute SMA100 reliably -- mirrors
    the "not enough data" guards used elsewhere in this pipeline
    (compute_holding_stop, RiskAgent) rather than silently zero-filling.
    """
    if len(close) < MIN_HISTORY_DAYS:
        return None

    price  = close[-1]
    sma20  = float(np.mean(close[-20:]))
    sma50  = float(np.mean(close[-50:]))
    sma100 = float(np.mean(close[-100:]))

    delta = np.diff(close[-15:])
    gain  = float(np.mean(np.clip(delta, 0, None)))
    loss  = float(np.mean(np.clip(-delta, 0, None)))
    rsi   = 100.0 - 100.0 / (1 + gain / loss) if loss > 0 else 100.0

    rs55 = (price / close[-56] - 1) if len(close) >= 56 else None

    return {
        "above_sma20":  price > sma20,
        "above_sma50":  price > sma50,
        "above_sma100": price > sma100,
        "above_rsi50":  rsi > 50,
        "above_rs55":   rs55 is not None and rs55 > 0,
    }


def compute_sector_breadth(as_of_date: str = None) -> list:
    """
    Returns a list of dicts, one per sector:
    {sector, breadth_date, pct_above_sma20, pct_above_sma50, pct_above_sma100,
     pct_above_rsi50, pct_above_rs55, stock_count}

    Reads price_history_deep for every ticker in nse_universe.py, computes
    per-stock flags as of the latest close on or before as_of_date (defaults
    to the latest date actually in the table), and aggregates to %-of-stocks-
    meeting-each-condition per sector.
    """
    fallback_sector = {item[0]: item[2] for item in NSE_UNIVERSE}
    official_sector = _load_official_sectors()
    ticker_to_sector = {}
    n_official, n_fallback = 0, 0
    for ticker in fallback_sector:
        if ticker in official_sector:
            ticker_to_sector[ticker] = official_sector[ticker]
            n_official += 1
        else:
            ticker_to_sector[ticker] = fallback_sector[ticker]
            n_fallback += 1
    tickers = list(ticker_to_sector.keys())

    if n_fallback:
        log.warning(f"  {n_fallback} ticker(s) not found in official NSE sector list — "
                    f"using nse_universe.py's own (possibly inconsistent) sector field for these")
    log.info(f"  Sector source: {n_official} from official NSE list, {n_fallback} fallback")

    conn = get_connection()
    if as_of_date is None:
        row = conn.execute("SELECT MAX(date) FROM price_history").fetchone()
        as_of_date = row[0] if row and row[0] else date.today().isoformat()

    log.info(f"Computing sector breadth as of {as_of_date} for {len(tickers)} tickers...")

    sector_flags = defaultdict(lambda: {
        "above_sma20": 0, "above_sma50": 0, "above_sma100": 0,
        "above_rsi50": 0, "above_rs55": 0, "count": 0,
    })

    n_skipped_no_data = 0
    n_skipped_short_history = 0

    for i, ticker in enumerate(tickers, 1):
        sector = ticker_to_sector[ticker]
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

        s = sector_flags[sector]
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
    for sector, s in sorted(sector_flags.items()):
        count = s["count"]
        if count == 0:
            continue
        results.append({
            "sector":            sector,
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
    print(f"\n{'SECTOR':<38}{'COUNT':>6}{'SMA20':>8}{'SMA50':>8}{'SMA100':>8}{'RSI50':>8}{'RS55':>8}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: -x["pct_above_sma50"]):
        print(f"{r['sector']:<38}{r['stock_count']:>6}"
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
        for r in results:
            try:
                client.execute(
                    """
                    INSERT INTO sector_breadth (
                        sector, breadth_date, pct_above_sma20, pct_above_sma50,
                        pct_above_sma100, pct_above_rsi50, pct_above_rs55,
                        stock_count, published_at
                    ) VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(sector, breadth_date) DO UPDATE SET
                        pct_above_sma20=excluded.pct_above_sma20,
                        pct_above_sma50=excluded.pct_above_sma50,
                        pct_above_sma100=excluded.pct_above_sma100,
                        pct_above_rsi50=excluded.pct_above_rsi50,
                        pct_above_rs55=excluded.pct_above_rs55,
                        stock_count=excluded.stock_count,
                        published_at=excluded.published_at
                    """,
                    [r["sector"], r["breadth_date"], r["pct_above_sma20"], r["pct_above_sma50"],
                     r["pct_above_sma100"], r["pct_above_rsi50"], r["pct_above_rs55"],
                     r["stock_count"], now],
                )
                published += 1
            except Exception as e:
                log.warning(f"  Failed to publish {r['sector']}: {e}")
    finally:
        client.close()
    return published


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Compute and print only, skip Turso publish")
    ap.add_argument("--date", default=None, help="Compute as of this date (YYYY-MM-DD); defaults to latest available")
    args = ap.parse_args()

    results = compute_sector_breadth(as_of_date=args.date)
    print_breadth_table(results)

    if args.dry_run:
        print(f"\n[DRY RUN] Would publish {len(results)} sector rows to Turso — skipped.")
    else:
        n = publish_to_turso(results)
        print(f"\nPublished {n}/{len(results)} sector rows to Turso sector_breadth table.")
