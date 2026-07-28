"""
NSE Momentum v6 — Deep Regime Backfill
Extends market_regime_history (same table regime_backfill.py wrote to,
same schema) back across the FULL available NIFTY/BankNifty/VIX history
instead of just 2 years, using price_history_deep (1999-2026) for the
breadth proxy instead of the old 2yr price_history.

Writes into the SAME market_regime_history table — validation/
pipeline_replay.py's existing _load_nifty_history() / _load_regime_map()
need no changes to read this deeper data.

KNOWN LIMITS (same honesty as regime_backfill.py):
  - Breadth is still a universe-bounded proxy (your own tickers' above-
    50-EMA %), not true NSE-wide historical A/D — that data isn't stored
    anywhere accessible.
  - India VIX (^INDIAVIX) historical data via yfinance realistically only
    goes back to ~2009 (the index didn't exist before then in its current
    form). Dates before VIX availability get a neutral 15.0 default,
    logged once at the start, not silently.
  - Uses ALL tickers present in price_history_deep on a given date for
    the breadth proxy, not strictly the point-in-time NIFTY 500 members
    (that would require joining against universe_snapshots per date,
    a further refinement not done here to keep this step bounded).

Usage:
    python regime_backfill_deep.py
    python regime_backfill_deep.py --stats
"""

import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from database.schema import get_connection, init_all_tables
from agents.market_agent import REGIME_CONFIG

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def _ema(v: np.ndarray, span: int) -> np.ndarray:
    alpha = 2 / (span + 1)
    e = np.zeros(len(v))
    e[0] = v[0]
    for i in range(1, len(v)):
        e[i] = alpha * v[i] + (1 - alpha) * e[i - 1]
    return e


def _fetch_index_history_max(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df[["date", "Close"]].rename(columns={"Close": "close"}).astype({"close": float})


def _fetch_vix_history_max() -> dict:
    try:
        df = yf.download("^INDIAVIX", period="max", progress=False, auto_adjust=True)
        if df is None or df.empty:
            log.warning("India VIX history unavailable at all — using neutral 15.0 for every date")
            return {}
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        vix_map = dict(zip(df["date"], df["Close"].astype(float)))
        log.info(f"India VIX history available from {df['date'].min()} onward "
                 f"— dates before this use neutral 15.0")
        return vix_map
    except Exception as e:
        log.warning(f"India VIX fetch failed ({e}) — using neutral 15.0 for every date")
        return {}


def _load_deep_close_matrix() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql("SELECT ticker, date, close FROM price_history_deep", conn)
    conn.close()
    wide = df.pivot(index="date", columns="ticker", values="close").sort_index()
    return wide


def _compute_above_50_ema_series(wide: pd.DataFrame) -> pd.Series:
    def _col_ema(col):
        filled = col.ffill().bfill().to_numpy(dtype=float)
        if len(filled) == 0 or np.all(np.isnan(filled)):
            return pd.Series(np.full(len(col), np.nan), index=col.index)
        return pd.Series(_ema(filled, 50), index=col.index)

    ema50 = wide.apply(_col_ema)
    above = (wide > ema50)
    valid = wide.notna()
    pct = (above & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan) * 100
    return pct


def backfill_deep() -> None:
    init_all_tables()

    log.info("Fetching FULL available NIFTY 50 and BANK NIFTY history (period=max)...")
    nifty = _fetch_index_history_max("^NSEI")
    bank = _fetch_index_history_max("^NSEBANK")
    log.info(f"NIFTY history: {nifty['date'].min()} to {nifty['date'].max()} ({len(nifty)} days)")
    vix_map = _fetch_vix_history_max()

    log.info("Loading price_history_deep close matrix for breadth proxy "
             "(this may take a minute — 1.75M rows)...")
    wide = _load_deep_close_matrix()
    above50_series = _compute_above_50_ema_series(wide)

    nifty_close_s = nifty.set_index("date")["close"]
    n = len(nifty_close_s)
    nifty_c = nifty_close_s.to_numpy(dtype=float)
    bank_close_s = bank.set_index("date")["close"].reindex(nifty_close_s.index).ffill()
    bank_c = bank_close_s.to_numpy(dtype=float)

    nifty_ema50 = _ema(nifty_c, 50)
    bank_ema50 = _ema(bank_c, 50)

    conn = get_connection()
    rows_written = 0
    regime_order = ["A", "B", "C", "D", "E"]

    for i, date in enumerate(nifty_close_s.index):
        if i < 55:
            continue

        above50 = above50_series.get(date, np.nan)
        if pd.isna(above50):
            continue

        vix = vix_map.get(date, 15.0)

        breadth_score = above50 / 10.0
        if breadth_score >= 8:   breadth_signal = "A"
        elif breadth_score >= 6: breadth_signal = "B"
        elif breadth_score >= 4: breadth_signal = "C"
        elif breadth_score >= 2: breadth_signal = "D"
        else:                    breadth_signal = "E"

        price = nifty_c[i]
        above_50_price = price > nifty_ema50[i]
        bank_val = bank_c[i]
        bank_ok = (bank_val > bank_ema50[i]) if not np.isnan(bank_val) else True

        if above_50_price and bank_ok and vix < 15:
            price_signal = "A"
        elif above_50_price and vix < 20:
            price_signal = "B"
        elif above_50_price:
            price_signal = "C"
        else:
            price_signal = "D"

        b_idx = regime_order.index(breadth_signal)
        p_idx = regime_order.index(price_signal)
        combined_idx = min(b_idx, p_idx)
        base_idx = max(b_idx - 1, combined_idx)
        regime = regime_order[base_idx]

        conn.execute("""
            INSERT OR REPLACE INTO market_regime_history
            (date, regime, breadth_score, ad_ratio, above_50_pct,
             new_highs, new_lows, nifty_close, vix)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (date, regime, round(breadth_score, 2), None, round(float(above50), 1),
              0, 0, round(float(price), 2), round(float(vix), 2)))
        rows_written += 1

    conn.commit()
    conn.close()

    log.info(f"Deep-backfilled {rows_written} days into market_regime_history")
    print_stats()


def print_stats() -> None:
    conn = get_connection()
    rows = conn.execute("""
        SELECT regime, COUNT(*) as n FROM market_regime_history
        GROUP BY regime ORDER BY regime
    """).fetchall()
    date_range = conn.execute(
        "SELECT MIN(date), MAX(date) FROM market_regime_history"
    ).fetchone()
    total = conn.execute("SELECT COUNT(*) FROM market_regime_history").fetchone()[0]
    conn.close()

    print("\n" + "=" * 55)
    print("  DEEP HISTORICAL REGIME DISTRIBUTION")
    print(f"  Range: {date_range[0]} to {date_range[1]}  ({total} days)")
    print("=" * 55)
    for r in rows:
        label = REGIME_CONFIG.get(r["regime"], {}).get("name", "?")
        pct = 100 * r["n"] / total if total else 0
        print(f"  {r['regime']} ({label:<14}) {r['n']:>5} days  ({pct:5.1f}%)")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    if args.stats:
        print_stats()
    else:
        backfill_deep()
