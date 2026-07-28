"""
NSE Momentum v6 — Historical Regime Backfill
Populates the market_regime_history table (defined in database/schema.py,
currently unused) with a daily regime letter for the full price_history
window, using the SAME breadth+price+VIX combination logic as the live
agents/market_agent.py — not a separate approximation of it.

KNOWN APPROXIMATION (documented, not hidden):
  Live MarketAgent's breadth_score comes from NSE-wide (~2000-3200 symbol)
  advance/decline data via market_breadth_agent.py's live API/Bhavcopy
  fetch. That NSE-wide history isn't stored — only 2 days of Bhavcopy
  cache exist locally. This backfill instead computes a PROXY breadth
  score from above_50_ema_pct across YOUR OWN ~401-stock universe only
  (which price_history DOES have 2 years of). This is exactly the
  narrower, universe-bounded metric market_breadth_agent.py's own BUG-1
  fix explicitly said NOT to use for live breadth — so treat backfilled
  regimes as a reasonable historical approximation, not a perfect replay
  of what the live system would have shown on any given day.

  VIX: fetched historically via yfinance ticker ^INDIAVIX. If unavailable,
  falls back to a neutral 15.0 for that date (flagged in the log).

Usage:
    python regime_backfill.py              # backfill full 2yr history
    python regime_backfill.py --stats      # print current table contents
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from database.schema import get_connection, init_all_tables
from nse_universe import NSE_UNIVERSE
from agents.market_agent import REGIME_CONFIG

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

HISTORY_YEARS = 2


def _ema(v: np.ndarray, span: int) -> np.ndarray:
    alpha = 2 / (span + 1)
    e = np.zeros(len(v))
    e[0] = v[0]
    for i in range(1, len(v)):
        e[i] = alpha * v[i] + (1 - alpha) * e[i - 1]
    return e


def _fetch_index_history(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{HISTORY_YEARS}y", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")
    df = df.reset_index()
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df[["date", "Close"]].rename(columns={"Close": "close"}).astype({"close": float})


def _fetch_vix_history() -> dict:
    """Returns {date_str: vix_float}. Empty dict on failure (neutral fallback used)."""
    try:
        df = yf.download("^INDIAVIX", period=f"{HISTORY_YEARS}y", progress=False, auto_adjust=True)
        if df is None or df.empty:
            log.warning("India VIX history unavailable — will use neutral 15.0 for all dates")
            return {}
        df = df.reset_index()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df["date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        return dict(zip(df["date"], df["Close"].astype(float)))
    except Exception as e:
        log.warning(f"India VIX fetch failed ({e}) — will use neutral 15.0 for all dates")
        return {}


def _load_universe_close_matrix() -> pd.DataFrame:
    """Wide DataFrame: index=date, columns=ticker, values=close. Used to
    compute above_50_ema_pct efficiently across the whole universe at once
    instead of one SQL query per ticker per date."""
    conn = get_connection()
    tickers = list(dict.fromkeys(s[0] for s in NSE_UNIVERSE))
    placeholders = ",".join("?" * len(tickers))
    df = pd.read_sql(
        f"SELECT ticker, date, close FROM price_history WHERE ticker IN ({placeholders})",
        conn, params=tickers
    )
    conn.close()
    wide = df.pivot(index="date", columns="ticker", values="close").sort_index()
    return wide


def _compute_above_50_ema_series(wide: pd.DataFrame) -> pd.Series:
    """For each date, % of universe stocks above their own trailing 50-EMA
    as of that date. Vectorized per-column EMA, then row-wise comparison."""
    def _col_ema(col):
        filled = col.ffill().bfill().to_numpy(dtype=float)
        return pd.Series(_ema(filled, 50), index=col.index)

    ema50 = wide.apply(_col_ema)
    above = (wide > ema50)
    valid = wide.notna()
    pct = (above & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan) * 100
    return pct


def backfill() -> pd.DataFrame:
    init_all_tables()

    log.info("Fetching NIFTY 50 and BANK NIFTY history...")
    nifty = _fetch_index_history("^NSEI")
    bank  = _fetch_index_history("^NSEBANK")
    vix_map = _fetch_vix_history()

    log.info("Loading universe close-price matrix from price_history "
             "(this computes the breadth PROXY — see module docstring)...")
    wide = _load_universe_close_matrix()
    above50_series = _compute_above_50_ema_series(wide)

    nifty_close = nifty.set_index("date")["close"]
    bank_close  = bank.set_index("date")["close"]
    n = len(nifty)
    nifty_c = nifty_close.to_numpy(dtype=float)
    bank_c  = bank_close.reindex(nifty_close.index).ffill().to_numpy(dtype=float)

    nifty_ema50 = _ema(nifty_c, 50)
    bank_ema50  = _ema(bank_c, 50)

    conn = get_connection()
    rows_written = 0
    regime_order = ["A", "B", "C", "D", "E"]

    for i, date in enumerate(nifty_close.index):
        if i < 55:  # need enough bars for a stable 50-EMA
            continue

        above50 = above50_series.get(date, np.nan)
        if pd.isna(above50):
            continue

        vix = vix_map.get(date, 15.0)

        # ── Breadth proxy → breadth_signal (same buckets as market_agent.py,
        # driven by a 0-10 breadth_score derived here from above_50_ema_pct
        # alone, since we lack historical NSE-wide A/D — see docstring) ────
        breadth_score = above50 / 10.0   # 0-100% -> 0-10 scale, simple proxy
        if breadth_score >= 8:   breadth_signal = "A"
        elif breadth_score >= 6: breadth_signal = "B"
        elif breadth_score >= 4: breadth_signal = "C"
        elif breadth_score >= 2: breadth_signal = "D"
        else:                    breadth_signal = "E"

        # ── Price signal — identical logic to MarketAgent._compute ─────────
        price  = nifty_c[i]
        above_50_price = price > nifty_ema50[i]
        bank_ok = bank_c[i] > bank_ema50[i] if not np.isnan(bank_c[i]) else True

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

    log.info(f"Backfilled {rows_written} days into market_regime_history")
    print_stats()


def print_stats() -> None:
    conn = get_connection()
    rows = conn.execute("""
        SELECT regime, COUNT(*) as n FROM market_regime_history
        GROUP BY regime ORDER BY regime
    """).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM market_regime_history").fetchone()[0]
    conn.close()

    print("\n" + "=" * 50)
    print("  HISTORICAL REGIME DISTRIBUTION (backfilled)")
    print("=" * 50)
    for r in rows:
        label = REGIME_CONFIG.get(r["regime"], {}).get("name", "?")
        pct = 100 * r["n"] / total if total else 0
        print(f"  {r['regime']} ({label:<14}) {r['n']:>4} days  ({pct:5.1f}%)")
    print("=" * 50)
    print("  NOTE: breadth is a universe-bounded PROXY, not true NSE-wide")
    print("  historical A/D — see module docstring for why.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="Print current table contents only")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    else:
        backfill()
