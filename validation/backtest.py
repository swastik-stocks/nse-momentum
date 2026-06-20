"""
NSE Momentum v4.0 — Backtesting Engine
Scans local price_history for all historical pattern occurrences,
forward-tests each one, and fills pattern_statistics with real win rates.

Usage:
    python validation/backtest.py          # run full backtest on all stocks
    python validation/backtest.py RELIANCE # single stock
    python validation/backtest.py --stats  # print current stats table

Run monthly after sufficient price_history has accumulated.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.schema import get_connection, init_all_tables
from agents.pattern_agent import PatternAgent, DEFAULT_WEIGHTS
from nse_universe import NSE_UNIVERSE

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Forward-test window: 20 bars after pattern detection
FORWARD_BARS = 20
# Win = price rises ≥ 5% within forward window
WIN_THRESHOLD = 0.05


def _load_price_history(ticker: str) -> pd.DataFrame:
    """Load full price history for a ticker from local DB."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM price_history "
        "WHERE ticker=? ORDER BY date ASC",
        conn, params=(ticker,)
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


def _forward_test(df: pd.DataFrame, signal_idx: int, entry: float) -> tuple[bool, float]:
    """
    Check if price rose ≥ WIN_THRESHOLD within FORWARD_BARS after signal.
    Returns (is_win, r_multiple).
    """
    future = df.iloc[signal_idx+1 : signal_idx+1+FORWARD_BARS]
    if future.empty:
        return False, 0.0

    max_price = float(future["High"].max())
    gain      = (max_price - entry) / entry if entry > 0 else 0
    is_win    = gain >= WIN_THRESHOLD

    # Approximate R-multiple (assuming 5% stop)
    r_mult = round(gain / 0.05, 2) if is_win else round(
        (float(future["Low"].min()) - entry) / entry / 0.05, 2
    )
    return is_win, r_mult


def backtest_ticker(ticker: str) -> dict:
    """
    Run pattern detection on every 60-bar window in price_history for a ticker.
    Record all occurrences and forward-test results.
    """
    df = _load_price_history(ticker)
    if len(df) < 80:
        return {"ticker": ticker, "signals": 0}

    conn = get_connection()
    pattern_results = {}  # pattern -> list of (win, r_mult)
    total_signals   = 0

    # Slide window: detect pattern, then forward-test
    step = 5  # check every 5 bars to avoid over-counting same setup
    for i in range(60, len(df) - FORWARD_BARS, step):
        window = df.iloc[max(0, i-120) : i].copy()
        window.index = range(len(window))

        try:
            pa = PatternAgent(window)
        except Exception:
            continue

        if not pa.pattern:
            continue

        entry = float(pa.entry_high) if pa.entry_high > 0 else float(window["Close"].iloc[-1])
        is_win, r_mult = _forward_test(df, i, entry)

        pat = pa.pattern
        if pat not in pattern_results:
            pattern_results[pat] = []
        pattern_results[pat].append((is_win, r_mult))
        total_signals += 1

        # Store occurrence
        conn.execute("""
            INSERT OR IGNORE INTO pattern_occurrences
            (ticker, date, pattern, breakout_level, score, regime)
            VALUES (?,?,?,?,?,?)
        """, (ticker, df.index[i].strftime("%Y-%m-%d"), pat, entry, pa.raw_score, "C"))

    conn.commit()
    conn.close()
    return {"ticker": ticker, "signals": total_signals, "results": pattern_results}


def aggregate_statistics(all_results: list) -> None:
    """
    Aggregate pattern win rates from all ticker backtests
    and store in pattern_statistics table.
    """
    combined = {}  # pattern -> list of (win, r_mult)
    for res in all_results:
        for pat, outcomes in res.get("results", {}).items():
            if pat not in combined:
                combined[pat] = []
            combined[pat].extend(outcomes)

    conn = get_connection()
    today = datetime.today().strftime("%Y-%m-%d")

    for pat, outcomes in combined.items():
        if not outcomes:
            continue
        wins   = sum(1 for w, r in outcomes if w)
        losses = len(outcomes) - wins
        total_r = sum(r for w, r in outcomes)
        wr     = wins / len(outcomes) if outcomes else 0
        avg_r  = total_r / len(outcomes) if outcomes else 0
        win_r  = sum(r for w, r in outcomes if w)
        los_r  = abs(sum(r for w, r in outcomes if not w))
        pf     = win_r / los_r if los_r > 0 else 2.0

        conn.execute("""
            INSERT OR REPLACE INTO pattern_statistics
            (pattern, total_signals, wins, losses, total_r, win_rate, avg_r,
             profit_factor, last_updated)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (pat, len(outcomes), wins, losses, total_r, wr, avg_r, pf, today))

    conn.commit()
    conn.close()
    log.info(f"Pattern statistics updated for {len(combined)} patterns")


def print_stats() -> None:
    """Print the pattern_statistics table."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT pattern, total_signals, win_rate, avg_r, profit_factor, last_updated
        FROM pattern_statistics
        ORDER BY avg_r DESC
    """).fetchall()
    conn.close()

    print("\n" + "="*75)
    print("  BACKTESTED PATTERN STATISTICS")
    print("="*75)
    print(f"  {'PATTERN':<25} {'N':>6} {'WIN%':>6} {'AVG R':>7} {'PF':>5}  {'UPDATED'}")
    print("  " + "-"*68)
    for r in rows:
        print(f"  {r['pattern']:<25} {r['total_signals']:>6} "
              f"{r['win_rate']*100:>5.0f}% {r['avg_r']:>7.2f}R {r['profit_factor']:>5.1f}  "
              f"{r['last_updated'] or '—'}")
    print("="*75 + "\n")


def run_backtest(tickers: list = None) -> None:
    """Run full backtest. Defaults to entire NSE_UNIVERSE."""
    init_all_tables()

    if tickers is None:
        tickers = list(dict.fromkeys(s[0] for s in NSE_UNIVERSE))

    log.info(f"Running backtest on {len(tickers)} stocks...")
    all_results = []
    total_signals = 0

    for i, ticker in enumerate(tickers, 1):
        res = backtest_ticker(ticker)
        all_results.append(res)
        total_signals += res["signals"]
        if i % 50 == 0:
            log.info(f"  {i}/{len(tickers)} | {total_signals:,} signals detected")

    aggregate_statistics(all_results)
    log.info(f"Backtest complete: {total_signals:,} total signals across {len(tickers)} stocks")
    print_stats()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--stats":
            print_stats()
        else:
            res = backtest_ticker(arg.upper())
            aggregate_statistics([res])
            print(f"Backtested {arg}: {res['signals']} signals")
            print_stats()
    else:
        run_backtest()
