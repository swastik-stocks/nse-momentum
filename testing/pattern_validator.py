"""
NSE Momentum v5.2 — Pattern Validator
testing/pattern_validator.py

PURPOSE:
  Proves which of the 19 patterns have genuine positive expectancy
  on NSE data (2023-2026). Run this ONCE after deploying v5.2.
  Results tell you which patterns to keep, which to prune.

HOW IT WORKS:
  For each pattern x each stock in universe:
    1. Detect all historical signals using YOUR existing PatternAgent logic
    2. Simulate next-day open entry (realistic -- mirrors 10am entry)
    3. Walk forward up to HOLD_DAYS bars:
       - Exit on stop (dynamic: ATR x 1.5, min 2%)
       - Exit on target (stop x MIN_RR)
       - Exit on timeout (close at day HOLD_DAYS)
    4. Record return, outcome, hold days, max adverse excursion

OUTPUT:
  Console: ranked table of all 19 patterns by expectancy
  File:    data/pattern_validation_results.csv
  File:    data/pattern_validation_summary.json

USAGE:
  python testing/pattern_validator.py

RUNTIME:
  ~10-15 minutes for 404 stocks x 2 years.
"""

import sys
import json
import sqlite3
import logging
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from loguru import logger as log
    log.remove()
    log.add(sys.stderr, level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
except ImportError:
    log = logging.getLogger(__name__)

# -- Repo path setup ----------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents"))

from agents.pattern_agent import PatternAgent, DEFAULT_WEIGHTS

# -- Config -------------------------------------------------------------------
DB_PATH      = ROOT / "data" / "momentum_v4.db"
START_DATE   = "2023-01-01"
END_DATE     = "2026-06-01"
HOLD_DAYS    = 10
MIN_RR       = 2.5
ATR_MULT     = 1.5
MIN_STOP_PCT = 2.0
OUTPUT_DIR   = ROOT / "data"

ALL_PATTERNS = list(DEFAULT_WEIGHTS.keys())


# -- Verdict ------------------------------------------------------------------

def _verdict(win_rate, expectancy, n):
    if n < 20:
        return "THIN  (< 20 signals)"
    if win_rate > 45 and expectancy > 0.5:
        return "KEEP  (positive edge)"
    if win_rate > 38 and expectancy > 0:
        return "WATCH (marginal edge)"
    return "PRUNE (negative expectancy)"


# -- Data loading -------------------------------------------------------------

def load_universe():
    from nse_universe import NSE_UNIVERSE
    return list(NSE_UNIVERSE)


def load_price_history(ticker):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            """
            SELECT date, open, high, low, close, volume
            FROM price_history
            WHERE ticker = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
            """,
            conn, params=(ticker, START_DATE, END_DATE)
        )
        conn.close()
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df = df.astype(float)
        return df
    except Exception as e:
        log.debug(f"DB load failed for {ticker}: {e}")
        return pd.DataFrame()


# -- Signal detection ---------------------------------------------------------

def detect_signals(df, pattern_name):
    signals = []
    min_bars = 60
    for i in range(min_bars, len(df) - HOLD_DAYS - 1):
        window = df.iloc[:i + 1]
        try:
            pa = PatternAgent(window)
            if pa.pattern == pattern_name:
                signals.append(i)
        except Exception:
            continue
    return signals


# -- Trade simulation ---------------------------------------------------------

def simulate_trade(df, signal_idx):
    entry_idx = signal_idx + 1
    if entry_idx >= len(df):
        return None

    entry = float(df.iloc[entry_idx]["Open"])
    if entry <= 0:
        return None

    window = df.iloc[:signal_idx + 1]
    try:
        pa = PatternAgent(window)
        atr_pct = pa.get_atr_pct()
    except Exception:
        atr_pct = 0.0

    stop_pct   = max(atr_pct * ATR_MULT, MIN_STOP_PCT)
    stop       = entry * (1 - stop_pct / 100)
    reward_pct = stop_pct * MIN_RR
    target     = entry * (1 + reward_pct / 100)

    max_adverse = 0.0
    for j in range(entry_idx + 1, min(entry_idx + HOLD_DAYS + 1, len(df))):
        bar_low   = float(df.iloc[j]["Low"])
        bar_high  = float(df.iloc[j]["High"])

        adverse = (entry - bar_low) / entry * 100
        max_adverse = max(max_adverse, adverse)

        if bar_low <= stop:
            return {
                "entry":       round(entry,   2),
                "exit":        round(stop,    2),
                "ret_pct":     round((stop - entry) / entry * 100, 2),
                "hold_days":   j - entry_idx,
                "outcome":     "STOP",
                "max_adverse": round(max_adverse, 2),
                "stop_pct":    round(stop_pct,    2),
            }
        if bar_high >= target:
            return {
                "entry":       round(entry,  2),
                "exit":        round(target, 2),
                "ret_pct":     round((target - entry) / entry * 100, 2),
                "hold_days":   j - entry_idx,
                "outcome":     "TARGET",
                "max_adverse": round(max_adverse, 2),
                "stop_pct":    round(stop_pct,    2),
            }

    exit_idx   = min(entry_idx + HOLD_DAYS, len(df) - 1)
    exit_price = float(df.iloc[exit_idx]["Close"])
    return {
        "entry":       round(entry,      2),
        "exit":        round(exit_price, 2),
        "ret_pct":     round((exit_price - entry) / entry * 100, 2),
        "hold_days":   HOLD_DAYS,
        "outcome":     "TIMEOUT",
        "max_adverse": round(max_adverse, 2),
        "stop_pct":    round(stop_pct,    2),
    }


# -- Per-pattern analysis -----------------------------------------------------

def validate_pattern(pattern_name, universe):
    all_trades = []
    for item in universe:
        ticker = item[0]
        df = load_price_history(ticker)
        if df.empty or len(df) < 80:
            continue
        signals = detect_signals(df, pattern_name)
        for sig_idx in signals:
            trade = simulate_trade(df, sig_idx)
            if trade:
                trade["ticker"]  = ticker
                trade["pattern"] = pattern_name
                trade["date"]    = str(df.index[sig_idx].date())
                all_trades.append(trade)
    return pd.DataFrame(all_trades)


def summarise(df, pattern_name, weight):
    if df.empty:
        return {
            "pattern": pattern_name, "weight": weight,
            "n": 0, "win_rate": 0, "expectancy": 0,
            "avg_win": 0, "avg_loss": 0, "profit_factor": 0,
            "avg_hold": 0, "avg_max_adverse": 0,
            "pct_stop": 0, "pct_target": 0, "pct_timeout": 0,
            "verdict": "THIN  (no signals detected)",
        }

    wins   = df[df["ret_pct"] > 0]
    losses = df[df["ret_pct"] <= 0]
    n      = len(df)
    wr     = len(wins) / n * 100
    exp    = df["ret_pct"].mean()

    pf = (abs(wins["ret_pct"].sum() / losses["ret_pct"].sum())
          if len(losses) > 0 and losses["ret_pct"].sum() != 0 else 99.0)

    outcomes = df["outcome"].value_counts(normalize=True) * 100

    return {
        "pattern":         pattern_name,
        "weight":          weight,
        "n":               n,
        "win_rate":        round(wr,  1),
        "expectancy":      round(exp, 2),
        "avg_win":         round(wins["ret_pct"].mean(),   2) if len(wins)   else 0,
        "avg_loss":        round(losses["ret_pct"].mean(), 2) if len(losses) else 0,
        "profit_factor":   round(pf, 2),
        "avg_hold":        round(df["hold_days"].mean(),      1),
        "avg_max_adverse": round(df["max_adverse"].mean(),    2),
        "pct_stop":        round(outcomes.get("STOP",    0),  1),
        "pct_target":      round(outcomes.get("TARGET",  0),  1),
        "pct_timeout":     round(outcomes.get("TIMEOUT", 0),  1),
        "verdict":         _verdict(wr, exp, n),
    }


def print_summary(s):
    sep = "-" * 62
    print(f"\n{sep}")
    print(f"  {s['pattern']}  (default weight: {s['weight']})")
    print(f"  {s['verdict']}")
    print(sep)
    if s["n"] == 0:
        print("  No signals detected in 2023-2026 period.")
        return
    print(f"  Signals:         {s['n']}")
    print(f"  Win rate:        {s['win_rate']:.1f}%")
    print(f"  Expectancy:      {s['expectancy']:+.2f}% per trade")
    print(f"  Avg win:         +{s['avg_win']:.2f}%")
    print(f"  Avg loss:         {s['avg_loss']:.2f}%")
    print(f"  Profit factor:   {s['profit_factor']:.2f}x")
    print(f"  Avg hold:        {s['avg_hold']:.1f} days")
    print(f"  Avg max adverse: {s['avg_max_adverse']:.2f}%")
    print(f"  Stop/Target/TO:  {s['pct_stop']:.0f}% / {s['pct_target']:.0f}% / {s['pct_timeout']:.0f}%")


# -- Main ---------------------------------------------------------------------

def main():
    log.info("NSE Momentum v5.2 - Pattern Validator")
    log.info(f"Period: {START_DATE} to {END_DATE} | Hold: {HOLD_DAYS}d | Min R:R: {MIN_RR}x")

    universe = load_universe()
    log.info(f"Universe: {len(universe)} stocks")

    if not DB_PATH.exists():
        log.error(f"Database not found at {DB_PATH}")
        log.error("Run the evening scanner first to populate price_history.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)

    all_summaries   = []
    all_trades_list = []

    for i, pattern_name in enumerate(ALL_PATTERNS, 1):
        log.info(f"[{i:02d}/{len(ALL_PATTERNS)}] Validating: {pattern_name}")
        trades_df = validate_pattern(pattern_name, universe)
        summary   = summarise(trades_df, pattern_name,
                              DEFAULT_WEIGHTS.get(pattern_name, 10))
        print_summary(summary)
        all_summaries.append(summary)
        if not trades_df.empty:
            all_trades_list.append(trades_df)

    summary_df = pd.DataFrame(all_summaries).sort_values("expectancy", ascending=False)

    trades_out  = OUTPUT_DIR / "pattern_validation_results.csv"
    summary_out = OUTPUT_DIR / "pattern_validation_summary.json"

    if all_trades_list:
        combined = pd.concat(all_trades_list, ignore_index=True)
        combined.to_csv(trades_out, index=False)
        log.info(f"Trade-level results saved to {trades_out}")

    summary_df.to_json(summary_out, orient="records", indent=2)
    log.info(f"Summary saved to {summary_out}")

    print("\n" + "=" * 70)
    print("  FINAL RANKING - by expectancy")
    print("=" * 70)
    print(f"  {'Pattern':<25} {'N':>5} {'WR%':>6} {'EXP%':>7} {'PF':>5}  Verdict")
    print("  " + "-" * 66)
    for _, row in summary_df.iterrows():
        print(
            f"  {row['pattern']:<25} {row['n']:>5} "
            f"{row['win_rate']:>5.1f}% {row['expectancy']:>+6.2f}% "
            f"{row['profit_factor']:>5.2f}x  {row['verdict']}"
        )

    keep  = [r for r in all_summaries if "KEEP"  in r["verdict"]]
    watch = [r for r in all_summaries if "WATCH" in r["verdict"]]
    prune = [r for r in all_summaries if "PRUNE" in r["verdict"]]
    thin  = [r for r in all_summaries if "THIN"  in r["verdict"]]

    print(f"\n  KEEP: {len(keep)}  |  WATCH: {len(watch)}  "
          f"|  PRUNE: {len(prune)}  |  THIN: {len(thin)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
