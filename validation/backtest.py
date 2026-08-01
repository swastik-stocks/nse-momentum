"""
NSE Momentum v6.0 — Backtesting Engine
Scans local price_history for all historical pattern occurrences,
forward-tests each one, and fills pattern_statistics with real win rates.

v6 CHANGE — Window-cap bug fix:
  Previously every test window was capped to the last 120 bars
  (`df.iloc[max(0, i-120):i]`). This silently made `above_200` — and
  therefore the entire High Base pattern, which requires it — permanently
  unreachable, since PatternAgent's `n >= 200` check could never be true
  with a 120-bar cap. High Base works correctly in the LIVE scanner
  (orchestrator.py passes full 2yr history), so this was a backtest-only
  bug, not a pattern-detection bug. Fixed by using full history up to
  each test point instead of a fixed cap — matches live behaviour exactly.

  Separately: DEFAULT_WEIGHTS/PATTERN_EXPECTANCY in pattern_agent.py were
  set from a v4.0-era backtest where ALL 19 patterns competed for the
  `max(detections, key=weight)` selection under 2023-era weights (Flat
  Base=10, near the bottom of 19). Those old counts measured "how often
  this pattern WON against 18 competitors," not "how often this shape
  occurs" — not a stable, comparable baseline against today's 5-pattern
  weight table. Don't try to reconcile old vs new counts; treat every
  fresh run of this script as the only trustworthy source going forward.


  Previously every forward-test win/loss was computed on GROSS price move
  only — no STT, stamp duty, exchange charges, GST, or DP charges were
  applied. For low-edge patterns already close to breakeven (High Base
  +0.09%, Volume Expansion 0.00% per PATTERN_EXPECTANCY in pattern_agent.py)
  that gap matters: real NSE delivery round-trip costs run ~0.24-0.26% of
  notional, which can erase or reverse a "positive" gross edge that thin.

  Costs are itemized below using the same structure verified against a
  live NSE delivery cost model (STT bilateral, stamp duty buy-only,
  exchange transaction charges, SEBI fees, GST on brokerage+exchange
  charges). DP charges are a flat per-scrip-per-sell-day rupee amount
  (not a %), so they're expressed here as an assumed-trade-value drag —
  see ASSUMED_TRADE_VALUE_INR below; this is a modeling approximation
  since this backtest works in % returns, not actual position sizing.

  win_rate / avg_r / profit_factor in pattern_statistics are now computed
  on NET (post-cost) returns, not gross. Both gross and net win rates are
  printed side-by-side in print_stats() so the cost impact stays visible
  rather than silently baked in.

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
# Win = NET price rise ≥ 5% within forward window (post-cost, see v6 note above)
WIN_THRESHOLD = 0.05

# ── NSE delivery equity transaction costs (v6) ──────────────────────────────
# Rates current as of this codebase's last update — SEBI/exchange fee
# schedules do change occasionally; recheck against your actual Dhan
# contract note periodically rather than trusting these as permanent.
STT_RATE            = 0.00100   # 0.1% — Securities Transaction Tax, BOTH legs (delivery)
STAMP_DUTY_RATE      = 0.00015   # 0.015% — stamp duty, BUY leg only
EXCHANGE_TXN_RATE    = 0.0000297 # NSE exchange transaction charge, both legs
SEBI_FEE_RATE        = 0.000001  # SEBI turnover fee, both legs (₹10/crore)
GST_RATE             = 0.18      # GST on (brokerage + exchange txn charges)
BROKERAGE_RATE       = 0.0035       # Calibarated against real Yes bank brokerage contract.
                                  # Change this if your actual plan charges delivery brokerage.
DP_CHARGE_INR        = 15.0      # Flat per-scrip, per-sell-day DP charge (approx, varies by DP)
ASSUMED_TRADE_VALUE_INR = 100_000  # Used ONLY to convert the flat DP charge into a % drag,
                                    # since this backtest works in % returns, not position size.
                                    # Adjust to roughly match your real typical position size —
                                    # DP charge as a % shrinks for larger positions and grows for
                                    # smaller ones, so this constant matters more for small trades.


def _buy_cost_pct() -> float:
    return (
        STAMP_DUTY_RATE
        + STT_RATE
        + EXCHANGE_TXN_RATE
        + SEBI_FEE_RATE
        + GST_RATE * (BROKERAGE_RATE + EXCHANGE_TXN_RATE)
    )


def _sell_cost_pct() -> float:
    dp_pct = DP_CHARGE_INR / ASSUMED_TRADE_VALUE_INR
    return (
        STT_RATE
        + EXCHANGE_TXN_RATE
        + SEBI_FEE_RATE
        + GST_RATE * (BROKERAGE_RATE + EXCHANGE_TXN_RATE)
        + dp_pct
    )


ROUND_TRIP_COST_PCT = _buy_cost_pct() + _sell_cost_pct()


def _print_cost_model() -> None:
    print("\n" + "=" * 60)
    print("  TRANSACTION COST MODEL (v6)")
    print("=" * 60)
    print(f"  STT (bilateral)         : {STT_RATE*100:.4f}% x2 legs")
    print(f"  Stamp duty (buy only)   : {STAMP_DUTY_RATE*100:.4f}%")
    print(f"  Exchange txn charge     : {EXCHANGE_TXN_RATE*100:.5f}% x2 legs")
    print(f"  SEBI fee                : {SEBI_FEE_RATE*100:.6f}% x2 legs")
    print(f"  Brokerage               : {BROKERAGE_RATE*100:.4f}% (real Yes Bank contract rate)")
    print(f"  GST                     : {GST_RATE*100:.0f}% on (brokerage + exchange charges)")
    print(f"  DP charge (flat)        : Rs.{DP_CHARGE_INR:.0f}/scrip/sell-day "
          f"(modeled as {DP_CHARGE_INR/ASSUMED_TRADE_VALUE_INR*100:.4f}% "
          f"assuming Rs.{ASSUMED_TRADE_VALUE_INR:,.0f} position)")
    print(f"  ──────────────────────────────────────────────────────")
    print(f"  Round-trip cost         : {ROUND_TRIP_COST_PCT*100:.3f}% of notional")
    print("=" * 60 + "\n")


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


def _forward_test(df: pd.DataFrame, signal_idx: int, entry: float) -> dict:
    """
    Check if price rose >= WIN_THRESHOLD (NET of transaction costs) within
    FORWARD_BARS after signal.

    Returns dict with both gross and net figures so callers can report the
    cost impact rather than only seeing the post-cost number:
        {
            "is_win_gross": bool,  "gross_gain": float,  "gross_r": float,
            "is_win_net":   bool,  "net_gain":   float,  "net_r":   float,
        }
    """
    future = df.iloc[signal_idx+1 : signal_idx+1+FORWARD_BARS]
    if future.empty:
        return {
            "is_win_gross": False, "gross_gain": 0.0, "gross_r": 0.0,
            "is_win_net":   False, "net_gain":   0.0, "net_r":   0.0,
        }

    max_price   = float(future["High"].max())
    min_price   = float(future["Low"].min())
    gross_gain  = (max_price - entry) / entry if entry > 0 else 0.0
    net_gain    = gross_gain - ROUND_TRIP_COST_PCT

    is_win_gross = gross_gain >= WIN_THRESHOLD
    is_win_net   = net_gain   >= WIN_THRESHOLD

    # R-multiple: on a win, use the favorable move; on a loss, use the
    # adverse move (assuming a 5% stop) — same convention as before, just
    # computed on both gross and net now.
    gross_r = round(gross_gain / 0.05, 2) if is_win_gross else round(
        (min_price - entry) / entry / 0.05, 2)
    net_loss_gain = (min_price - entry) / entry - ROUND_TRIP_COST_PCT
    net_r = round(net_gain / 0.05, 2) if is_win_net else round(net_loss_gain / 0.05, 2)

    return {
        "is_win_gross": is_win_gross, "gross_gain": gross_gain, "gross_r": gross_r,
        "is_win_net":   is_win_net,   "net_gain":   net_gain,   "net_r":   net_r,
    }


def backtest_ticker(ticker: str) -> dict:
    """
    Run pattern detection on every 60-bar window in price_history for a ticker.
    Record all occurrences and forward-test results (gross AND net of costs).
    """
    df = _load_price_history(ticker)
    if len(df) < 80:
        return {"ticker": ticker, "signals": 0}

    conn = get_connection()
    pattern_results = {}  # pattern -> list of forward-test result dicts
    total_signals   = 0

    # Slide window: detect pattern, then forward-test
    step = 5  # check every 5 bars to avoid over-counting same setup
    for i in range(60, len(df) - FORWARD_BARS, step):
        # v6 FIX: previously capped to the last 120 bars, which meant `n`
        # inside PatternAgent could never reach 200 — so above_200 (and
        # therefore the entire High Base pattern, which requires it) was
        # UNREACHABLE for the whole backtest, regardless of real price
        # action. High Base works correctly live because orchestrator.py
        # passes full 2yr history to PatternAgent — this backtest now does
        # the same, using all history up to bar i rather than a fixed cap.
        window = df.iloc[:i].copy()
        window.index = range(len(window))

        try:
            pa = PatternAgent(window)
        except Exception:
            continue

        if not pa.pattern:
            continue

        entry = float(pa.entry_high) if pa.entry_high > 0 else float(window["Close"].iloc[-1])
        result = _forward_test(df, i, entry)

        pat = pa.pattern
        if pat not in pattern_results:
            pattern_results[pat] = []
        pattern_results[pat].append(result)
        total_signals += 1

        # Store occurrence. NOTE: regime is still hardcoded "C" here — that
        # is a SEPARATE known issue (tracked next, not fixed in this change)
        # and out of scope for the transaction-cost fix in this edit.
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
    Aggregate pattern win rates from all ticker backtests and store in
    pattern_statistics. Stored win_rate/avg_r/profit_factor are NET of
    transaction costs (v6) — gross figures are computed here too and
    printed in print_stats(), but not persisted, to keep the existing
    table schema unchanged.
    """
    combined = {}  # pattern -> list of result dicts
    for res in all_results:
        for pat, outcomes in res.get("results", {}).items():
            if pat not in combined:
                combined[pat] = []
            combined[pat].extend(outcomes)

    conn = get_connection()
    today = datetime.today().strftime("%Y-%m-%d")

    gross_summary = {}  # pattern -> (gross_win_rate, gross_avg_r) — for print_stats only

    for pat, outcomes in combined.items():
        if not outcomes:
            continue

        # NET figures — these are what get persisted
        net_wins    = sum(1 for o in outcomes if o["is_win_net"])
        net_total_r = sum(o["net_r"] for o in outcomes)
        net_wr      = net_wins / len(outcomes) if outcomes else 0
        net_avg_r   = net_total_r / len(outcomes) if outcomes else 0
        net_win_r   = sum(o["net_r"] for o in outcomes if o["is_win_net"])
        net_los_r   = abs(sum(o["net_r"] for o in outcomes if not o["is_win_net"]))
        net_pf      = net_win_r / net_los_r if net_los_r > 0 else 2.0
        net_losses  = len(outcomes) - net_wins

        # GROSS figures — printed only, not persisted (see docstring)
        gross_wins  = sum(1 for o in outcomes if o["is_win_gross"])
        gross_wr    = gross_wins / len(outcomes) if outcomes else 0
        gross_avg_r = sum(o["gross_r"] for o in outcomes) / len(outcomes) if outcomes else 0
        gross_summary[pat] = (gross_wr, gross_avg_r)

        conn.execute("""
            INSERT OR REPLACE INTO pattern_statistics
            (pattern, total_signals, wins, losses, total_r, win_rate, avg_r,
             profit_factor, last_updated)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (pat, len(outcomes), net_wins, net_losses, net_total_r,
              net_wr, net_avg_r, net_pf, today))

    conn.commit()
    conn.close()

    # Stash gross figures on the module for print_stats() to read this run
    # (not persisted to DB — recomputed each run from raw outcomes above).
    global _LAST_GROSS_SUMMARY
    _LAST_GROSS_SUMMARY = gross_summary

    log.info(f"Pattern statistics updated for {len(combined)} patterns "
             f"(NET of {ROUND_TRIP_COST_PCT*100:.3f}% round-trip cost)")


_LAST_GROSS_SUMMARY: dict = {}


def print_stats() -> None:
    """Print the pattern_statistics table — NET figures from DB, plus GROSS
    figures alongside (from the most recent run only) so the cost impact
    is visible rather than hidden."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT pattern, total_signals, win_rate, avg_r, profit_factor, last_updated
        FROM pattern_statistics
        ORDER BY avg_r DESC
    """).fetchall()
    conn.close()

    _print_cost_model()

    print("\n" + "="*100)
    print("  BACKTESTED PATTERN STATISTICS  (NET = post transaction costs, GROSS = pre-cost)")
    print("="*100)
    print(f"  {'PATTERN':<22} {'N':>6} {'NET WIN%':>9} {'GROSS WIN%':>11} "
          f"{'NET R':>7} {'GROSS R':>8} {'PF':>5}  {'UPDATED'}")
    print("  " + "-"*95)
    for r in rows:
        gross = _LAST_GROSS_SUMMARY.get(r["pattern"])
        gross_wr_str  = f"{gross[0]*100:>10.0f}%" if gross else f"{'—':>11}"
        gross_r_str   = f"{gross[1]:>8.2f}R" if gross else f"{'—':>8}"
        print(f"  {r['pattern']:<22} {r['total_signals']:>6} "
              f"{r['win_rate']*100:>8.0f}% {gross_wr_str} "
              f"{r['avg_r']:>6.2f}R {gross_r_str} {r['profit_factor']:>5.1f}  "
              f"{r['last_updated'] or '—'}")
    print("="*100)
    print("  NOTE: GROSS columns only populate for patterns backtested in the CURRENT run")
    print("  (not persisted to DB). Run a fresh backtest to see gross-vs-net for all patterns.")
    print("="*100 + "\n")


def run_backtest(tickers: list = None) -> None:
    """Run full backtest. Defaults to entire NSE_UNIVERSE."""
    init_all_tables()
    _print_cost_model()

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
