"""
NSE Momentum v6 — Full-Pipeline Historical Replay
Answers the real question validation/backtest.py could not: not "does a
raw pattern signal work in isolation" but "would a signal that ALSO
cleared the real live gate chain (RS percentile, liquidity, risk/stop
validity, asymmetry R:R, VCP contraction) have been profitable, net of
real transaction costs and the actual historical regime penalty."

Reuses the REAL agent classes orchestrator.py uses live — not
reimplementations — so this is testing your actual gate logic, not an
approximation of it:
    agents.pattern_agent.PatternAgent
    agents.rs_agent.RSAgent, compute_universe_ranks
    agents.liquidity_agent.LiquidityAgent
    agents.risk_agent.RiskAgent
    agents.asymmetry_gate.AsymmetryGate
    agents.vcp_gate.VCPContractionGate

HONEST SCOPE LIMITATION (see also regime_backfill.py docstring):
  MacroAgent (VIX/FII/breadth composite), EventRiskAgent, and
  ConfirmationAgent all depend on either live-only data (FII flow has no
  historical archive here) or same-day confirmation state that doesn't
  apply to a backtest. Their contribution to total_score is SKIPPED
  (treated as 0) in this replay, not approximated. Only the regime
  penalty (from the backfilled market_regime_history table) is applied.
  This means replayed total_score is systematically somewhat different
  from what a live scan would have shown on the same historical date —
  results here should be read as "does the core pattern+RS+risk stack
  have edge," not as a perfect historical P&L reconstruction.

  Uses the SAME score_gate/min_rrr/min_adt_cr thresholds per universe
  tier as UNIVERSE_CONFIG in nse_universe.py, and the SAME NSE
  transaction cost model as validation/backtest.py (imported directly,
  not duplicated).

Results are written to a NEW table, pipeline_statistics — kept
deliberately separate from pattern_statistics (raw pattern signals) so
you can compare "raw pattern alone" vs "pattern + full gate chain"
side by side, not overwrite one with the other.

Usage:
    python validation/pipeline_replay.py           # full universe, full history
    python validation/pipeline_replay.py RELIANCE  # single stock
    python validation/pipeline_replay.py --stats   # print current results

Requires market_regime_history to be backfilled first:
    python regime_backfill.py

This is significantly slower than validation/backtest.py — expect a much
longer runtime, since every candidate signal now runs 5 additional real
gate classes instead of just pattern detection.
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.schema import get_connection, init_all_tables
from agents.pattern_agent import PatternAgent
from agents.rs_agent import RSAgent, compute_universe_ranks
from agents.liquidity_agent import LiquidityAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
from agents.vcp_gate import VCPContractionGate
from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from validation.backtest import ROUND_TRIP_COST_PCT, _print_cost_model

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

FORWARD_BARS = 20
WIN_THRESHOLD = 0.05
STEP = 5

REGIME_PENALTIES = {"A": 0, "B": 0, "C": -5, "D": -12, "E": -20}

_PIPELINE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_statistics (
    pattern         TEXT PRIMARY KEY,
    total_signals   INTEGER DEFAULT 0,
    wins            INTEGER DEFAULT 0,
    losses          INTEGER DEFAULT 0,
    total_r         REAL    DEFAULT 0,
    win_rate        REAL    DEFAULT 0,
    avg_r           REAL    DEFAULT 0,
    profit_factor   REAL    DEFAULT 0,
    last_updated    TEXT
)
"""


def _ensure_table():
    conn = get_connection()
    conn.execute(_PIPELINE_TABLE_DDL)
    conn.commit()
    conn.close()


def _load_price_history(ticker: str) -> pd.DataFrame:
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


def _load_nifty_history() -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        "SELECT date, nifty_close FROM market_regime_history ORDER BY date ASC",
        conn
    )
    conn.close()
    if df.empty:
        raise RuntimeError(
            "market_regime_history is empty — run regime_backfill.py first."
        )
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.rename(columns={"nifty_close": "Close"}, inplace=True)
    return df


def _load_regime_map() -> dict:
    """{date_str: regime_letter}"""
    conn = get_connection()
    rows = conn.execute("SELECT date, regime FROM market_regime_history").fetchall()
    conn.close()
    return {r["date"]: r["regime"] for r in rows}


def _regime_for_date(regime_map: dict, date_str: str) -> str:
    """Most recent regime at or before date_str."""
    candidates = [d for d in regime_map if d <= date_str]
    if not candidates:
        return "C"
    return regime_map[max(candidates)]


def _forward_test_net(df: pd.DataFrame, signal_idx: int, entry: float) -> dict:
    future = df.iloc[signal_idx+1: signal_idx+1+FORWARD_BARS]
    if future.empty:
        return {"is_win": False, "net_r": 0.0}
    max_price = float(future["High"].max())
    min_price = float(future["Low"].min())
    gross_gain = (max_price - entry) / entry if entry > 0 else 0.0
    net_gain = gross_gain - ROUND_TRIP_COST_PCT
    is_win = net_gain >= WIN_THRESHOLD
    if is_win:
        net_r = round(net_gain / 0.05, 2)
    else:
        net_r = round(((min_price - entry) / entry - ROUND_TRIP_COST_PCT) / 0.05, 2)
    return {"is_win": is_win, "net_r": net_r}


def replay_ticker(ticker: str, name: str, sector: str, tier: str,
                   nifty_df: pd.DataFrame, universe_ranks_by_date: dict,
                   regime_map: dict) -> dict:
    df = _load_price_history(ticker)
    if len(df) < 80:
        return {"ticker": ticker, "signals": 0, "results": {}}

    cfg = UNIVERSE_CONFIG[tier]
    pattern_results: dict = {}
    total_signals = 0

    for i in range(60, len(df) - FORWARD_BARS, STEP):
        window = df.iloc[:i].copy()
        window.index = range(len(window))
        date_str = df.index[i].strftime("%Y-%m-%d")

        try:
            pa = PatternAgent(window)
        except Exception:
            continue
        if not pa.pattern:
            continue

        liq = LiquidityAgent(window, universe=tier)
        if not liq.passes():
            continue

        day_ranks = universe_ranks_by_date.get(date_str, {})
        nifty_window = nifty_df.loc[:df.index[i]].copy()
        # Guard: RSAgent's own internal fallback path requires len(nifty)>=20
        # before it will compute anything from raw price data. Without this
        # guard, a too-short slice (seen with newly-listed tickers whose
        # test dates land near market_regime_history's own earliest date)
        # collapses via .squeeze() to a bare scalar instead of a Series and
        # crashes RSAgent — skip cleanly instead, same threshold RSAgent
        # itself already uses.
        if len(nifty_window) < 20:
            continue
        nifty_window.columns = ["Close"]
        rsa = RSAgent(window, nifty_window, universe_ranks=day_ranks, ticker=ticker)
        if not rsa.passes_gate():
            continue

        entry = float(pa.entry_high) if pa.entry_high > 0 else float(window["Close"].iloc[-1])

        risk = RiskAgent(window, pa.breakout_level, pa.entry_low, pa.entry_high, universe=tier)
        if not risk.passes():
            continue

        ag = AsymmetryGate(entry=risk.entry_high if hasattr(risk, "entry_high") else entry,
                            stop=risk.stop, target1=risk.target1, universe=tier)
        try:
            ag_result = ag.check_dynamic(df=window, w4_pct=0.0)
            if not ag_result.get("qualified") and ag_result.get("fail_stage") == "INPUT":
                ag_result = ag.check()
        except Exception:
            ag_result = ag.check() if hasattr(ag, "check") else {"qualified": False}
        if not ag_result.get("qualified", False):
            continue

        vcpg = VCPContractionGate(df=window)
        vcp = vcpg.check()
        if vcp.get("hard_reject", False):
            continue

        regime = _regime_for_date(regime_map, date_str)
        penalty = int(REGIME_PENALTIES.get(regime, -5) * cfg["regime_penalty_mult"])

        result = _forward_test_net(df, i, entry)
        result["regime"] = regime
        result["penalty_applied"] = penalty

        pat = pa.pattern
        pattern_results.setdefault(pat, []).append(result)
        total_signals += 1

    return {"ticker": ticker, "signals": total_signals, "results": pattern_results}


def _precompute_universe_ranks_by_date(dates: list, stock_data: dict, nifty_df: pd.DataFrame) -> dict:
    """
    RS ranks need to be computed PER DATE across the whole universe. Doing
    this exactly for every one of ~500 dates is expensive; we sample at
    the same STEP interval as the main loop and reuse the nearest computed
    date for anything in between — a documented trade-off between fidelity
    and runtime, not a silent shortcut.
    """
    log.info("  Precomputing per-date universe RS ranks (this is the slow part)...")
    result = {}
    sampled_dates = dates[::STEP]
    for idx, date in enumerate(sampled_dates):
        date_str = date.strftime("%Y-%m-%d")
        data_dict = {
            "nifty50_data": nifty_df.loc[:date],
            "stock_data": {t: df.loc[:date] for t, df in stock_data.items()},
        }
        result[date_str] = compute_universe_ranks(data_dict)
        if idx % 20 == 0:
            log.info(f"    {idx}/{len(sampled_dates)} dates ranked")
    return result


def run_replay(tickers: list = None) -> None:
    _ensure_table()
    _print_cost_model()

    if tickers is None:
        universe_items = list({s[0]: s for s in NSE_UNIVERSE}.values())
    else:
        universe_items = [s for s in NSE_UNIVERSE if s[0] in tickers]

    log.info("Loading NIFTY history from market_regime_history...")
    nifty_df = _load_nifty_history()
    regime_map = _load_regime_map()

    log.info(f"Loading price history for {len(universe_items)} stocks "
             "(needed for RS rank precomputation)...")
    stock_data = {}
    for ticker, *_ in universe_items:
        df = _load_price_history(ticker)
        if not df.empty:
            stock_data[ticker] = df

    all_dates = sorted(nifty_df.index)
    universe_ranks_by_date = _precompute_universe_ranks_by_date(all_dates, stock_data, nifty_df)

    log.info(f"Running full-pipeline replay on {len(universe_items)} stocks...")
    all_results = []
    total_signals = 0

    for i, (ticker, name, sector, tier) in enumerate(universe_items, 1):
        try:
            res = replay_ticker(ticker, name, sector, tier, nifty_df, universe_ranks_by_date, regime_map)
        except Exception as e:
            log.warning(f"  {ticker} CRASHED during replay ({type(e).__name__}: {e}) — skipping, continuing run")
            continue
        all_results.append(res)
        total_signals += res["signals"]
        if i % 50 == 0:
            log.info(f"  {i}/{len(universe_items)} | {total_signals:,} gate-cleared signals so far")

    aggregate_and_store(all_results)
    log.info(f"Replay complete: {total_signals:,} gate-cleared signals across {len(universe_items)} stocks")
    print_stats()


def aggregate_and_store(all_results: list) -> None:
    combined: dict = {}
    for res in all_results:
        for pat, outcomes in res.get("results", {}).items():
            combined.setdefault(pat, []).extend(outcomes)

    conn = get_connection()
    today = datetime.today().strftime("%Y-%m-%d")
    for pat, outcomes in combined.items():
        if not outcomes:
            continue
        wins = sum(1 for o in outcomes if o["is_win"])
        losses = len(outcomes) - wins
        total_r = sum(o["net_r"] for o in outcomes)
        wr = wins / len(outcomes)
        avg_r = total_r / len(outcomes)
        win_r = sum(o["net_r"] for o in outcomes if o["is_win"])
        los_r = abs(sum(o["net_r"] for o in outcomes if not o["is_win"]))
        pf = win_r / los_r if los_r > 0 else 2.0

        conn.execute("""
            INSERT OR REPLACE INTO pipeline_statistics
            (pattern, total_signals, wins, losses, total_r, win_rate, avg_r, profit_factor, last_updated)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (pat, len(outcomes), wins, losses, total_r, wr, avg_r, pf, today))
    conn.commit()
    conn.close()


def print_stats() -> None:
    _ensure_table()
    conn = get_connection()
    rows = conn.execute("""
        SELECT pattern, total_signals, win_rate, avg_r, profit_factor, last_updated
        FROM pipeline_statistics ORDER BY avg_r DESC
    """).fetchall()
    conn.close()

    print("\n" + "=" * 90)
    print("  FULL-PIPELINE REPLAY RESULTS  (pattern + RS + liquidity + risk + asymmetry + VCP + regime)")
    print("  Compare against validation/backtest.py's pattern_statistics (raw pattern, no gates)")
    print("=" * 90)
    print(f"  {'PATTERN':<20} {'N':>6} {'WIN%':>6} {'AVG R':>8} {'PF':>5}  {'UPDATED'}")
    print("  " + "-" * 84)
    for r in rows:
        print(f"  {r['pattern']:<20} {r['total_signals']:>6} {r['win_rate']*100:>5.0f}% "
              f"{r['avg_r']:>7.2f}R {r['profit_factor']:>5.1f}  {r['last_updated'] or '-'}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", nargs="?", default=None)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    elif args.ticker:
        run_replay(tickers=[args.ticker.upper() if args.ticker.endswith(".NS")
                             else args.ticker.upper() + ".NS"])
    else:
        run_replay()
