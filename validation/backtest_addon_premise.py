"""
validation/backtest_addon_premise.py — P2-08 Phase 1 (Statistical Feasibility)

Tests the raw PREMISE behind P2-02's ADD-ON signal: "a held stock giving a
SECOND qualifying breakout later is a statistically favorable event" --
before building (Phase 3) or trusting with real capital (P2-08's real gate,
ADDON_LIVE_EXECUTION in orchestrator.py) any tranche-sizing/blended-stop
mechanism on top of it.

Does NOT model tranches, blended cost, or stop recompute -- that's Phase 3,
and only worth building if this test supports the premise. This only asks:
does a second breakout's own forward return beat sensible baselines?

Four comparison groups, same point-in-time universe + real gate chain as
pipeline_replay_deep.py (imported directly, not reimplemented):

  1. FIRST_BREAKOUT   — a ticker's first date ever clearing the full gate
                        chain (Pattern+Liquidity+RS+Risk+Asymmetry+VCP).
  2. SECOND_BREAKOUT  — any LATER date for the same ticker that separately
                        clears the full gate chain AND meets P2-02's own
                        criteria (RVOL>=1.5x, RS>=70th) -- this is the
                        literal real-world ADD-ON trigger being tested.
  3. RANDOM_BASELINE  — dates sampled uniformly at random from the
                        point-in-time universe, no gate criteria at all
                        (the null: "did nothing but time in the market").
  4. RS_ONLY_BASELINE — dates where RS>=70th percentile alone, WITHOUT
                        requiring the rest of the gate chain (isolates
                        whether RS alone explains any edge, or whether the
                        full gate combination adds something beyond "strong
                        stock").

For each group, at 5/10/20-day forward horizons: mean return, median
return, win rate (>=5% net, same WIN_THRESHOLD as pipeline_replay_deep.py),
max drawdown within the window, both gross and cost-adjusted (using the
same ROUND_TRIP_COST_PCT already established). This is a RETURN-DISTRIBUTION
question, not a pattern-edge question -- deliberately different framing from
pipeline_replay_deep.py's win/R-multiple test, which asks "is this pattern
profitable" rather than "how does this signal's return distribution compare
to a baseline."

DECISION RULE (per the staged research plan): if SECOND_BREAKOUT's returns
do not clearly beat RANDOM_BASELINE and RS_ONLY_BASELINE, the ADD-ON premise
does not hold up historically -- do not proceed to Phase 2/3, and leave
ADDON_LIVE_EXECUTION=False permanently for this feature. If it does beat
both baselines, proceed to Phase 2 (sensitivity analysis on ladder ratios
and signal timing).

Usage:
    python validation/backtest_addon_premise.py
    python validation/backtest_addon_premise.py --tickers RELIANCE,TCS,INFY   # quick subset test
    python validation/backtest_addon_premise.py --sample-cap 2000            # cap baseline sample size
"""

import sys
import json
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.schema import get_connection
from agents.pattern_agent import PatternAgent
from agents.rs_agent import RSAgent, compute_universe_ranks
from agents.liquidity_agent import LiquidityAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
from agents.vcp_gate import VCPContractionGate
from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from validation.backtest import ROUND_TRIP_COST_PCT, _print_cost_model
from load_deep_history import get_point_in_time_universe

# Reuse pipeline_replay_deep.py's existing, already-tested I/O helpers
# rather than reimplementing them -- same DB connections, same price
# loading, same regime map, same ticker metadata.
from validation.pipeline_replay_deep import (
    _load_price_history_deep, _load_nifty_history, _load_regime_map,
    _regime_for_date, _build_ticker_meta, _all_deep_tickers,
)

log = logging.getLogger(__name__)

# P4-09: file logging — P2-08's feasibility test runs for a long time;
# dated log in logs/ so the run that produced a particular decision is
# permanently auditable, not just the JSON output it already writes.
_LOG_DIR = BASE_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = _LOG_DIR / f"addon_premise_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ]
)
log.info(f"Log file: {_log_file}")

STEP = 5
HORIZONS = [5, 10, 20]
WIN_THRESHOLD = 0.05   # same threshold pipeline_replay_deep.py uses, for consistency
RANDOM_SEED = 42        # reproducible sampling

_RESULTS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS addon_premise_test (
    test_group      TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    signal_date     TEXT NOT NULL,
    horizon_days    INTEGER NOT NULL,
    gross_return    REAL,
    net_return      REAL,
    is_win          INTEGER,
    max_drawdown    REAL,
    computed_at     TEXT,
    PRIMARY KEY (test_group, ticker, signal_date, horizon_days)
)
"""


def _ensure_table():
    conn = get_connection()
    conn.execute(_RESULTS_TABLE_DDL)
    conn.commit()
    conn.close()


def _rvol(df: pd.DataFrame, i: int) -> float:
    vol = df["Volume"].to_numpy(dtype=float)
    if i < 20:
        return 0.0
    avg20 = np.mean(vol[i-20:i])
    return float(vol[i] / avg20) if avg20 > 0 else 0.0


def _forward_returns(df: pd.DataFrame, signal_idx: int, entry: float) -> dict:
    """
    Close-to-close returns at each horizon (deliberately different from
    pipeline_replay_deep.py's _forward_test_net, which uses max-high/min-low
    over a single 20-bar window for a win/R-multiple pattern-edge test).
    Here we want an actual RETURN DISTRIBUTION at each of 5/10/20 days, plus
    max drawdown within the window -- the question is "how does this
    signal's return distribution compare to a baseline," not "is this
    pattern profitable."
    Returns {horizon: {"gross": x, "net": x, "is_win": bool, "max_dd": x}}
    or {} if there isn't enough forward data for a given horizon (skipped,
    not zero-filled).
    """
    out = {}
    close = df["Close"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    for h in HORIZONS:
        end_idx = signal_idx + h
        if end_idx >= len(close):
            continue
        exit_price = close[end_idx]
        gross = (exit_price - entry) / entry if entry > 0 else 0.0
        net = gross - ROUND_TRIP_COST_PCT
        window_low = float(np.min(low[signal_idx+1:end_idx+1])) if end_idx > signal_idx else exit_price
        max_dd = (window_low - entry) / entry if entry > 0 else 0.0
        out[h] = {
            "gross": round(gross, 4), "net": round(net, 4),
            "is_win": net >= WIN_THRESHOLD, "max_dd": round(max_dd, 4),
        }
    return out


def _passes_full_gate_chain(window: pd.DataFrame, tier: str, ticker: str,
                             day_ranks: dict, nifty_window: pd.DataFrame) -> tuple:
    """
    Runs the SAME gate chain pipeline_replay_deep.py uses (Pattern, Liquidity,
    RS, Risk, Asymmetry, VCP) -- reused conceptually rather than imported as
    one function since pipeline_replay_deep.py inlines this in replay_ticker()
    rather than exposing it separately. Returns (passed: bool, rvol: float,
    rs_percentile: float, entry: float) so callers can check P2-02's own
    RVOL/RS criteria on top of a pass here without recomputing anything.
    """
    try:
        pa = PatternAgent(window)
    except Exception:
        return False, 0.0, 0.0, 0.0
    if not pa.pattern:
        return False, 0.0, 0.0, 0.0

    liq = LiquidityAgent(window, universe=tier)
    if not liq.passes():
        return False, 0.0, 0.0, 0.0

    if len(nifty_window) < 20:
        return False, 0.0, 0.0, 0.0
    rsa = RSAgent(window, nifty_window, universe_ranks=day_ranks, ticker=ticker)
    rs_percentile = rsa.get_percentile()
    if not rsa.passes_gate():
        return False, 0.0, rs_percentile, 0.0

    last_close = float(window["Close"].iloc[-1])
    entry = float(pa.breakout_level) if pa.breakout_level > 0 else last_close
    entry_low = last_close * 0.995
    entry_high = entry * 1.005

    risk = RiskAgent(window, pa.breakout_level, entry_low, entry_high, universe=tier)
    if not risk.passes():
        return False, 0.0, rs_percentile, 0.0

    ag = AsymmetryGate(entry=risk.entry_high if hasattr(risk, "entry_high") else entry,
                        stop=risk.stop, target1=risk.target1, universe=tier)
    try:
        ag_result = ag.check_dynamic(df=window, w4_pct=0.0)
        if not ag_result.get("qualified") and ag_result.get("fail_stage") == "INPUT":
            ag_result = ag.check()
    except Exception:
        ag_result = ag.check() if hasattr(ag, "check") else {"qualified": False}
    if not ag_result.get("qualified", False):
        return False, 0.0, rs_percentile, 0.0

    vcpg = VCPContractionGate(df=window)
    vcp = vcpg.check()
    if vcp.get("hard_reject", False):
        return False, 0.0, rs_percentile, 0.0

    rvol = _rvol(window, len(window) - 1)
    return True, rvol, rs_percentile, entry


def find_gate_qualifying_signals(ticker: str, tier: str, nifty_df: pd.DataFrame,
                                  universe_ranks_by_date: dict, earliest_snapshot_date: str) -> list:
    """
    Chronological list of every date this ticker clears the FULL gate chain,
    with rvol/rs_percentile/entry recorded -- the raw material both
    FIRST_BREAKOUT and SECOND_BREAKOUT groups are built from.
    """
    df = _load_price_history_deep(ticker)
    if len(df) < 80:
        return []

    signals = []
    for i in range(60, len(df) - max(HORIZONS), STEP):
        date_str = df.index[i].strftime("%Y-%m-%d")
        if date_str < earliest_snapshot_date:
            continue
        pit_universe = get_point_in_time_universe(date_str)
        if ticker not in pit_universe:
            continue

        window = df.iloc[:i+1].copy()
        window.index = range(len(window))

        day_ranks = universe_ranks_by_date.get(date_str, {})
        nifty_window = nifty_df.loc[:df.index[i]].copy()
        nifty_window.columns = ["Close"]

        passed, rvol, rs_pct, entry = _passes_full_gate_chain(window, tier, ticker, day_ranks, nifty_window)
        if passed:
            signals.append({"idx": i, "date": date_str, "rvol": rvol, "rs_percentile": rs_pct, "entry": entry})
    return signals


def find_rs_only_signals(ticker: str, nifty_df: pd.DataFrame, universe_ranks_by_date: dict,
                          earliest_snapshot_date: str, sample_rate: int = 1) -> list:
    """RS>=70th alone, no other gate -- isolates whether RS alone explains any edge."""
    df = _load_price_history_deep(ticker)
    if len(df) < 80:
        return []
    signals = []
    for i in range(60, len(df) - max(HORIZONS), STEP * sample_rate):
        date_str = df.index[i].strftime("%Y-%m-%d")
        if date_str < earliest_snapshot_date:
            continue
        pit_universe = get_point_in_time_universe(date_str)
        if ticker not in pit_universe:
            continue
        window = df.iloc[:i+1].copy()
        window.index = range(len(window))
        day_ranks = universe_ranks_by_date.get(date_str, {})
        nifty_window = nifty_df.loc[:df.index[i]].copy()
        if len(nifty_window) < 20:
            continue
        nifty_window.columns = ["Close"]
        try:
            rsa = RSAgent(window, nifty_window, universe_ranks=day_ranks, ticker=ticker)
            rs_pct = rsa.get_percentile()
        except Exception:
            continue
        if rs_pct >= 70:
            entry = float(window["Close"].iloc[-1])
            signals.append({"idx": i, "date": date_str, "entry": entry})
    return signals


def run_test(tickers: list = None, sample_cap: int = 3000):
    _ensure_table()
    _print_cost_model()
    random.seed(RANDOM_SEED)

    meta = _build_ticker_meta()
    all_tickers = _all_deep_tickers()
    if tickers:
        all_tickers = [t for t in all_tickers if t in tickers]

    log.info("Loading NIFTY history from market_regime_history (deep-backfilled)...")
    nifty_df = _load_nifty_history()

    conn = get_connection()
    earliest_snapshot_date = conn.execute("SELECT MIN(effective_date) FROM universe_snapshots").fetchone()[0]
    conn.close()
    if not earliest_snapshot_date:
        raise RuntimeError("universe_snapshots is empty — run load_deep_history.py --universe-csv first.")
    log.info(f"Point-in-time universe coverage starts {earliest_snapshot_date}.")

    log.info(f"Loading price history for {len(all_tickers)} tickers...")
    stock_data = {}
    for ticker in all_tickers:
        df = _load_price_history_deep(ticker)
        if not df.empty:
            stock_data[ticker] = df

    all_dates = sorted(nifty_df.index)
    log.info("Precomputing per-date universe RS ranks (this is the slow part)...")
    universe_ranks_by_date = {}
    sampled_dates = [d for d in all_dates[::STEP] if d.strftime("%Y-%m-%d") >= earliest_snapshot_date]
    for idx, date in enumerate(sampled_dates):
        date_str = date.strftime("%Y-%m-%d")
        pit_universe = set(get_point_in_time_universe(date_str))
        relevant = {t: df.loc[:date] for t, df in stock_data.items() if t in pit_universe}
        universe_ranks_by_date[date_str] = compute_universe_ranks(
            {"nifty50_data": nifty_df.loc[:date], "stock_data": relevant}
        )
        if idx % 20 == 0:
            log.info(f"  {idx}/{len(sampled_dates)} dates ranked ({len(relevant)} point-in-time members)")

    first_breakout_signals = []    # [(ticker, tier, signal_dict)]
    second_breakout_signals = []
    rs_only_signals = []
    random_pool = []               # (ticker, idx, entry) candidates for random baseline

    log.info(f"Scanning {len(all_tickers)} tickers for gate-qualifying signals...")
    for i, ticker in enumerate(all_tickers, 1):
        name, sector, tier = meta.get(ticker, (ticker.replace(".NS", ""), "Unknown", "MID"))
        signals = find_gate_qualifying_signals(ticker, tier, nifty_df, universe_ranks_by_date, earliest_snapshot_date)
        if signals:
            first_breakout_signals.append((ticker, tier, signals[0]))
            for s in signals[1:]:
                if s["rvol"] >= 1.5 and s["rs_percentile"] >= 70:
                    second_breakout_signals.append((ticker, tier, s))

        rs_signals = find_rs_only_signals(ticker, nifty_df, universe_ranks_by_date, earliest_snapshot_date)
        rs_only_signals.extend((ticker, s) for s in rs_signals)

        df = stock_data.get(ticker)
        if df is not None:
            for idx in range(60, len(df) - max(HORIZONS), STEP * 4):
                random_pool.append((ticker, idx, float(df["Close"].iloc[idx])))

        if i % 50 == 0:
            log.info(f"  {i}/{len(all_tickers)} tickers scanned | "
                     f"{len(first_breakout_signals)} first-breakout, "
                     f"{len(second_breakout_signals)} second-breakout so far")

    n_second = len(second_breakout_signals)
    log.info(f"Found {len(first_breakout_signals)} first-breakout, {n_second} second-breakout, "
             f"{len(rs_only_signals)} RS-only, {len(random_pool)} random-pool candidates.")

    if n_second == 0:
        log.warning("Zero second-breakout signals found historically — the premise cannot be tested "
                    "on this data. Stopping here (see P2-08 decision rule).")
        return

    # BUG FIX (caught by the 10-ticker test run): baselines must NOT be
    # shrunk to match n_second. A rare event (second-breakout) having few
    # occurrences is real information -- but RANDOM_BASELINE and
    # RS_ONLY_BASELINE exist to provide a STABLE reference distribution,
    # which requires their own large sample regardless of how rare the
    # thing being compared against is. Originally sampled both down to
    # min(n_second, ...), which collapsed a 2460-candidate random pool down
    # to N=1 in testing -- statistically meaningless. Each group now uses
    # up to sample_cap independently.
    random_sample = random.sample(random_pool, min(sample_cap, len(random_pool)))
    rs_only_sample = random.sample(rs_only_signals, min(sample_cap, len(rs_only_signals)))
    second_sample = second_breakout_signals[:sample_cap] if len(second_breakout_signals) > sample_cap else second_breakout_signals
    first_sample = first_breakout_signals[:sample_cap] if len(first_breakout_signals) > sample_cap else first_breakout_signals

    log.info(f"Computing forward returns: {len(first_sample)} first-breakout, {len(second_sample)} "
             f"second-breakout, {len(rs_only_sample)} RS-only, {len(random_sample)} random...")

    results = defaultdict(list)

    for ticker, tier, s in first_sample:
        df = stock_data.get(ticker)
        if df is None:
            continue
        fr = _forward_returns(df, s["idx"], s["entry"])
        for h, r in fr.items():
            results[("FIRST_BREAKOUT", h)].append((ticker, s["date"], r))

    for ticker, tier, s in second_sample:
        df = stock_data.get(ticker)
        if df is None:
            continue
        fr = _forward_returns(df, s["idx"], s["entry"])
        for h, r in fr.items():
            results[("SECOND_BREAKOUT", h)].append((ticker, s["date"], r))

    for ticker, s in rs_only_sample:
        df = stock_data.get(ticker)
        if df is None:
            continue
        fr = _forward_returns(df, s["idx"], s["entry"])
        for h, r in fr.items():
            results[("RS_ONLY_BASELINE", h)].append((ticker, s["date"], r))

    for ticker, idx, entry in random_sample:
        df = stock_data.get(ticker)
        if df is None:
            continue
        fr = _forward_returns(df, idx, entry)
        for h, r in fr.items():
            date_str = df.index[idx].strftime("%Y-%m-%d")
            results[("RANDOM_BASELINE", h)].append((ticker, date_str, r))

    _store_and_print(results)


def _store_and_print(results: dict):
    conn = get_connection()
    now = datetime.now().isoformat()
    summary_rows = []

    for (group, horizon), items in sorted(results.items()):
        if not items:
            continue
        net_returns = [r["net"] for _, _, r in items]
        gross_returns = [r["gross"] for _, _, r in items]
        wins = sum(1 for _, _, r in items if r["is_win"])
        max_dds = [r["max_dd"] for _, _, r in items]

        summary_rows.append({
            "group": group, "horizon": horizon, "n": len(items),
            "mean_net": np.mean(net_returns), "median_net": np.median(net_returns),
            "mean_gross": np.mean(gross_returns),
            "win_rate": wins / len(items), "mean_max_dd": np.mean(max_dds),
        })

        for ticker, date_str, r in items:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO addon_premise_test
                    (test_group, ticker, signal_date, horizon_days, gross_return,
                     net_return, is_win, max_drawdown, computed_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (group, ticker, date_str, horizon, r["gross"], r["net"],
                      int(r["is_win"]), r["max_dd"], now))
            except Exception as e:
                log.debug(f"  Store failed for {group}/{ticker}/{date_str}/{horizon}d: {e}")
    conn.commit()
    conn.close()

    print("\n" + "=" * 100)
    print("  P2-08 PHASE 1 — ADD-ON PREMISE STATISTICAL FEASIBILITY TEST")
    print("=" * 100)
    print(f"  {'GROUP':<20}{'HORIZON':>8}{'N':>8}{'MEAN NET':>11}{'MEDIAN NET':>12}"
          f"{'WIN%':>8}{'MEAN MAXDD':>12}")
    print("  " + "-" * 96)
    for row in sorted(summary_rows, key=lambda x: (x["horizon"], x["group"])):
        print(f"  {row['group']:<20}{row['horizon']:>7}d{row['n']:>8}"
              f"{row['mean_net']*100:>10.2f}%{row['median_net']*100:>11.2f}%"
              f"{row['win_rate']*100:>7.1f}%{row['mean_max_dd']*100:>11.2f}%")
    print("=" * 100)
    print("\n  DECISION RULE: does SECOND_BREAKOUT clearly beat RANDOM_BASELINE and")
    print("  RS_ONLY_BASELINE at each horizon? If not, the ADD-ON premise does not hold —")
    print("  do not proceed to Phase 2/3, leave ADDON_LIVE_EXECUTION=False permanently.\n")

    out_path = BASE_DIR / "logs" / f"addon_premise_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)
    print(f"  Full summary written to {out_path}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default=None, help="Comma-separated tickers for a quick subset test, e.g. RELIANCE,TCS,INFY")
    ap.add_argument("--sample-cap", type=int, default=3000, help="Cap sample size per group (default 3000)")
    args = ap.parse_args()

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() + (".NS" if not t.strip().upper().endswith(".NS") else "")
                   for t in args.tickers.split(",")]

    run_test(tickers=tickers, sample_cap=args.sample_cap)
