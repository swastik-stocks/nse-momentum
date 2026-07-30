"""
NSE Momentum — Hourly Full-Pipeline Replay
Same real gate chain as validation/pipeline_replay_deep.py (PatternAgent,
RSAgent, LiquidityAgent, RiskAgent, AsymmetryGate, VCPContractionGate),
adapted to run against price_history_hourly instead of price_history_deep.

THREE HONEST DIFFERENCES FROM THE DAILY DEEP REPLAY (read before trusting
results against the same bar as Cup & Handle / Swing High Breakout):

  1. NO POINT-IN-TIME UNIVERSE. universe_snapshots isn't available (the
     daily-deep tables were found missing from momentum_v4.db mid-session
     and a rebuild was deliberately deferred — not blocking this work).
     This replay uses the CURRENT static NSE_UNIVERSE for every test date,
     reintroducing a small amount of survivorship bias the daily deep
     replay was specifically built to eliminate. Worth knowing, not hidden.

  2. bars_per_day=7 threaded through every agent (PatternAgent, RSAgent,
     RiskAgent, LiquidityAgent, AsymmetryGate, VCPContractionGate's
     windows= param) — see hourly_scaling.py and each agent's own
     docstring for what was scaled and why. Verified against the original
     daily behavior at bars_per_day=1 before this was built.

  3. FORWARD_BARS and STEP are scaled by BARS_PER_DAY so the forward-test
     window and sampling cadence represent the same REAL CALENDAR TIME as
     the daily replay (a 20-daily-bar forward test becomes a 140-hourly-
     bar forward test — the same ~20 trading days either way), not 7x
     more or less of it.

SCOPE: only the 3 live-weighted patterns (Cup & Handle, Swing High
Breakout, VCP) can ever be assigned to self.pattern regardless of
--all-patterns, since PatternAgent's own hourly scaling only covers those
3 (see pattern_agent.py's scope note) -- the other 16 detected patterns
run on unscaled daily windows and would produce methodologically invalid
hourly results if tested, so --all-patterns here only tests all patterns
PatternAgent CAN validly detect at this resolution (which is exactly the
subset most worth testing anyway).

Usage:
    python validation/pipeline_replay_hourly.py
    python validation/pipeline_replay_hourly.py RELIANCE.NS
    python validation/pipeline_replay_hourly.py --stats
    python validation/pipeline_replay_hourly.py --fresh
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

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
from hourly_scaling import BARS_PER_DAY

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# [SCALED] 20-daily-bar forward window -> ~20 trading days either way.
# 5-daily-bar sampling step -> same real-time sampling cadence, not 7x finer.
FORWARD_BARS = round(20 * BARS_PER_DAY)
STEP         = round(5  * BARS_PER_DAY)
MIN_BARS     = round(80 * BARS_PER_DAY)
WIN_THRESHOLD = 0.05   # magnitude threshold, NOT a bar-count -- unscaled by design

PROGRESS_SCHEMA_VERSION = 1
REGIME_PENALTIES = {"A": 0, "B": 0, "C": -5, "D": -12, "E": -20}
DEFAULT_TIER = "MID"

_HOURLY_STATS_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_statistics_hourly (
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

_PROGRESS_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_replay_hourly_progress (
    ticker        TEXT PRIMARY KEY,
    signals       INTEGER,
    results_json  TEXT,
    completed_at  TEXT
)
"""

_META_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_replay_hourly_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""


def _ensure_tables():
    conn = get_connection()
    conn.execute(_HOURLY_STATS_DDL)
    conn.execute(_PROGRESS_DDL)
    conn.execute(_META_DDL)
    conn.commit()
    conn.close()


def _clear_progress():
    conn = get_connection()
    conn.execute(_PROGRESS_DDL)
    conn.execute(_META_DDL)
    conn.execute("DELETE FROM pipeline_replay_hourly_progress")
    conn.execute("DELETE FROM pipeline_replay_hourly_meta")
    conn.commit()
    conn.close()


def _check_or_set_mode(test_all_patterns: bool) -> None:
    conn = get_connection()
    conn.execute(_META_DDL)
    row = conn.execute("SELECT value FROM pipeline_replay_hourly_meta WHERE key='test_all_patterns'").fetchone()
    current = "1" if test_all_patterns else "0"
    if row is None:
        conn.execute("INSERT INTO pipeline_replay_hourly_meta (key, value) VALUES ('test_all_patterns', ?)", (current,))
        conn.commit()
        conn.close()
        return
    conn.close()
    if row["value"] != current:
        raise RuntimeError(
            f"Checkpoint mode mismatch (existing='{row['value']}', this run='{current}'). "
            f"Re-run with --fresh to start clean in the new mode."
        )


def _load_completed_tickers() -> set:
    conn = get_connection()
    rows = conn.execute("SELECT ticker FROM pipeline_replay_hourly_progress").fetchall()
    conn.close()
    return {r["ticker"] for r in rows}


def _save_ticker_progress(ticker: str, res: dict) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO pipeline_replay_hourly_progress
        (ticker, signals, results_json, completed_at)
        VALUES (?,?,?,?)
    """, (ticker, res["signals"], json.dumps(res["results"]), datetime.today().isoformat()))
    conn.commit()
    conn.close()


def _load_all_progress() -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT ticker, signals, results_json FROM pipeline_replay_hourly_progress"
    ).fetchall()
    conn.close()
    return [
        {"ticker": r["ticker"], "signals": r["signals"], "results": json.loads(r["results_json"])}
        for r in rows
    ]


def _build_ticker_meta() -> dict:
    return {t[0]: (t[1], t[2], t[3]) for t in NSE_UNIVERSE}


def _all_hourly_tickers() -> list:
    conn = get_connection()
    rows = conn.execute("SELECT DISTINCT ticker FROM price_history_hourly").fetchall()
    conn.close()
    return [r["ticker"] for r in rows]


def _load_price_history_hourly(ticker: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        "SELECT datetime, open, high, low, close, volume FROM price_history_hourly "
        "WHERE ticker=? ORDER BY datetime ASC",
        conn, params=(ticker,)
    )
    conn.close()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df


def _load_nifty_history_daily() -> pd.DataFrame:
    """Regime lookup stays DAILY-resolution -- a regime letter per calendar
    day is meaningful and sufficient; it doesn't need hourly granularity."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT date, nifty_close FROM market_regime_history ORDER BY date ASC",
        conn
    )
    conn.close()
    if df.empty:
        raise RuntimeError("market_regime_history is empty — run regime_backfill.py first.")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.rename(columns={"nifty_close": "Close"}, inplace=True)
    return df


def _load_regime_map() -> dict:
    conn = get_connection()
    rows = conn.execute("SELECT date, regime FROM market_regime_history").fetchall()
    conn.close()
    return {r["date"]: r["regime"] for r in rows}


def _regime_for_date(regime_map: dict, date_str: str) -> str:
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
                   regime_map: dict, test_all_patterns: bool = False) -> dict:
    df = _load_price_history_hourly(ticker)
    if len(df) < MIN_BARS:
        return {"ticker": ticker, "signals": 0, "results": {}}

    cfg = UNIVERSE_CONFIG[tier]
    pattern_results: dict = {}
    total_signals = 0

    for i in range(MIN_BARS - 20, len(df) - FORWARD_BARS, STEP):
        date_str = df.index[i].strftime("%Y-%m-%d")

        window = df.iloc[:i].copy()
        window.index = range(len(window))

        try:
            pa = PatternAgent(window, bars_per_day=BARS_PER_DAY)
        except Exception:
            continue

        if test_all_patterns:
            candidates = list(pa.all_detections)
        else:
            candidates = [(pa.pattern, pa.breakout_level)] if pa.pattern else []

        if not candidates:
            continue

        liq = LiquidityAgent(window, universe=tier, bars_per_day=BARS_PER_DAY)
        if not liq.passes():
            continue

        day_ranks = universe_ranks_by_date.get(date_str, {})
        nifty_window = nifty_df.loc[:pd.Timestamp(date_str)].copy()
        if len(nifty_window) < 20:
            continue
        nifty_window.columns = ["Close"]
        rsa = RSAgent(window, nifty_window, universe_ranks=day_ranks, ticker=ticker,
                      bars_per_day=BARS_PER_DAY)
        if not rsa.passes_gate():
            continue

        last_close = float(window["Close"].iloc[-1])

        for pat_name, breakout_level in candidates:
            if not pat_name:
                continue

            entry      = float(breakout_level) if breakout_level > 0 else last_close
            entry_low  = last_close * 0.995
            entry_high = entry * 1.005

            risk = RiskAgent(window, breakout_level, entry_low, entry_high,
                              universe=tier, bars_per_day=BARS_PER_DAY)
            if not risk.passes():
                continue

            ag = AsymmetryGate(entry=risk.entry_high if hasattr(risk, "entry_high") else entry,
                                stop=risk.stop, target1=risk.target1, universe=tier,
                                bars_per_day=BARS_PER_DAY)
            try:
                ag_result = ag.check_dynamic(df=window, w4_pct=0.0)
                if not ag_result.get("qualified") and ag_result.get("fail_stage") == "INPUT":
                    ag_result = ag.check()
            except Exception:
                ag_result = ag.check() if hasattr(ag, "check") else {"qualified": False}
            if not ag_result.get("qualified", False):
                continue

            vcp_windows = [round(20 * BARS_PER_DAY), round(10 * BARS_PER_DAY), round(5 * BARS_PER_DAY)]
            vcpg = VCPContractionGate(df=window, windows=vcp_windows)
            vcp = vcpg.check()
            if vcp.get("hard_reject", False):
                continue

            regime = _regime_for_date(regime_map, date_str)
            penalty = int(REGIME_PENALTIES.get(regime, -5) * cfg["regime_penalty_mult"])

            result = _forward_test_net(df, i, entry)
            result["regime"] = regime
            result["penalty_applied"] = penalty
            result["date"] = date_str

            pattern_results.setdefault(pat_name, []).append(result)
            total_signals += 1

    return {"ticker": ticker, "signals": total_signals, "results": pattern_results}


def _precompute_universe_ranks_by_date(dates: list, stock_data: dict, nifty_df: pd.DataFrame) -> dict:
    """
    Sampled by unique CALENDAR DATE (not per-hour) -- RS percentile is
    inherently a slower, multi-week signal; recomputing it per intraday
    bar would be expensive and meaningless. All intraday bars sharing a
    date reuse that date's ranks.
    """
    log.info("  Precomputing per-date universe RS ranks (this is the slow part)...")
    result = {}
    unique_dates = sorted(set(d.strftime("%Y-%m-%d") for d in dates))
    sampled_dates = unique_dates[::max(1, STEP // 7)]  # ~1 recompute per sampled calendar day
    for idx, date_str in enumerate(sampled_dates):
        ts = pd.Timestamp(date_str)
        data_dict = {
            "nifty50_data": nifty_df.loc[:ts],
            "stock_data": {t: df.loc[:ts] for t, df in stock_data.items()},
        }
        result[date_str] = compute_universe_ranks(data_dict, bars_per_day=BARS_PER_DAY)
        if idx % 20 == 0:
            log.info(f"    {idx}/{len(sampled_dates)} dates ranked")
    return result


def run_replay(tickers: list = None, fresh: bool = False, test_all_patterns: bool = False) -> None:
    _ensure_tables()
    _print_cost_model()
    log.info(f"BARS_PER_DAY = {BARS_PER_DAY} | FORWARD_BARS = {FORWARD_BARS} | "
             f"STEP = {STEP} | MIN_BARS = {MIN_BARS}")
    log.info("NOTE: no point-in-time universe filtering this run (universe_snapshots "
             "unavailable) -- using current static NSE_UNIVERSE. See module docstring.")

    if fresh:
        log.info("--fresh given: clearing existing checkpoint, starting over.")
        _clear_progress()

    _check_or_set_mode(test_all_patterns)

    completed = _load_completed_tickers()
    if completed:
        log.info(f"Resuming: {len(completed)} tickers already completed, will be skipped.")

    meta = _build_ticker_meta()
    all_tickers = _all_hourly_tickers()
    if tickers:
        all_tickers = [t for t in all_tickers if t in tickers]

    log.info("Loading NIFTY daily history from market_regime_history...")
    nifty_df = _load_nifty_history_daily()
    regime_map = _load_regime_map()

    log.info(f"Loading hourly price history for {len(all_tickers)} tickers "
             "(needed for RS rank precomputation)...")
    stock_data = {}
    for ticker in all_tickers:
        df = _load_price_history_hourly(ticker)
        if not df.empty:
            stock_data[ticker] = df

    all_dates = sorted(set(idx for df in stock_data.values() for idx in df.index))
    universe_ranks_by_date = _precompute_universe_ranks_by_date(all_dates, stock_data, nifty_df)

    remaining = [t for t in all_tickers if t not in completed]
    log.info(f"Running hourly full-pipeline replay on {len(remaining)} remaining "
             f"of {len(all_tickers)} total tickers...")

    total_signals = sum(r["signals"] for r in _load_all_progress())

    for i, ticker in enumerate(remaining, 1):
        name, sector, tier = meta.get(ticker, (ticker.replace(".NS", ""), "Unknown", DEFAULT_TIER))
        try:
            res = replay_ticker(ticker, name, sector, tier, nifty_df,
                                 universe_ranks_by_date, regime_map,
                                 test_all_patterns=test_all_patterns)
        except Exception as e:
            log.warning(f"  {ticker} CRASHED during replay ({type(e).__name__}: {e}) — skipping, continuing run")
            continue

        _save_ticker_progress(ticker, res)
        total_signals += res["signals"]
        if i % 25 == 0:
            log.info(f"  {i}/{len(remaining)} remaining | {total_signals:,} gate-cleared signals so far")

    all_results = _load_all_progress()
    aggregate_and_store(all_results)
    log.info(f"Hourly replay complete: {total_signals:,} gate-cleared signals across {len(all_tickers)} tickers")
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
            INSERT OR REPLACE INTO pipeline_statistics_hourly
            (pattern, total_signals, wins, losses, total_r, win_rate, avg_r, profit_factor, last_updated)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (pat, len(outcomes), wins, losses, total_r, wr, avg_r, pf, today))
    conn.commit()
    conn.close()


def print_stats() -> None:
    _ensure_tables()
    conn = get_connection()
    rows = conn.execute("""
        SELECT pattern, total_signals, win_rate, avg_r, profit_factor, last_updated
        FROM pipeline_statistics_hourly ORDER BY avg_r DESC
    """).fetchall()
    conn.close()

    print("\n" + "=" * 95)
    print(f"  HOURLY FULL-PIPELINE REPLAY  (bars_per_day={BARS_PER_DAY}, no point-in-time universe)")
    print("  Compare against pipeline_statistics_deep (daily, point-in-time) for the same patterns")
    print("=" * 95)
    print(f"  {'PATTERN':<20} {'N':>6} {'WIN%':>6} {'AVG R':>8} {'PF':>5}  {'UPDATED'}")
    print("  " + "-" * 89)
    for r in rows:
        print(f"  {r['pattern']:<20} {r['total_signals']:>6} {r['win_rate']*100:>5.0f}% "
              f"{r['avg_r']:>7.2f}R {r['profit_factor']:>5.1f}  {r['last_updated'] or '-'}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", nargs="?", default=None)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--all-patterns", action="store_true")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    elif args.ticker:
        t = args.ticker.upper() if args.ticker.endswith(".NS") else args.ticker.upper() + ".NS"
        run_replay(tickers=[t], fresh=args.fresh, test_all_patterns=args.all_patterns)
    else:
        run_replay(fresh=args.fresh, test_all_patterns=args.all_patterns)
