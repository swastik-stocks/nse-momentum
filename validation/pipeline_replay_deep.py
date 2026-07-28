"""
NSE Momentum v6 — Deep-History Full-Pipeline Replay
Same real gate chain as validation/pipeline_replay.py (PatternAgent,
RSAgent, LiquidityAgent, RiskAgent, AsymmetryGate, VCPContractionGate),
but reading from:
    price_history_deep    (1999-2026, 501 tickers, Kaggle-sourced)
    universe_snapshots    (47 point-in-time NIFTY 500 snapshots, 2016-2026)
    market_regime_history (deep-backfilled — run regime_backfill_deep.py first)

instead of the 2-year price_history + static current NSE_UNIVERSE the
original pipeline_replay.py used. This is the test that actually answers
whether the earlier "everything is net-negative" finding holds up across
multiple real market regimes (2016-2026 spans several distinct NIFTY
cycles) and without survivorship bias (a stock that got removed from the
index for poor performance is correctly excluded from the universe on
dates after its removal, not just silently absent from today's list).

SCOPE NOTE: universe_snapshots only covers 2016-2026 (47 dates) even
though price_history_deep goes back to 1999. Test dates before the
earliest snapshot (2016-01-18) have no point-in-time membership data, so
this replay only runs on dates from 2016-01-18 onward — still a genuine
~10-year, multi-regime test, just not the full 27 years of price data
available. Extending universe_snapshots earlier than 2016 would need
additional historical reconstitution records not currently available.

For tickers not present in the CURRENT nse_universe.py (delisted, renamed,
or removed since), tier defaults to "MID" and sector to "Unknown" — a
documented approximation, since UNIVERSE_CONFIG's score_gate/min_rrr/
min_adt_cr thresholds differ by tier and we have no historical tier
classification for stocks no longer in the live universe.

Usage:
    python validation/pipeline_replay_deep.py
    python validation/pipeline_replay_deep.py RELIANCE
    python validation/pipeline_replay_deep.py --stats
    python validation/pipeline_replay_deep.py --fresh          # ignore checkpoint, start over
    python validation/pipeline_replay_deep.py --all-patterns --fresh
        # Test all 19 detected patterns (weighted + pruned) independently
        # through the real gate chain, instead of only the 5 pre-weighted
        # ones that can currently win pattern_agent.py's selection. Use
        # --fresh when switching modes — a mismatch raises rather than
        # silently mixing incompatible checkpoint data.

Requires (in this order):
    python load_deep_history.py --price-csv ... --universe-csv ...
    python regime_backfill_deep.py

CHANGELOG (v6.1 — perf + resumability fix):
    1) get_point_in_time_universe() (in load_deep_history.py) now answers
       from an in-memory cache instead of opening a new SQLite connection
       per call. Previously this ran ~250,000 times per full replay
       against only 47 distinct snapshot rows — the dominant cost, and
       the likely cause of a stale-lock hang after an interrupted run.
       See load_deep_history.py's own changelog for details.

    2) Per-ticker checkpointing added (pipeline_replay_deep_progress
       table). Each completed ticker's results are committed to the DB
       immediately, so killing the process (Ctrl+C, crash, laptop sleep)
       no longer loses the whole run — rerunning the same command resumes
       from the last completed ticker automatically. Use --fresh to
       discard the checkpoint and start clean.
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
from load_deep_history import get_point_in_time_universe

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

FORWARD_BARS = 20
WIN_THRESHOLD = 0.05
STEP = 5

# Bump whenever the shape of a stored trade outcome changes (new/removed
# keys in the result dict). Prevents resuming a checkpoint whose saved
# outcomes are missing a field a newer script version depends on (e.g.
# "date", added in schema 2 for split_period_significance.py).
PROGRESS_SCHEMA_VERSION = 2

REGIME_PENALTIES = {"A": 0, "B": 0, "C": -5, "D": -12, "E": -20}
DEFAULT_TIER = "MID"

_DEEP_PIPELINE_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_statistics_deep (
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

# Per-ticker checkpoint — lets a killed/hung/crashed run resume instead of
# starting over. Written immediately after each ticker completes.
_PROGRESS_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_replay_deep_progress (
    ticker        TEXT PRIMARY KEY,
    signals       INTEGER,
    results_json  TEXT,
    completed_at  TEXT
)
"""

# Tiny single-row table recording which mode (all-19-patterns vs the
# original 5-weighted-only) the current checkpoint was built with, so a
# mode switch without --fresh gets caught instead of silently mixing
# incompatible results in the same aggregate.
_META_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_replay_deep_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""


def _ensure_table():
    conn = get_connection()
    conn.execute(_DEEP_PIPELINE_TABLE_DDL)
    conn.commit()
    conn.close()


def _ensure_progress_table():
    conn = get_connection()
    conn.execute(_PROGRESS_DDL)
    conn.execute(_META_DDL)
    conn.commit()
    conn.close()


def _clear_progress():
    conn = get_connection()
    conn.execute(_PROGRESS_DDL)
    conn.execute(_META_DDL)
    conn.execute("DELETE FROM pipeline_replay_deep_progress")
    conn.execute("DELETE FROM pipeline_replay_deep_meta")
    conn.commit()
    conn.close()


def _check_or_set_mode(test_all_patterns: bool) -> None:
    """Guards against silently mixing two incompatible checkpoint runs
    (all-19-patterns vs original 5-weighted-only, or across schema
    versions where the stored outcome dict shape changed) in the same
    aggregate."""
    conn = get_connection()
    conn.execute(_META_DDL)
    row = conn.execute("SELECT value FROM pipeline_replay_deep_meta WHERE key='test_all_patterns'").fetchone()
    schema_row = conn.execute("SELECT value FROM pipeline_replay_deep_meta WHERE key='schema_version'").fetchone()
    current = "1" if test_all_patterns else "0"
    current_schema = str(PROGRESS_SCHEMA_VERSION)

    if row is None:
        conn.execute("INSERT INTO pipeline_replay_deep_meta (key, value) VALUES ('test_all_patterns', ?)",
                      (current,))
        conn.execute("INSERT INTO pipeline_replay_deep_meta (key, value) VALUES ('schema_version', ?)",
                      (current_schema,))
        conn.commit()
        conn.close()
        return
    conn.close()

    if row["value"] != current:
        prev_desc = "all-19-patterns" if row["value"] == "1" else "5-weighted-only (original)"
        this_desc = "all-19-patterns" if test_all_patterns else "5-weighted-only (original)"
        raise RuntimeError(
            f"Checkpoint mode mismatch: existing progress was built in '{prev_desc}' mode, "
            f"but this run is '{this_desc}' mode. Mixing them would corrupt the aggregate "
            f"(same pattern name could mean different things across tickers). "
            f"Re-run with --fresh to start clean in the new mode."
        )
    if schema_row is None or schema_row["value"] != current_schema:
        raise RuntimeError(
            f"Checkpoint schema mismatch: existing progress was saved under an older outcome "
            f"format (missing fields a newer script depends on, e.g. per-trade 'date'). "
            f"Re-run with --fresh to rebuild the checkpoint in the current format."
        )


def _load_completed_tickers() -> set:
    conn = get_connection()
    rows = conn.execute("SELECT ticker FROM pipeline_replay_deep_progress").fetchall()
    conn.close()
    return {r["ticker"] for r in rows}


def _save_ticker_progress(ticker: str, res: dict) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO pipeline_replay_deep_progress
        (ticker, signals, results_json, completed_at)
        VALUES (?,?,?,?)
    """, (ticker, res["signals"], json.dumps(res["results"]), datetime.today().isoformat()))
    conn.commit()
    conn.close()


def _load_all_progress() -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT ticker, signals, results_json FROM pipeline_replay_deep_progress"
    ).fetchall()
    conn.close()
    return [
        {"ticker": r["ticker"], "signals": r["signals"], "results": json.loads(r["results_json"])}
        for r in rows
    ]


def _build_ticker_meta() -> dict:
    """{ticker: (name, sector, tier)} — from current nse_universe.py where
    known, defaulted otherwise (see module docstring)."""
    meta = {t[0]: (t[1], t[2], t[3]) for t in NSE_UNIVERSE}
    return meta


def _all_deep_tickers() -> list:
    conn = get_connection()
    rows = conn.execute("SELECT DISTINCT ticker FROM price_history_deep").fetchall()
    conn.close()
    return [r["ticker"] for r in rows]


def _load_price_history_deep(ticker: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM price_history_deep "
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
        raise RuntimeError("market_regime_history is empty — run regime_backfill_deep.py first.")
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
                   regime_map: dict, earliest_snapshot_date: str,
                   test_all_patterns: bool = False) -> dict:
    """
    test_all_patterns=False (default, original behaviour): only the single
        "winning" pattern PatternAgent selects (self.pattern) is tested per
        bar — i.e. only the 5 currently-weighted patterns can ever appear,
        exactly as in the live scanner's conviction pick.
    test_all_patterns=True (v6.1): every pattern PatternAgent detected that
        bar (self.all_detections — all 19, weighted or pruned) is tested
        independently through the SAME real gate chain (liquidity, RS,
        risk, asymmetry, VCP). Liquidity/RS gates are pattern-independent
        (computed once per date); Risk/Asymmetry/VCP are recomputed per
        pattern since they depend on that pattern's own breakout level.
        This answers "which of all 19 patterns actually has gate-adjusted
        edge" instead of only ever being able to test the 5 pre-weighted
        ones.
    """
    df = _load_price_history_deep(ticker)
    if len(df) < 80:
        return {"ticker": ticker, "signals": 0, "results": {}}

    cfg = UNIVERSE_CONFIG[tier]
    pattern_results: dict = {}
    total_signals = 0

    for i in range(60, len(df) - FORWARD_BARS, STEP):
        date_str = df.index[i].strftime("%Y-%m-%d")

        # Point-in-time universe gate — the whole reason this script exists.
        # Skip entirely if this ticker was not a real NIFTY 500 member on
        # this historical date, or if we're before snapshot coverage begins.
        if date_str < earliest_snapshot_date:
            continue
        pit_universe = get_point_in_time_universe(date_str)
        if ticker not in pit_universe:
            continue

        window = df.iloc[:i].copy()
        window.index = range(len(window))

        try:
            pa = PatternAgent(window)
        except Exception:
            continue

        if test_all_patterns:
            # Every shape detected this bar — 19-pattern-wide candidate list.
            candidates = list(pa.all_detections)
        else:
            # Original behaviour: only the single pre-weighted "winner".
            candidates = [(pa.pattern, pa.breakout_level)] if pa.pattern else []

        if not candidates:
            continue

        # Liquidity + RS gates don't depend on which pattern fired — compute
        # once per date/ticker and reuse across every candidate pattern.
        liq = LiquidityAgent(window, universe=tier)
        if not liq.passes():
            continue

        day_ranks = universe_ranks_by_date.get(date_str, {})
        nifty_window = nifty_df.loc[:df.index[i]].copy()
        if len(nifty_window) < 20:
            continue
        nifty_window.columns = ["Close"]
        rsa = RSAgent(window, nifty_window, universe_ranks=day_ranks, ticker=ticker)
        if not rsa.passes_gate():
            continue

        last_close = float(window["Close"].iloc[-1])

        for pat_name, breakout_level in candidates:
            if not pat_name:
                continue

            entry      = float(breakout_level) if breakout_level > 0 else last_close
            entry_low  = last_close * 0.995
            entry_high = entry * 1.005

            # Risk/Asymmetry/VCP DO depend on the specific pattern's entry —
            # recomputed per candidate, same as the live gate chain would.
            risk = RiskAgent(window, breakout_level, entry_low, entry_high, universe=tier)
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
            result["date"] = date_str  # needed for out-of-sample period splits (validation/split_period_significance.py)

            pattern_results.setdefault(pat_name, []).append(result)
            total_signals += 1

    return {"ticker": ticker, "signals": total_signals, "results": pattern_results}


def _precompute_universe_ranks_by_date(dates: list, stock_data: dict, nifty_df: pd.DataFrame,
                                        earliest_snapshot_date: str) -> dict:
    log.info("  Precomputing per-date universe RS ranks against the POINT-IN-TIME "
             "universe (this is the slow part)...")
    result = {}
    sampled_dates = [d for d in dates[::STEP] if d.strftime("%Y-%m-%d") >= earliest_snapshot_date]
    for idx, date in enumerate(sampled_dates):
        date_str = date.strftime("%Y-%m-%d")
        pit_universe = set(get_point_in_time_universe(date_str))
        relevant_stock_data = {t: df.loc[:date] for t, df in stock_data.items() if t in pit_universe}
        data_dict = {
            "nifty50_data": nifty_df.loc[:date],
            "stock_data": relevant_stock_data,
        }
        result[date_str] = compute_universe_ranks(data_dict)
        if idx % 20 == 0:
            log.info(f"    {idx}/{len(sampled_dates)} dates ranked "
                     f"({len(relevant_stock_data)} point-in-time members)")
    return result


def run_replay(tickers: list = None, fresh: bool = False, test_all_patterns: bool = False) -> None:
    _ensure_table()
    _ensure_progress_table()
    _print_cost_model()

    if test_all_patterns:
        log.info("--all-patterns given: testing all 19 detected patterns (weighted + pruned) "
                 "independently through the real gate chain, not just the 5 pre-weighted ones.")

    if fresh:
        log.info("--fresh given: clearing existing checkpoint, starting over.")
        _clear_progress()

    _check_or_set_mode(test_all_patterns)  # raises if this run's mode conflicts with existing checkpoint

    completed = _load_completed_tickers()
    if completed:
        log.info(f"Resuming from checkpoint: {len(completed)} tickers already completed, "
                 f"will be skipped. (Use --fresh to ignore checkpoint and start over.)")

    meta = _build_ticker_meta()
    all_tickers = _all_deep_tickers()
    if tickers:
        all_tickers = [t for t in all_tickers if t in tickers]

    log.info("Loading NIFTY history from market_regime_history (deep-backfilled)...")
    nifty_df = _load_nifty_history()
    regime_map = _load_regime_map()

    conn = get_connection()
    earliest_snapshot_date = conn.execute(
        "SELECT MIN(effective_date) FROM universe_snapshots"
    ).fetchone()[0]
    conn.close()
    if not earliest_snapshot_date:
        raise RuntimeError("universe_snapshots is empty — run load_deep_history.py --universe-csv first.")
    log.info(f"Point-in-time universe coverage starts {earliest_snapshot_date} — "
             f"replay will only test dates from here onward.")

    log.info(f"Loading price history for {len(all_tickers)} tickers "
             "(needed for RS rank precomputation)...")
    stock_data = {}
    for ticker in all_tickers:
        df = _load_price_history_deep(ticker)
        if not df.empty:
            stock_data[ticker] = df

    all_dates = sorted(nifty_df.index)
    universe_ranks_by_date = _precompute_universe_ranks_by_date(
        all_dates, stock_data, nifty_df, earliest_snapshot_date
    )

    remaining = [t for t in all_tickers if t not in completed]
    log.info(f"Running deep-history full-pipeline replay on {len(remaining)} remaining "
             f"of {len(all_tickers)} total tickers...")

    total_signals = sum(r["signals"] for r in _load_all_progress())

    for i, ticker in enumerate(remaining, 1):
        name, sector, tier = meta.get(ticker, (ticker.replace(".NS", ""), "Unknown", DEFAULT_TIER))
        try:
            res = replay_ticker(ticker, name, sector, tier, nifty_df,
                                 universe_ranks_by_date, regime_map, earliest_snapshot_date,
                                 test_all_patterns=test_all_patterns)
        except Exception as e:
            log.warning(f"  {ticker} CRASHED during replay ({type(e).__name__}: {e}) — skipping, continuing run")
            continue

        _save_ticker_progress(ticker, res)  # committed immediately — a kill after this point is not lost
        total_signals += res["signals"]
        if i % 50 == 0:
            log.info(f"  {i}/{len(remaining)} remaining | {total_signals:,} gate-cleared signals so far")

    all_results = _load_all_progress()
    aggregate_and_store(all_results)
    log.info(f"Deep replay complete: {total_signals:,} gate-cleared signals across {len(all_tickers)} tickers")
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
            INSERT OR REPLACE INTO pipeline_statistics_deep
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
        FROM pipeline_statistics_deep ORDER BY avg_r DESC
    """).fetchall()
    conn.close()

    print("\n" + "=" * 95)
    print("  DEEP-HISTORY FULL-PIPELINE REPLAY  (10yr point-in-time universe, all real gates, real costs)")
    print("  Compare against pipeline_statistics (2yr, current-universe-only) from earlier tonight")
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
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any existing checkpoint and start the replay over from scratch")
    parser.add_argument("--all-patterns", action="store_true",
                         help="Test all 19 detected patterns (weighted + pruned) independently "
                              "through the real gate chain, instead of only the 5 currently-"
                              "weighted patterns that can win live pattern selection.")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    elif args.ticker:
        t = args.ticker.upper() if args.ticker.endswith(".NS") else args.ticker.upper() + ".NS"
        run_replay(tickers=[t], fresh=args.fresh, test_all_patterns=args.all_patterns)
    else:
        run_replay(fresh=args.fresh, test_all_patterns=args.all_patterns)
