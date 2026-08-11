"""
validation/btst_backtest.py — Backtest of classify_btst()'s thresholds
(2026-08-11)

WHY: confirm_picks.py's classify_btst() (the 15:15 BTST scan) applies two
gates -- closing strength (CMP within 2% of day's high, full-day RVOL >=
1.5x) and R:R favorability (<70% of entry->T1 already captured) -- that
were REASONED OUT, not backtested, when BTST was built. This codebase's
whole culture (see agents/pattern_agent.py's docstring) is "don't trust a
pattern until it clears Monte Carlo" -- BTST shouldn't be the exception.

METHOD: reuses the SAME validated gate chain as pipeline_replay_deep.py
(PatternAgent, LiquidityAgent, RSAgent, RiskAgent, AsymmetryGate,
VCPContractionGate, point-in-time universe) -- deliberately NOT
reimplemented, to avoid any drift from the logic that already produced
Cup & Handle / Swing High Breakout's validated N=1119/N=1456 results. At
every bar where a signal clears the full gate chain (equivalent to a
live CONFIRMED pick), this script:

  1. Uses that SAME day's own OHLC bar as the "15:15 snapshot" proxy --
     Close ~= CMP at 15:15 (15 min before the 15:30 close), High = day's
     high so far (by definition, the day's real high already reflects
     any earlier intraday peak). No intraday minute data needed for
     these two fields specifically.
  2. Computes RVOL proxy = that day's Volume / mean(prior 20 days' Volume)
     -- the closest available proxy to classify_btst()'s elapsed-time
     RVOL without historical minute bars for the full universe.
  3. Computes pct_off_high, pct_captured (vs pattern's own entry/T1 from
     RiskAgent/AsymmetryGate) using the EXACT same formulas as
     classify_btst() in confirm_picks.py -- kept in sync manually (see
     _WOULD_PASS_BTST below); if those thresholds change, update both.
  4. Looks at the NEXT trading day's Open (pure overnight gap, what the
     ASM/GSM and earnings-eve gates specifically protect against) and
     Close (full T+1-exit-at-close, the actual BTST P&L) to compute two
     realized, cost-adjusted outcomes.

Then compares: does the subset of signals that WOULD have cleared
classify_btst()'s gates show a better next-day outcome than the subset
that would NOT have, and than the unfiltered baseline? Same bootstrap +
permutation approach as validation/monte_carlo_significance.py.

Checkpoints to a NEW table (btst_backtest_progress) -- fully additive,
never touches pipeline_replay_deep_progress or any existing validated
data.

Usage:
    python validation/btst_backtest.py                 # full run, resumable
    python validation/btst_backtest.py --fresh          # ignore checkpoint
    python validation/btst_backtest.py RELIANCE TCS     # specific tickers
    python validation/btst_backtest.py --stats          # analyze existing checkpoint only
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

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
from validation.backtest import ROUND_TRIP_COST_PCT
from load_deep_history import get_point_in_time_universe
from validation.pipeline_replay_deep import (
    _load_price_history_deep, _load_nifty_history, _load_regime_map,
    _regime_for_date, _all_deep_tickers, _build_ticker_meta,
    _precompute_universe_ranks_by_date, REGIME_PENALTIES, DEFAULT_TIER, STEP,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Kept in sync with confirm_picks.py's classify_btst() thresholds ────────
BTST_MIN_RVOL            = 1.5
BTST_MAX_PCT_OFF_HIGH    = 2.0
BTST_MAX_PCT_T1_CAPTURED = 70.0
RVOL_LOOKBACK_DAYS       = 20

_PROGRESS_DDL = """
CREATE TABLE IF NOT EXISTS btst_backtest_progress (
    ticker        TEXT PRIMARY KEY,
    signals       INTEGER,
    trades_json   TEXT,
    completed_at  TEXT
)
"""


def _ensure_table():
    conn = get_connection()
    conn.execute(_PROGRESS_DDL)
    conn.commit()
    conn.close()


def _load_completed_tickers() -> set:
    conn = get_connection()
    rows = conn.execute("SELECT ticker FROM btst_backtest_progress").fetchall()
    conn.close()
    return {r["ticker"] for r in rows}


def _save_ticker_progress(ticker: str, trades: list) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO btst_backtest_progress
        (ticker, signals, trades_json, completed_at)
        VALUES (?,?,?,?)
    """, (ticker, len(trades), json.dumps(trades), datetime.today().isoformat()))
    conn.commit()
    conn.close()


def _load_all_progress() -> list:
    conn = get_connection()
    rows = conn.execute("SELECT ticker, trades_json FROM btst_backtest_progress").fetchall()
    conn.close()
    trades = []
    for r in rows:
        trades.extend(json.loads(r["trades_json"]))
    return trades


def _clear_progress():
    conn = get_connection()
    conn.execute(_PROGRESS_DDL)
    conn.execute("DELETE FROM btst_backtest_progress")
    conn.commit()
    conn.close()


def replay_ticker_btst(ticker: str, tier: str, nifty_df: pd.DataFrame,
                        universe_ranks_by_date: dict, regime_map: dict,
                        earliest_snapshot_date: str) -> list:
    """Same gate chain as pipeline_replay_deep.replay_ticker(), but records
    BTST-relevant proxies + next-day outcome instead of the multi-day
    forward test. Only the single pre-weighted pattern per bar (matches
    what actually reaches CONFIRMED live -- Cup & Handle / Swing High
    Breakout, the only two non-zero DEFAULT_WEIGHTS)."""
    df = _load_price_history_deep(ticker)
    if len(df) < 80:
        return []

    cfg = UNIVERSE_CONFIG[tier]
    trades = []

    for i in range(60, len(df) - 1, STEP):  # -1: need at least 1 future bar
        date_str = df.index[i].strftime("%Y-%m-%d")

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
        if not pa.pattern:
            continue

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
        breakout_level = pa.breakout_level
        entry = float(breakout_level) if breakout_level > 0 else last_close
        entry_low = last_close * 0.995
        entry_high = entry * 1.005

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

        sl = float(risk.stop)
        t1 = float(risk.target1)
        pivot = entry

        # ── BTST proxies from THIS bar's own OHLC (the "15:15 snapshot") ──
        close_i = float(df["Close"].iloc[i])
        high_i  = float(df["High"].iloc[i])
        vol_i   = float(df["Volume"].iloc[i])

        vol_hist = df["Volume"].iloc[max(0, i - RVOL_LOOKBACK_DAYS):i]
        rvol = float(round(vol_i / vol_hist.mean(), 2)) if len(vol_hist) >= 10 and vol_hist.mean() > 0 else -1.0

        pct_off_high = round(((high_i - close_i) / high_i) * 100, 2) if high_i > 0 else None
        pct_captured = round(((close_i - entry) / (t1 - entry)) * 100, 2) if t1 != entry else None

        would_pass_btst = bool(
            close_i > sl
            and close_i >= pivot
            and pct_off_high is not None and pct_off_high <= BTST_MAX_PCT_OFF_HIGH
            and rvol >= BTST_MIN_RVOL
            and pct_captured is not None and pct_captured < BTST_MAX_PCT_T1_CAPTURED
        )

        # ── Next trading day's realized outcome ────────────────────────────
        next_open  = float(df["Open"].iloc[i + 1])
        next_close = float(df["Close"].iloc[i + 1])
        gap_return_net    = round(((next_open - close_i) / close_i) - ROUND_TRIP_COST_PCT, 5)
        t1exit_return_net = round(((next_close - close_i) / close_i) - ROUND_TRIP_COST_PCT, 5)

        regime = _regime_for_date(regime_map, date_str)

        trades.append({
            "date": date_str,
            "pattern": pa.pattern,
            "regime": regime,
            "would_pass_btst": would_pass_btst,
            "pct_off_high": pct_off_high,
            "rvol": rvol,
            "pct_captured": pct_captured,
            "gap_return_net": gap_return_net,           # overnight gap only, cost-adjusted
            "t1exit_return_net": t1exit_return_net,      # full T+1-close exit, cost-adjusted
        })

    return trades


def run_backtest(tickers: list = None, fresh: bool = False) -> None:
    _ensure_table()

    if fresh:
        log.info("--fresh given: clearing existing checkpoint, starting over.")
        _clear_progress()

    completed = _load_completed_tickers()
    if completed:
        log.info(f"Resuming: {len(completed)} tickers already completed, will be skipped.")

    meta = _build_ticker_meta()
    all_tickers = _all_deep_tickers()
    if tickers:
        all_tickers = [t for t in all_tickers if t.replace(".NS", "") in tickers or t in tickers]

    log.info("Loading NIFTY history + regime map...")
    nifty_df = _load_nifty_history()
    regime_map = _load_regime_map()

    conn = get_connection()
    earliest_snapshot_date = conn.execute(
        "SELECT MIN(effective_date) FROM universe_snapshots"
    ).fetchone()[0]
    conn.close()
    if not earliest_snapshot_date:
        raise RuntimeError("universe_snapshots is empty.")

    log.info(f"Loading price history for {len(all_tickers)} tickers "
             "(needed for RS rank precomputation)...")
    stock_data = {}
    for ticker in all_tickers:
        pdf = _load_price_history_deep(ticker)
        if not pdf.empty:
            stock_data[ticker] = pdf

    all_dates = sorted(nifty_df.index)
    universe_ranks_by_date = _precompute_universe_ranks_by_date(
        all_dates, stock_data, nifty_df, earliest_snapshot_date
    )

    remaining = [t for t in all_tickers if t not in completed]
    log.info(f"Running BTST backtest on {len(remaining)} remaining of {len(all_tickers)} tickers...")

    total_signals = sum(1 for _ in _load_all_progress())
    for i, ticker in enumerate(remaining, 1):
        _, _, tier = meta.get(ticker, (ticker.replace(".NS", ""), "Unknown", DEFAULT_TIER))
        try:
            trades = replay_ticker_btst(ticker, tier, nifty_df, universe_ranks_by_date,
                                         regime_map, earliest_snapshot_date)
        except Exception as e:
            log.warning(f"  {ticker} CRASHED ({type(e).__name__}: {e}) — skipping")
            continue
        _save_ticker_progress(ticker, trades)
        total_signals += len(trades)
        if i % 50 == 0:
            log.info(f"  {i}/{len(remaining)} tickers | {total_signals:,} signals so far")

    log.info(f"BTST backtest complete: {total_signals:,} total signals recorded.")
    print_stats()


# ── Bootstrap + permutation stats, same approach as monte_carlo_significance.py ──

def _bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple:
    if len(values) < 10:
        return (None, None)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                       for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int = 10000, seed: int = 42) -> float:
    """Two-sided permutation test: is mean(a) - mean(b) distinguishable from
    a random relabeling of the pooled a+b values?"""
    if len(a) < 10 or len(b) < 10:
        return float("nan")
    rng = np.random.default_rng(seed)
    observed = a.mean() - b.mean()
    pooled = np.concatenate([a, b])
    n_a = len(a)
    diffs = np.empty(n_perm)
    for k in range(n_perm):
        rng.shuffle(pooled)
        diffs[k] = pooled[:n_a].mean() - pooled[n_a:].mean()
    return float(np.mean(np.abs(diffs) >= abs(observed)))


def print_stats():
    trades = _load_all_progress()
    if not trades:
        log.info("No BTST backtest trades recorded yet.")
        return

    df = pd.DataFrame(trades)
    log.info(f"\n{'='*100}\n  BTST BACKTEST RESULTS — classify_btst() thresholds vs T+1 outcomes\n{'='*100}")

    for metric in ("gap_return_net", "t1exit_return_net"):
        log.info(f"\n--- {metric} ---")
        approved = df[df["would_pass_btst"]][metric].to_numpy()
        rejected = df[~df["would_pass_btst"]][metric].to_numpy()
        baseline = df[metric].to_numpy()

        for label, arr in (("BTST-APPROVED", approved), ("BTST-REJECTED", rejected), ("ALL SIGNALS (baseline)", baseline)):
            if len(arr) == 0:
                log.info(f"  {label:24s} N=0")
                continue
            ci_low, ci_high = _bootstrap_ci(arr)
            win_rate = float((arr > 0).mean())
            log.info(f"  {label:24s} N={len(arr):5d}  mean={arr.mean()*100:+.3f}%  "
                     f"median={np.median(arr)*100:+.3f}%  win_rate={win_rate:.1%}  "
                     f"95% CI=[{ci_low*100 if ci_low is not None else float('nan'):+.3f}%, "
                     f"{ci_high*100 if ci_high is not None else float('nan'):+.3f}%]")

        if len(approved) >= 10 and len(rejected) >= 10:
            p = _permutation_p(approved, rejected)
            log.info(f"  Permutation p-value (APPROVED vs REJECTED, two-sided): {p:.4f} "
                     f"{'*** significant at 5% ***' if p < 0.05 else '(not distinguishable at 5%)'}")

    log.info(f"\nTotal signals: {len(df)}  |  Would pass BTST: {df['would_pass_btst'].sum()} "
             f"({df['would_pass_btst'].mean():.1%})")
    by_pattern = df.groupby("pattern").size()
    log.info(f"By pattern: {dict(by_pattern)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--stats" in args:
        print_stats()
    else:
        fresh = "--fresh" in args
        tickers = [a.upper() for a in args if not a.startswith("--")] or None
        run_backtest(tickers=tickers, fresh=fresh)
