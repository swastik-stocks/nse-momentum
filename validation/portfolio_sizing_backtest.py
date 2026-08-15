"""
NSE Momentum v6 — Portfolio Sizing Backtest (factor-library backlog item:
Raju 2023 concentrated sizing, MAX_POSITIONS 10 -> 15)

WHY THIS EXISTS
    Every other backlog/factor-library item in this codebase gets evaluated
    against a real historical replay before any config/code change ships
    (see FACTOR_LIBRARY_IMPLEMENTATION_PLAN.md). MAX_POSITIONS (currently
    10, in portfolio_engine.py) had never actually been backtested at the
    portfolio level -- pipeline_replay_deep.py validates individual signals
    against the gate chain, but never tracks concurrent open positions, so
    it can't answer "what happens to the equity curve if more positions can
    be open at once." This script builds that missing piece: a genuine
    day-by-day concurrent-position simulator, reusing the SAME gate chain
    and point-in-time universe as pipeline_replay_deep.py.

TWO STAGES
    1. Signal generation (generate_signals): walks every ticker/date in
       price_history_deep (2016-2026 point-in-time universe), exactly like
       pipeline_replay_deep.py's replay_ticker but capturing the LIVE
       single-pattern-per-bar selection (test_all_patterns=False semantics)
       plus the RS score (0-20, from RSAgent.score() -- the single largest
       component of the live total_score, see orchestrator.py's Master
       score) as a ranking proxy, and entry/stop/target1 from RiskAgent.
       Stored flat (not per-ticker) in portfolio_sizing_signals so stage 2
       can do a global chronological pass across all tickers at once.

    2. Portfolio simulation (simulate_portfolio): walks the flat signal
       table in date order. On each date, newly-eligible signals compete
       for open slots (capped at MAX_POSITIONS, plus the same sector/
       universe caps portfolio_engine.py enforces live), ranked by RS
       score descending (ties broken by pattern weight). A filled slot
       holds for FORWARD_BARS (20) trading days on the SAME master
       calendar as market_regime_history, then exits and realizes the
       net_r outcome already computed by the gate-chain replay. Equity is
       tracked at trade-exit events (see simulate_portfolio's own
       docstring for the explicit compounding-model caveat -- this is a
       trade-level NAV, not a true daily mark-to-market, since interim
       price paths for open positions aren't reconstructed here).

HONEST SCOPE NOTE: this does not replicate orchestrator.py's full
    total_score (which also includes earnings-catalyst, macro regime,
    sector-breadth, and confirmation-agent components -- several of which
    have no historical-replay equivalent, e.g. EarningsCatalystAgent hits
    a live API). RS score is used as the ranking proxy because it's this
    codebase's single largest, most-validated scoring component (see
    factor-library item 1) and the one component every other validator in
    this repo already computes identically in a replay context. Results
    should be read as "does more concurrent capacity help, given the same
    RS-led ranking this codebase already trusts" -- not as a byte-for-byte
    replay of live tier/score output.

Usage:
    python validation/portfolio_sizing_backtest.py --generate --fresh
        # stage 1: build the flat signal table (slow, ~500 tickers)
    python validation/portfolio_sizing_backtest.py --simulate --max-positions 10
    python validation/portfolio_sizing_backtest.py --simulate --max-positions 15
        # stage 2: fast, re-run instantly against different MAX_POSITIONS
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
from agents.pattern_agent import PatternAgent, DEFAULT_WEIGHTS
from agents.rs_agent import RSAgent, compute_universe_ranks
from agents.liquidity_agent import LiquidityAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
from agents.vcp_gate import VCPContractionGate
from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from validation.backtest import ROUND_TRIP_COST_PCT, _print_cost_model
from validation.pipeline_replay_deep import (
    _load_price_history_deep, _load_nifty_history, _load_regime_map,
    _regime_for_date, _forward_test_net, _build_ticker_meta, _all_deep_tickers,
    _precompute_universe_ranks_by_date,
)
from load_deep_history import get_point_in_time_universe

log = logging.getLogger(__name__)

_LOG_DIR = BASE_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = _LOG_DIR / f"portfolio_sizing_backtest_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(_log_file, encoding="utf-8")]
)
log.info(f"Log file: {_log_file}")

FORWARD_BARS = 20
STEP = 5
REGIME_PENALTIES = {"A": 0, "B": 0, "C": -5, "D": -12, "E": -20}
DEFAULT_TIER = "MID"

# Baseline (current live) portfolio-construction caps -- see portfolio_engine.py
BASE_MAX_UNIVERSE_EXPOSURE = {"LARGE": 4, "MID": 4, "SMALL": 3}
BASE_MAX_SECTOR_EXPOSURE_PCT = 30.0

_SIGNALS_DDL = """
CREATE TABLE IF NOT EXISTS portfolio_sizing_signals (
    ticker        TEXT,
    date          TEXT,
    sector        TEXT,
    tier          TEXT,
    pattern       TEXT,
    rs_score      INTEGER,
    entry         REAL,
    stop          REAL,
    target1       REAL,
    net_r         REAL,
    is_win        INTEGER,
    regime        TEXT,
    PRIMARY KEY (ticker, date)
)
"""
_PROGRESS_DDL = """
CREATE TABLE IF NOT EXISTS portfolio_sizing_progress (
    ticker TEXT PRIMARY KEY, completed_at TEXT
)
"""


def _ensure_tables():
    conn = get_connection()
    conn.execute(_SIGNALS_DDL)
    conn.execute(_PROGRESS_DDL)
    conn.commit()
    conn.close()


def _clear_tables():
    conn = get_connection()
    conn.execute(_SIGNALS_DDL)
    conn.execute(_PROGRESS_DDL)
    conn.execute("DELETE FROM portfolio_sizing_signals")
    conn.execute("DELETE FROM portfolio_sizing_progress")
    conn.commit()
    conn.close()


def _load_completed() -> set:
    conn = get_connection()
    rows = conn.execute("SELECT ticker FROM portfolio_sizing_progress").fetchall()
    conn.close()
    return {r["ticker"] for r in rows}


def _save_signals(ticker: str, rows: list) -> None:
    conn = get_connection()
    conn.executemany("""
        INSERT OR REPLACE INTO portfolio_sizing_signals
        (ticker, date, sector, tier, pattern, rs_score, entry, stop, target1, net_r, is_win, regime)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, [
        (r["ticker"], r["date"], r["sector"], r["tier"], r["pattern"], r["rs_score"],
         r["entry"], r["stop"], r["target1"], r["net_r"], int(r["is_win"]), r["regime"])
        for r in rows
    ])
    conn.execute("INSERT OR REPLACE INTO portfolio_sizing_progress (ticker, completed_at) VALUES (?,?)",
                 (ticker, datetime.today().isoformat()))
    conn.commit()
    conn.close()


def replay_ticker_for_sizing(ticker: str, name: str, sector: str, tier: str,
                              nifty_df: pd.DataFrame, universe_ranks_by_date: dict,
                              regime_map: dict, earliest_snapshot_date: str) -> list:
    """
    Same gate chain as pipeline_replay_deep.replay_ticker, but:
      - only the single LIVE-selected pattern per bar (pa.pattern), not
        all 19 -- matches what a real scan would actually present as a
        candidate, since a stock can only occupy one portfolio slot.
      - captures RS score (0-20) and entry/stop/target1 for the portfolio
        simulator's ranking and sizing.
    """
    df = _load_price_history_deep(ticker)
    if len(df) < 80:
        return []

    cfg = UNIVERSE_CONFIG[tier]
    out = []

    for i in range(60, len(df) - FORWARD_BARS, STEP):
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
        entry = float(pa.breakout_level) if pa.breakout_level > 0 else last_close
        entry_low  = last_close * 0.995
        entry_high = entry * 1.005

        risk = RiskAgent(window, pa.breakout_level, entry_low, entry_high, universe=tier)
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
        if vcpg.check().get("hard_reject", False):
            continue

        regime = _regime_for_date(regime_map, date_str)
        result = _forward_test_net(df, i, entry)

        out.append({
            "ticker": ticker, "date": date_str, "sector": sector, "tier": tier,
            "pattern": pa.pattern, "rs_score": rsa.score(),
            "entry": risk.entry, "stop": risk.stop, "target1": risk.target1,
            "net_r": result["net_r"], "is_win": result["is_win"], "regime": regime,
        })

    return out


def generate_signals(tickers: list = None, fresh: bool = False) -> None:
    _ensure_tables()
    _print_cost_model()
    if fresh:
        log.info("--fresh given: clearing existing signal table, starting over.")
        _clear_tables()

    completed = _load_completed()
    if completed:
        log.info(f"Resuming: {len(completed)} tickers already done, will be skipped.")

    meta = _build_ticker_meta()
    all_tickers = _all_deep_tickers()
    if tickers:
        all_tickers = [t for t in all_tickers if t in tickers]

    log.info("Loading NIFTY history + regime map...")
    nifty_df = _load_nifty_history()
    regime_map = _load_regime_map()

    conn = get_connection()
    earliest_snapshot_date = conn.execute("SELECT MIN(effective_date) FROM universe_snapshots").fetchone()[0]
    conn.close()
    if not earliest_snapshot_date:
        raise RuntimeError("universe_snapshots is empty — run load_deep_history.py --universe-csv first.")

    log.info(f"Loading price history for {len(all_tickers)} tickers...")
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
    log.info(f"Generating portfolio-sizing signals for {len(remaining)} remaining of {len(all_tickers)} tickers...")

    total_signals = 0
    for i, ticker in enumerate(remaining, 1):
        name, sector, tier = meta.get(ticker, (ticker.replace(".NS", ""), "Unknown", DEFAULT_TIER))
        try:
            rows = replay_ticker_for_sizing(ticker, name, sector, tier, nifty_df,
                                             universe_ranks_by_date, regime_map, earliest_snapshot_date)
        except Exception as e:
            log.warning(f"  {ticker} CRASHED ({type(e).__name__}: {e}) — skipping, continuing run")
            continue
        _save_signals(ticker, rows)
        total_signals += len(rows)
        if i % 50 == 0:
            log.info(f"  {i}/{len(remaining)} | {total_signals:,} signals so far")

    log.info(f"Signal generation complete: {total_signals:,} signals generated this run.")


def simulate_portfolio(max_positions: int, scale_caps: bool = False) -> dict:
    """
    Trade-level equity-curve simulation.

    COMPOUNDING MODEL (explicit, so results aren't over-trusted): equity is
    marked only at trade-open and trade-close events, not daily. When a
    slot opens, it reserves current_NAV / max_positions from cash (where
    current_NAV = cash + reserved value of all still-open positions, held
    flat at their reservation value until they close — no interim
    mark-to-market, since this replay doesn't reconstruct each open
    position's intraday price path). At close, cash is credited
    reserved * (1 + net_r * WIN_THRESHOLD). This is a standard
    trade-return-compounding approximation, not a true daily NAV — read
    Sharpe/drawdown here as directionally comparative between the two
    max_positions settings tested side by side, not as an absolute
    forecast.

    scale_caps=True: sector/universe caps scaled by max_positions/10 (the
    ratio to the current live baseline) rather than left at their
    MAX_POSITIONS=10-tuned absolute values — otherwise a MAX_POSITIONS=15
    run could never actually use its extra capacity (the old caps sum to
    4+4+3=11, already close to 10).
    """
    from validation.backtest import WIN_THRESHOLD

    conn = get_connection()
    rows = conn.execute("""
        SELECT ticker, date, sector, tier, pattern, rs_score, net_r, is_win
        FROM portfolio_sizing_signals ORDER BY date ASC
    """).fetchall()
    conn.close()
    if not rows:
        raise RuntimeError("portfolio_sizing_signals is empty — run --generate first.")

    signals = [dict(r) for r in rows]
    dates = sorted(set(s["date"] for s in signals))
    date_idx = {d: i for i, d in enumerate(dates)}
    by_date = {}
    for s in signals:
        by_date.setdefault(s["date"], []).append(s)

    pat_weight = DEFAULT_WEIGHTS  # tie-break only

    if scale_caps:
        ratio = max_positions / 10.0
        universe_caps = {k: max(1, round(v * ratio)) for k, v in BASE_MAX_UNIVERSE_EXPOSURE.items()}
    else:
        universe_caps = dict(BASE_MAX_UNIVERSE_EXPOSURE)
    sector_cap_pct = BASE_MAX_SECTOR_EXPOSURE_PCT

    open_positions = []   # list of dicts: exit_date_idx, sector, tier, reserved, net_r
    cash = 1.0             # normalized starting NAV = 1.0
    trade_log = []         # (exit_date, nav_after) for equity curve / drawdown

    for d in dates:
        di = date_idx[d]

        # 1) close any positions whose holding period has elapsed
        still_open = []
        for p in open_positions:
            if p["exit_date_idx"] <= di:
                cash += p["reserved"] * (1 + p["net_r"] * WIN_THRESHOLD)
                trade_log.append((d, cash + sum(q["reserved"] for q in open_positions if q is not p)))
            else:
                still_open.append(p)
        open_positions = still_open

        # 2) admit new candidates into any free slots, ranked by RS score desc
        free_slots = max_positions - len(open_positions)
        if free_slots <= 0:
            continue
        candidates = sorted(
            by_date.get(d, []),
            key=lambda s: (s["rs_score"], pat_weight.get(s["pattern"], 0)),
            reverse=True
        )
        exit_idx = min(di + FORWARD_BARS, len(dates) - 1)
        exit_date = dates[exit_idx]

        for c in candidates:
            if free_slots <= 0:
                break
            sector, tier = c["sector"], c["tier"]
            uni_count = sum(1 for p in open_positions if p["tier"] == tier)
            if uni_count >= universe_caps.get(tier, 4):
                continue
            sec_count = sum(1 for p in open_positions if p["sector"] == sector)
            if (sec_count + 1) / max_positions * 100 > sector_cap_pct:
                continue

            nav = cash + sum(p["reserved"] for p in open_positions)
            reserved = nav / max_positions
            if reserved > cash:
                continue
            cash -= reserved
            open_positions.append({
                "exit_date_idx": date_idx.get(exit_date, di + FORWARD_BARS),
                "sector": sector, "tier": tier, "reserved": reserved,
                "net_r": c["net_r"],
            })
            free_slots -= 1

    # close anything still open at the end at its last known net_r (already realized outcome)
    for p in open_positions:
        cash += p["reserved"] * (1 + p["net_r"] * WIN_THRESHOLD)

    if not trade_log:
        return {"max_positions": max_positions, "n_trades": 0}

    nav_series = pd.Series([t[1] for t in trade_log], index=pd.to_datetime([t[0] for t in trade_log]))
    nav_series = nav_series.sort_index()
    running_max = nav_series.cummax()
    drawdown = (nav_series - running_max) / running_max
    max_dd = float(drawdown.min())

    total_return = cash - 1.0
    n_days = (nav_series.index[-1] - nav_series.index[0]).days
    years = max(n_days / 365.25, 0.1)
    cagr = (cash ** (1 / years)) - 1 if cash > 0 else -1.0

    trade_returns = nav_series.pct_change().dropna()
    sharpe = float(trade_returns.mean() / trade_returns.std() * np.sqrt(252 / max(1, FORWARD_BARS))
                   ) if trade_returns.std() > 0 else 0.0

    return {
        "max_positions": max_positions,
        "n_trades": len(trade_log),
        "final_nav": round(cash, 4),
        "total_return_pct": round(total_return * 100, 1),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "sharpe_approx": round(sharpe, 2),
        "universe_caps_used": universe_caps,
    }


def print_comparison(results: list) -> None:
    print("\n" + "=" * 90)
    print("  PORTFOLIO SIZING BACKTEST — MAX_POSITIONS comparison (Raju 2023 concentrated sizing)")
    print("=" * 90)
    print(f"  {'MAX_POS':>8} {'TRADES':>8} {'CAGR%':>8} {'MAXDD%':>8} {'SHARPE~':>9} {'FINAL NAV':>10}")
    print("  " + "-" * 84)
    for r in results:
        print(f"  {r['max_positions']:>8} {r['n_trades']:>8} {r['cagr_pct']:>8.2f} "
              f"{r['max_drawdown_pct']:>8.1f} {r['sharpe_approx']:>9.2f} {r['final_nav']:>10.3f}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Stage 1: build the flat signal table")
    parser.add_argument("--simulate", action="store_true", help="Stage 2: run the portfolio simulator")
    parser.add_argument("--fresh", action="store_true", help="Clear existing signal table before generating")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--scale-caps", action="store_true",
                         help="Scale sector/universe caps proportionally to max-positions")
    parser.add_argument("--compare", action="store_true",
                         help="Run both MAX_POSITIONS=10 (baseline caps) and =15 (scaled caps) and compare")
    args = parser.parse_args()

    if args.generate:
        generate_signals(fresh=args.fresh)
    elif args.compare:
        r10 = simulate_portfolio(10, scale_caps=False)
        r15 = simulate_portfolio(15, scale_caps=True)
        print_comparison([r10, r15])
    elif args.simulate:
        r = simulate_portfolio(args.max_positions, scale_caps=args.scale_caps)
        print_comparison([r])
    else:
        print("Usage:\n"
              "  python validation/portfolio_sizing_backtest.py --generate --fresh\n"
              "  python validation/portfolio_sizing_backtest.py --compare\n")
