"""
validation/fifty_two_wk_high_validate.py — Factor-library item 3 validation

Tests FiftyTwoWeekHighAgent (Raju 2023 + Anonymous 2024 cap-tier
weighting — see FACTOR_LIBRARY_IMPLEMENTATION_PLAN.md, Tier 2) against the
same gate-cleared signal set validation/tier1_factor_validate.py already
uses, same real agent classes, same deep multi-regime history
(price_history_deep, 2007-2026 via market_regime_history), same honest
scope limitation (current static universe, not point-in-time
reconstruction — see tier1_factor_validate.py's own docstring for the
full rationale, unchanged here).

For every signal that clears the gate chain, this additionally computes:
  - proximity_pct: FiftyTwoWeekHighAgent's price/52wk-high ratio
  - gate_pass: whether it clears the 85%-proximity diagnostic gate
then buckets the SAME forward net-R outcomes by proximity tercile and by
gate pass/fail, with a permutation test per bucket against the full pool
— identical methodology to tier1_factor_validate.py's item 1/2 analysis,
applied to this third factor.

Usage:
    python validation/fifty_two_wk_high_validate.py
    python validation/fifty_two_wk_high_validate.py RELIANCE
    python validation/fifty_two_wk_high_validate.py --stats
    python validation/fifty_two_wk_high_validate.py --fresh
"""

import sys
import json
import bisect
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
from agents.fifty_two_wk_high_agent import FiftyTwoWeekHighAgent
from agents.liquidity_agent import LiquidityAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
from agents.vcp_gate import VCPContractionGate
from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from validation.backtest import ROUND_TRIP_COST_PCT, _print_cost_model

log = logging.getLogger(__name__)

_LOG_DIR = BASE_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = _LOG_DIR / f"fifty_two_wk_high_validate_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ]
)
log.info(f"Log file: {_log_file}")

FORWARD_BARS = 20
WIN_THRESHOLD = 0.05
STEP = 5
DEFAULT_TIER = "MID"
N_PERM = 10_000
RANDOM_SEED = 43   # different seed from tier1_factor_validate.py — independent draws

REGIME_PENALTIES = {"A": 0, "B": 0, "C": -5, "D": -12, "E": -20}

_PROGRESS_DDL = """
CREATE TABLE IF NOT EXISTS fifty_two_wk_high_validate_progress (
    ticker        TEXT PRIMARY KEY,
    signals       INTEGER,
    results_json  TEXT,
    completed_at  TEXT
)
"""


def _ensure_progress_table():
    conn = get_connection()
    conn.execute(_PROGRESS_DDL)
    conn.commit()
    conn.close()


def _clear_progress():
    conn = get_connection()
    conn.execute(_PROGRESS_DDL)
    conn.execute("DELETE FROM fifty_two_wk_high_validate_progress")
    conn.commit()
    conn.close()


def _load_completed_tickers() -> set:
    conn = get_connection()
    rows = conn.execute("SELECT ticker FROM fifty_two_wk_high_validate_progress").fetchall()
    conn.close()
    return {r["ticker"] for r in rows}


def _save_ticker_progress(ticker: str, res: dict) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO fifty_two_wk_high_validate_progress
        (ticker, signals, results_json, completed_at)
        VALUES (?,?,?,?)
    """, (ticker, res["signals"], json.dumps(res["results"]), datetime.today().isoformat()))
    conn.commit()
    conn.close()


def _load_all_progress() -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT ticker, signals, results_json FROM fifty_two_wk_high_validate_progress"
    ).fetchall()
    conn.close()
    return [
        {"ticker": r["ticker"], "signals": r["signals"], "results": json.loads(r["results_json"])}
        for r in rows
    ]


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


class DateIndexedRanks:
    """Nearest-prior-date lookup — see tier1_factor_validate.py's own
    DateIndexedRanks docstring for why exact date matching fails here
    (mismatched calendars between the precompute and per-ticker replay)."""

    def __init__(self, ranks_by_date: dict):
        self._data = ranks_by_date
        self._dates = sorted(ranks_by_date.keys())

    def get_day(self, date_str: str) -> dict:
        idx = bisect.bisect_right(self._dates, date_str) - 1
        if idx < 0:
            return {}
        return self._data[self._dates[idx]]


def replay_ticker(ticker: str, name: str, sector: str, tier: str,
                   nifty_df: pd.DataFrame, universe_ranks_by_date: DateIndexedRanks,
                   regime_map: dict) -> dict:
    df = _load_price_history_deep(ticker)
    if len(df) < 150:
        return {"ticker": ticker, "signals": 0, "results": []}

    cfg = UNIVERSE_CONFIG.get(tier, UNIVERSE_CONFIG[DEFAULT_TIER])
    results = []

    for i in range(150, len(df) - FORWARD_BARS, STEP):
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

        day_ranks = universe_ranks_by_date.get_day(date_str)
        nifty_window = nifty_df.loc[:df.index[i]].copy()
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

        # ── NEW: item 3 diagnostics, computed on the SAME gate-cleared
        # signal — no effect on whether the signal qualifies.
        fwa = FiftyTwoWeekHighAgent(window, universe=tier)
        proximity_pct = fwa.get_proximity_pct() if fwa.get_bars_available() >= 150 else None
        gate_pass = fwa.passes_gate()

        regime = _regime_for_date(regime_map, date_str)
        penalty = int(REGIME_PENALTIES.get(regime, -5) * cfg["regime_penalty_mult"])

        outcome = _forward_test_net(df, i, entry)
        outcome.update({
            "date": date_str,
            "pattern": pa.pattern,
            "tier": tier,
            "regime": regime,
            "penalty_applied": penalty,
            "proximity_pct": proximity_pct,
            "gate_pass": gate_pass,
        })
        results.append(outcome)

    return {"ticker": ticker, "signals": len(results), "results": results}


def _precompute_universe_ranks_by_date(dates: list, stock_data: dict, nifty_df: pd.DataFrame) -> dict:
    log.info("  Precomputing per-date universe RS ranks (needed for the same RS gate "
              "the live scanner uses)...")
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


def run_replay(tickers: list = None, fresh: bool = False) -> None:
    _ensure_progress_table()
    if fresh:
        _clear_progress()
    _print_cost_model()

    if tickers is None:
        universe_items = list({s[0]: s for s in NSE_UNIVERSE}.values())
    else:
        universe_items = [s for s in NSE_UNIVERSE if s[0] in tickers]

    completed = _load_completed_tickers()
    remaining = [it for it in universe_items if it[0] not in completed]
    log.info(f"{len(completed)} tickers already checkpointed, {len(remaining)} remaining.")

    log.info("Loading NIFTY history from market_regime_history...")
    nifty_df = _load_nifty_history()
    regime_map = _load_regime_map()

    log.info(f"Loading deep price history for {len(universe_items)} stocks...")
    stock_data = {}
    for ticker, *_ in universe_items:
        df = _load_price_history_deep(ticker)
        if not df.empty:
            stock_data[ticker] = df

    all_dates = sorted(nifty_df.index)
    universe_ranks_raw = _precompute_universe_ranks_by_date(all_dates, stock_data, nifty_df)
    universe_ranks_by_date = DateIndexedRanks(universe_ranks_raw)

    log.info(f"Running 52-week-high factor validation replay on {len(remaining)} stocks...")
    total_signals = 0

    for i, (ticker, name, sector, tier) in enumerate(remaining, 1):
        try:
            res = replay_ticker(ticker, name, sector, tier, nifty_df, universe_ranks_by_date, regime_map)
        except Exception as e:
            log.warning(f"  {ticker} CRASHED ({type(e).__name__}: {e}) — skipping, continuing run")
            continue
        _save_ticker_progress(ticker, res)
        total_signals += res["signals"]
        if i % 25 == 0:
            log.info(f"  {i}/{len(remaining)} | {total_signals:,} gate-cleared signals so far")

    log.info(f"Replay complete: {total_signals:,} new gate-cleared signals")
    analyze_and_store()
    print_stats()


# ---------------------------------------------------------------------------
# Analysis — permutation test per factor bucket vs. the full pooled sample
# ---------------------------------------------------------------------------

def _permutation_test(bucket_r: np.ndarray, pool_r: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    n = len(bucket_r)
    if n == 0 or len(pool_r) < n:
        return 1.0
    observed = float(np.mean(bucket_r))
    draws = np.array([
        np.mean(rng.choice(pool_r, size=n, replace=False)) for _ in range(n_perm)
    ])
    return float(np.mean(draws >= observed))


def _bucket_stats(outcomes: list) -> dict:
    if not outcomes:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "profit_factor": 0.0}
    n = len(outcomes)
    wins = sum(1 for o in outcomes if o["is_win"])
    total_r = sum(o["net_r"] for o in outcomes)
    win_r = sum(o["net_r"] for o in outcomes if o["is_win"])
    los_r = abs(sum(o["net_r"] for o in outcomes if not o["is_win"]))
    return {
        "n": n,
        "win_rate": round(wins / n, 3),
        "avg_r": round(total_r / n, 3),
        "profit_factor": round(win_r / los_r, 2) if los_r > 0 else round(win_r, 2),
    }


def analyze_and_store() -> None:
    all_progress = _load_all_progress()
    all_outcomes = [o for res in all_progress for o in res["results"]]
    if not all_outcomes:
        log.warning("No gate-cleared signals found — nothing to analyze.")
        return

    df = pd.DataFrame(all_outcomes)
    pool_r = df["net_r"].to_numpy(dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)

    report = {"pool": _bucket_stats(all_outcomes), "generated_at": datetime.today().isoformat()}

    # ── Item 3: proximity-to-52wk-high, top vs bottom tercile ──
    pr = df.dropna(subset=["proximity_pct"])
    if len(pr) >= 30:
        top = pr[pr["proximity_pct"] >= pr["proximity_pct"].quantile(2/3)]
        bot = pr[pr["proximity_pct"] <= pr["proximity_pct"].quantile(1/3)]
        top_stats = _bucket_stats(top.to_dict("records"))
        bot_stats = _bucket_stats(bot.to_dict("records"))
        top_stats["p_value_vs_pool"] = round(
            _permutation_test(top["net_r"].to_numpy(dtype=float), pool_r, N_PERM, rng), 4)
        bot_stats["p_value_vs_pool"] = round(
            _permutation_test(bot["net_r"].to_numpy(dtype=float), pool_r, N_PERM, rng), 4)
        report["item3_proximity_top_tercile"] = top_stats
        report["item3_proximity_bottom_tercile"] = bot_stats

        top_sorted = top.sort_values("date")
        mid = len(top_sorted) // 2
        report["item3_top_tercile_first_half"] = _bucket_stats(top_sorted.iloc[:mid].to_dict("records"))
        report["item3_top_tercile_second_half"] = _bucket_stats(top_sorted.iloc[mid:].to_dict("records"))

        # Cap-tier breakdown of the top tercile — tests Anonymous (2024)'s
        # "effect concentrated in mid/small-cap" claim directly.
        for t in ("LARGE", "MID", "SMALL"):
            sub = top[top["tier"] == t]
            if len(sub) >= 15:
                report[f"item3_top_tercile_{t.lower()}"] = _bucket_stats(sub.to_dict("records"))
    else:
        log.warning(f"Item 3: only {len(pr)} signals have a proximity_pct — too few to bucket.")

    # ── gate pass/fail ──
    gp = df.dropna(subset=["proximity_pct"])
    if len(gp) >= 30:
        passed = gp[gp["gate_pass"] == True]
        failed = gp[gp["gate_pass"] == False]
        pass_stats = _bucket_stats(passed.to_dict("records"))
        fail_stats = _bucket_stats(failed.to_dict("records"))
        if len(passed) > 0:
            pass_stats["p_value_vs_pool"] = round(
                _permutation_test(passed["net_r"].to_numpy(dtype=float), pool_r, N_PERM, rng), 4)
        if len(failed) > 0:
            fail_stats["p_value_vs_pool"] = round(
                _permutation_test(failed["net_r"].to_numpy(dtype=float), pool_r, N_PERM, rng), 4)
        report["item3_gate_pass"] = pass_stats
        report["item3_gate_fail"] = fail_stats

    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fifty_two_wk_high_validate_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_json TEXT,
            generated_at TEXT
        )
    """)
    conn.execute(
        "INSERT INTO fifty_two_wk_high_validate_results (report_json, generated_at) VALUES (?,?)",
        (json.dumps(report), report["generated_at"])
    )
    conn.commit()
    conn.close()
    log.info("Analysis stored in fifty_two_wk_high_validate_results.")


def print_stats() -> None:
    conn = get_connection()
    row = conn.execute(
        "SELECT report_json FROM fifty_two_wk_high_validate_results ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        print("No results yet — run without --stats first.")
        return
    report = json.loads(row["report_json"])

    print("\n" + "=" * 90)
    print("  FACTOR-LIBRARY ITEM 3 VALIDATION — 52-week-high proximity (Raju 2023 + Anonymous 2024)")
    print("=" * 90)
    for key, stats in report.items():
        if not isinstance(stats, dict):
            continue
        n = stats.get("n", 0)
        wr = stats.get("win_rate", 0) * 100
        ar = stats.get("avg_r", 0)
        pf = stats.get("profit_factor", 0)
        pv = stats.get("p_value_vs_pool")
        pv_str = f" p={pv}" if pv is not None else ""
        print(f"  {key:<35} N={n:>5}  WR={wr:>5.1f}%  AvgR={ar:>7.2f}  PF={pf:>5.2f}{pv_str}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", nargs="?", default=None)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    elif args.ticker:
        run_replay(tickers=[args.ticker.upper() if args.ticker.endswith(".NS")
                             else args.ticker.upper() + ".NS"], fresh=args.fresh)
    else:
        run_replay(fresh=args.fresh)
