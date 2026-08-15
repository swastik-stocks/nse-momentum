"""
validation/skew_penalty_validate.py — Factor-library backlog item validation

Tests SkewPenaltyAgent (agents/skew_penalty_agent.py) — isolated
idiosyncratic-skewness, one component of a public-repo composite score
(tanish35/Momentum-Investing: score = w_m*Momentum + w_f*FIP - w_s*Skewness)
tested standalone rather than blended — against the same gate-cleared
signal set every other backlog item used (pattern + RS + liquidity + risk
+ asymmetry + VCP, price_history_deep, 2007-2026), reusing the real agent
classes directly.

Two competing literature theses give OPPOSITE predicted signs here:
  - Chung (2019) lottery-demand: high skew attracts retail flow, predicts
    STRONGER continuation (this codebase's already-live Lottery Index
    found exactly this, blended with MAX/IVOL).
  - Skewness-preference overpricing (Barberis-Huang / Bali-Cakici-Whitelaw
    MAX-effect literature): investors overpay for lottery-like payoffs,
    predicts high-skew stocks have WEAKER forward returns.
This harness doesn't assume either — it buckets gate-cleared signals by
iskew tercile (post-hoc, computed across all collected signals) and
reports whichever direction (if any) the data actually shows.

Usage:
    python validation/skew_penalty_validate.py
    python validation/skew_penalty_validate.py RELIANCE
    python validation/skew_penalty_validate.py --stats
    python validation/skew_penalty_validate.py --fresh
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
from agents.skew_penalty_agent import compute_stock_skew_components, compute_universe_skew_index
from agents.liquidity_agent import LiquidityAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
from agents.vcp_gate import VCPContractionGate
from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from validation.backtest import ROUND_TRIP_COST_PCT, _print_cost_model

log = logging.getLogger(__name__)

_LOG_DIR = BASE_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = _LOG_DIR / f"skew_penalty_validate_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(_log_file, encoding="utf-8")]
)
log.info(f"Log file: {_log_file}")

FORWARD_BARS = 20
WIN_THRESHOLD = 0.05
STEP = 5
DEFAULT_TIER = "MID"
N_PERM = 10_000
RANDOM_SEED = 47

REGIME_PENALTIES = {"A": 0, "B": 0, "C": -5, "D": -12, "E": -20}

_PROGRESS_DDL = """
CREATE TABLE IF NOT EXISTS skew_penalty_validate_progress (
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
    conn.execute("DELETE FROM skew_penalty_validate_progress")
    conn.commit()
    conn.close()


def _load_completed_tickers() -> set:
    conn = get_connection()
    rows = conn.execute("SELECT ticker FROM skew_penalty_validate_progress").fetchall()
    conn.close()
    return {r["ticker"] for r in rows}


def _save_ticker_progress(ticker: str, res: dict) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO skew_penalty_validate_progress
        (ticker, signals, results_json, completed_at)
        VALUES (?,?,?,?)
    """, (ticker, res["signals"], json.dumps(res["results"]), datetime.today().isoformat()))
    conn.commit()
    conn.close()


def _load_all_progress() -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT ticker, signals, results_json FROM skew_penalty_validate_progress"
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
    df = pd.read_sql("SELECT date, nifty_close FROM market_regime_history ORDER BY date ASC", conn)
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


def _forward_test_net(df: pd.DataFrame, signal_idx: int, entry: float):
    future = df.iloc[signal_idx+1: signal_idx+1+FORWARD_BARS]
    if len(future) < FORWARD_BARS:
        return None
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
    def __init__(self, ranks_by_date: dict):
        self._data = ranks_by_date
        self._dates = sorted(ranks_by_date.keys())

    def get_day(self, date_str: str) -> dict:
        idx = bisect.bisect_right(self._dates, date_str) - 1
        if idx < 0:
            return {}
        return self._data[self._dates[idx]]


def _precompute_universe_ranks_by_date(dates: list, stock_data: dict, nifty_df: pd.DataFrame) -> dict:
    log.info("  Precomputing per-date universe RS ranks...")
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


def _precompute_skew_index_by_date(dates: list, components_by_ticker: dict, tickers: list) -> dict:
    log.info("  Precomputing per-date cross-sectional idiosyncratic skew...")
    result = {}
    sampled_dates = dates[::STEP]
    for idx, date in enumerate(sampled_dates):
        date_str = date.strftime("%Y-%m-%d")
        result[date_str] = compute_universe_skew_index(components_by_ticker, date, tickers)
        if idx % 20 == 0:
            n_valid = len(result[date_str])
            log.info(f"    {idx}/{len(sampled_dates)} dates ranked ({n_valid} valid members)")
    return result


def replay_ticker(ticker: str, tier: str, nifty_df: pd.DataFrame,
                   universe_ranks_by_date: DateIndexedRanks,
                   skew_index_by_date: DateIndexedRanks,
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
        if vcpg.check().get("hard_reject", False):
            continue

        skew_day = skew_index_by_date.get_day(date_str)
        iskew = skew_day.get(ticker)  # None if missing — not zero-filled

        regime = _regime_for_date(regime_map, date_str)
        penalty = int(REGIME_PENALTIES.get(regime, -5) * cfg["regime_penalty_mult"])

        out = _forward_test_net(df, i, entry)
        if out is None:
            continue

        result = {
            "date": date_str, "pattern": pa.pattern, "regime": regime,
            "penalty_applied": penalty, "iskew": iskew,
            "net_r": out["net_r"], "is_win": out["is_win"],
        }
        results.append(result)

    return {"ticker": ticker, "signals": len(results), "results": results}


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

    log.info("Loading NIFTY history + regime map...")
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

    log.info("Precomputing rolling idiosyncratic skew per ticker (vectorized, once each)...")
    components_by_ticker = {
        t: compute_stock_skew_components(df, nifty_df) for t, df in stock_data.items()
    }
    skew_index_raw = _precompute_skew_index_by_date(all_dates, components_by_ticker, list(stock_data.keys()))
    skew_index_by_date = DateIndexedRanks(skew_index_raw)

    log.info(f"Running skew-penalty validation on {len(remaining)} stocks...")
    total_signals = 0

    for i, (ticker, name, sector, tier) in enumerate(remaining, 1):
        try:
            res = replay_ticker(ticker, tier, nifty_df, universe_ranks_by_date,
                                 skew_index_by_date, regime_map)
        except Exception as e:
            log.warning(f"  {ticker} CRASHED ({type(e).__name__}: {e}) — skipping, continuing run")
            continue
        _save_ticker_progress(ticker, res)
        total_signals += res["signals"]
        if i % 25 == 0:
            log.info(f"  {i}/{len(remaining)} | {total_signals:,} signals so far")

    log.info(f"Replay complete: {total_signals:,} gate-cleared signals")
    analyze_and_store()
    print_stats()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _permutation_test(bucket_r: np.ndarray, pool_r: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    n = len(bucket_r)
    if n == 0 or len(pool_r) < n:
        return 1.0
    observed = float(np.mean(bucket_r))
    draws = np.array([np.mean(rng.choice(pool_r, size=n, replace=False)) for _ in range(n_perm)])
    return float(np.mean(draws >= observed))


def _bucket_stats(outcomes: list) -> dict:
    if not outcomes:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "profit_factor": 0.0}
    n = len(outcomes)
    wins = [o["net_r"] for o in outcomes if o["is_win"]]
    losses = [o["net_r"] for o in outcomes if not o["is_win"]]
    win_r = sum(wins)
    los_r = abs(sum(losses))
    return {
        "n": n,
        "win_rate": round(len(wins) / n, 3),
        "avg_r": round(sum(o["net_r"] for o in outcomes) / n, 3),
        "profit_factor": round(win_r / los_r, 2) if los_r > 0 else round(win_r, 2),
    }


def analyze_and_store() -> None:
    all_progress = _load_all_progress()
    all_outcomes = [o for res in all_progress for o in res["results"]]
    with_skew = [o for o in all_outcomes if o.get("iskew") is not None]

    if len(with_skew) < 30:
        log.warning(f"Only {len(with_skew)} signals have an iskew — too few to analyze.")
        return

    report = {"generated_at": datetime.today().isoformat(),
              "total_signals": len(all_outcomes), "with_iskew": len(with_skew)}
    rng = np.random.default_rng(RANDOM_SEED)

    pool_r = np.array([o["net_r"] for o in with_skew], dtype=float)
    report["pool"] = _bucket_stats(with_skew)

    values = np.array([o["iskew"] for o in with_skew])
    p33, p67 = np.percentile(values, [33, 67])
    top    = [o for o in with_skew if o["iskew"] >= p67]   # most positively skewed
    mid    = [o for o in with_skew if p33 < o["iskew"] < p67]
    bottom = [o for o in with_skew if o["iskew"] <= p33]   # most negatively skewed

    report["tercile_cutoffs"] = {"p33": round(float(p33), 3), "p67": round(float(p67), 3)}

    top_stats = _bucket_stats(top)
    bottom_stats = _bucket_stats(bottom)
    mid_stats = _bucket_stats(mid)
    if len(top) >= 10:
        top_stats["p_value_vs_pool"] = round(
            _permutation_test(np.array([o["net_r"] for o in top]), pool_r, N_PERM, rng), 4)
    if len(bottom) >= 10:
        bottom_stats["p_value_vs_pool"] = round(
            _permutation_test(np.array([o["net_r"] for o in bottom]), pool_r, N_PERM, rng), 4)
    report["top_tercile_skew"] = top_stats       # most positive skew
    report["mid_tercile_skew"] = mid_stats
    report["bottom_tercile_skew"] = bottom_stats  # most negative skew

    top_sorted = sorted(top, key=lambda o: o["date"])
    mid_split = len(top_sorted) // 2
    report["top_tercile_first_half"] = _bucket_stats(top_sorted[:mid_split])
    report["top_tercile_second_half"] = _bucket_stats(top_sorted[mid_split:])

    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skew_penalty_validate_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_json TEXT,
            generated_at TEXT
        )
    """)
    conn.execute(
        "INSERT INTO skew_penalty_validate_results (report_json, generated_at) VALUES (?,?)",
        (json.dumps(report), report["generated_at"])
    )
    conn.commit()
    conn.close()
    log.info("Analysis stored in skew_penalty_validate_results.")


def print_stats() -> None:
    conn = get_connection()
    row = conn.execute(
        "SELECT report_json FROM skew_penalty_validate_results ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        print("No results yet — run without --stats first.")
        return
    report = json.loads(row["report_json"])

    print("\n" + "=" * 95)
    print("  FACTOR-LIBRARY BACKLOG (2) — Skewness-Penalized Composite (isolated idiosyncratic skew)")
    print("=" * 95)
    print(f"  Total gate-cleared signals: {report.get('total_signals', 0)} "
          f"({report.get('with_iskew', 0)} with a computable iskew)")
    if "tercile_cutoffs" in report:
        print(f"  Tercile cutoffs (post-hoc, this run): {report['tercile_cutoffs']}")
    for key, stats in report.items():
        if not isinstance(stats, dict) or key == "tercile_cutoffs":
            continue
        n = stats.get("n", 0)
        wr = stats.get("win_rate", 0) * 100
        ar = stats.get("avg_r", 0)
        pf = stats.get("profit_factor", 0)
        pv = stats.get("p_value_vs_pool")
        pv_str = f" p={pv}" if pv is not None else ""
        print(f"  {key:<28} N={n:>5}  WR={wr:>5.1f}%  AvgR={ar:>7.2f}  PF={pf:>5.2f}{pv_str}")
    print("=" * 95 + "\n")


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
