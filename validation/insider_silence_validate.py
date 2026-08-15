"""
validation/insider_silence_validate.py — Factor-library item 5 validation

Tests InsiderSilenceAgent (agents/insider_silence_agent.py) — Ma (2013)'s
silence-vs-traded classification — against the same gate-cleared signal
set items 1-4 used (pattern + RS + liquidity + risk + asymmetry + VCP,
price_history_deep, 2007-2026), reusing the real agent classes directly.

HONEST SCOPE NOTE (see also the agent's own docstring): Ma's original
finding was at a multi-year horizon. This harness computes BOTH:
  - net_r_20:  ~1-month forward outcome, same FORWARD_BARS convention as
               items 1-4, for comparability with those results.
  - net_r_250: ~1-year forward outcome, closer to Ma's actual "year 2+"
               divergence claim. Requires 250 future bars to exist, so
               signals from roughly the last year of available history
               are excluded from this measure (not from net_r_20).

Data: insider_transactions table (416/500 tickers, 2015-2026, bulk-loaded
by load_shareholding_insider_history.py from NSE's corporates-pit).

Usage:
    python validation/insider_silence_validate.py
    python validation/insider_silence_validate.py RELIANCE
    python validation/insider_silence_validate.py --stats
    python validation/insider_silence_validate.py --fresh
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
from agents.insider_silence_agent import InsiderSilenceAgent
from agents.liquidity_agent import LiquidityAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
from agents.vcp_gate import VCPContractionGate
from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from validation.backtest import ROUND_TRIP_COST_PCT, _print_cost_model

log = logging.getLogger(__name__)

_LOG_DIR = BASE_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = _LOG_DIR / f"insider_silence_validate_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ]
)
log.info(f"Log file: {_log_file}")

FORWARD_BARS_SHORT = 20    # ~1 month, matches items 1-4
FORWARD_BARS_LONG  = 250   # ~1 year, closer to Ma's actual horizon
WIN_THRESHOLD = 0.05
STEP = 5
DEFAULT_TIER = "MID"
N_PERM = 10_000
RANDOM_SEED = 46

REGIME_PENALTIES = {"A": 0, "B": 0, "C": -5, "D": -12, "E": -20}

_PROGRESS_DDL = """
CREATE TABLE IF NOT EXISTS insider_silence_validate_progress (
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
    conn.execute("DELETE FROM insider_silence_validate_progress")
    conn.commit()
    conn.close()


def _load_completed_tickers() -> set:
    conn = get_connection()
    rows = conn.execute("SELECT ticker FROM insider_silence_validate_progress").fetchall()
    conn.close()
    return {r["ticker"] for r in rows}


def _save_ticker_progress(ticker: str, res: dict) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO insider_silence_validate_progress
        (ticker, signals, results_json, completed_at)
        VALUES (?,?,?,?)
    """, (ticker, res["signals"], json.dumps(res["results"]), datetime.today().isoformat()))
    conn.commit()
    conn.close()


def _load_all_progress() -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT ticker, signals, results_json FROM insider_silence_validate_progress"
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


def _forward_test_net(df: pd.DataFrame, signal_idx: int, entry: float, forward_bars: int):
    future = df.iloc[signal_idx+1: signal_idx+1+forward_bars]
    if len(future) < forward_bars:
        return None   # not enough future data — excluded, not zero-filled
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
    """Nearest-prior-date lookup — see fifty_two_wk_high_validate.py's own
    DateIndexedRanks for the full rationale."""

    def __init__(self, ranks_by_date: dict):
        self._data = ranks_by_date
        self._dates = sorted(ranks_by_date.keys())

    def get_day(self, date_str: str) -> dict:
        idx = bisect.bisect_right(self._dates, date_str) - 1
        if idx < 0:
            return {}
        return self._data[self._dates[idx]]


def _load_insider_txns_by_ticker() -> dict:
    conn = get_connection()
    rows = conn.execute(
        "SELECT ticker, disclosure_dt, transaction_type, buy_value, sell_value "
        "FROM insider_transactions ORDER BY ticker, disclosure_dt"
    ).fetchall()
    conn.close()
    by_ticker = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append({
            "disclosure_dt": r["disclosure_dt"],
            "transaction_type": r["transaction_type"],
            "buy_value": r["buy_value"],
            "sell_value": r["sell_value"],
        })
    return by_ticker


def replay_ticker(ticker: str, tier: str, nifty_df: pd.DataFrame,
                   universe_ranks_by_date: DateIndexedRanks, regime_map: dict,
                   insider_txns: list) -> dict:
    df = _load_price_history_deep(ticker)
    if len(df) < 150:
        return {"ticker": ticker, "signals": 0, "results": []}

    cfg = UNIVERSE_CONFIG.get(tier, UNIVERSE_CONFIG[DEFAULT_TIER])
    results = []

    for i in range(150, len(df) - FORWARD_BARS_SHORT, STEP):
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

        # ── NEW: item 5 diagnostic, computed on the SAME gate-cleared signal ──
        isa = InsiderSilenceAgent(insider_txns, as_of=date_str)

        regime = _regime_for_date(regime_map, date_str)
        penalty = int(REGIME_PENALTIES.get(regime, -5) * cfg["regime_penalty_mult"])

        out_short = _forward_test_net(df, i, entry, FORWARD_BARS_SHORT)
        if out_short is None:
            continue
        out_long = _forward_test_net(df, i, entry, FORWARD_BARS_LONG)

        result = {
            "date": date_str,
            "pattern": pa.pattern,
            "regime": regime,
            "penalty_applied": penalty,
            "traded": isa.get_traded(),
            "n_buys": isa.get_n_buys(),
            "n_sells": isa.get_n_sells(),
            "net_r_20": out_short["net_r"],
            "is_win_20": out_short["is_win"],
        }
        if out_long is not None:
            result["net_r_250"] = out_long["net_r"]
            result["is_win_250"] = out_long["is_win"]
        results.append(result)

    return {"ticker": ticker, "signals": len(results), "results": results}


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

    log.info("Loading insider transaction history for all tickers...")
    insider_by_ticker = _load_insider_txns_by_ticker()
    tickers_with_data = sum(1 for t, *_ in universe_items if t in insider_by_ticker)
    log.info(f"  {tickers_with_data}/{len(universe_items)} tickers have insider_transactions rows")

    log.info(f"Loading deep price history for {len(universe_items)} stocks...")
    stock_data = {}
    for ticker, *_ in universe_items:
        df = _load_price_history_deep(ticker)
        if not df.empty:
            stock_data[ticker] = df

    all_dates = sorted(nifty_df.index)
    universe_ranks_raw = _precompute_universe_ranks_by_date(all_dates, stock_data, nifty_df)
    universe_ranks_by_date = DateIndexedRanks(universe_ranks_raw)

    log.info(f"Running insider-silence (item 5) validation on {len(remaining)} stocks...")
    total_signals = 0

    for i, (ticker, name, sector, tier) in enumerate(remaining, 1):
        try:
            res = replay_ticker(ticker, tier, nifty_df, universe_ranks_by_date, regime_map,
                                 insider_by_ticker.get(ticker, []))
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
    draws = np.array([
        np.mean(rng.choice(pool_r, size=n, replace=False)) for _ in range(n_perm)
    ])
    return float(np.mean(draws >= observed))


def _bucket_stats(outcomes: list, r_key: str) -> dict:
    vals = [o[r_key] for o in outcomes if r_key in o]
    if not vals:
        return {"n": 0, "win_rate": 0.0, "avg_r": 0.0, "profit_factor": 0.0}
    n = len(vals)
    win_key = r_key.replace("net_r", "is_win")
    wins_list = [o[r_key] for o in outcomes if r_key in o and o.get(win_key)]
    losses_list = [o[r_key] for o in outcomes if r_key in o and not o.get(win_key)]
    win_r = sum(wins_list)
    los_r = abs(sum(losses_list))
    return {
        "n": n,
        "win_rate": round(len(wins_list) / n, 3),
        "avg_r": round(sum(vals) / n, 3),
        "profit_factor": round(win_r / los_r, 2) if los_r > 0 else round(win_r, 2),
    }


def analyze_and_store() -> None:
    all_progress = _load_all_progress()
    all_outcomes = [o for res in all_progress for o in res["results"]]
    if not all_outcomes:
        log.warning("No signals found — nothing to analyze.")
        return

    report = {"generated_at": datetime.today().isoformat(), "total_signals": len(all_outcomes)}
    rng = np.random.default_rng(RANDOM_SEED)

    for r_key in ("net_r_20", "net_r_250"):
        with_r = [o for o in all_outcomes if r_key in o]
        if len(with_r) < 30:
            log.warning(f"{r_key}: only {len(with_r)} signals — too few to analyze.")
            continue
        pool_r = np.array([o[r_key] for o in with_r], dtype=float)
        report[f"pool_{r_key}"] = _bucket_stats(with_r, r_key)

        traded = [o for o in with_r if o["traded"]]
        silent = [o for o in with_r if not o["traded"]]
        traded_stats = _bucket_stats(traded, r_key)
        silent_stats = _bucket_stats(silent, r_key)
        if len(traded) >= 10:
            traded_stats["p_value_vs_pool"] = round(
                _permutation_test(np.array([o[r_key] for o in traded]), pool_r, N_PERM, rng), 4)
        if len(silent) >= 10:
            silent_stats["p_value_vs_pool"] = round(
                _permutation_test(np.array([o[r_key] for o in silent]), pool_r, N_PERM, rng), 4)
        report[f"{r_key}__traded"] = traded_stats
        report[f"{r_key}__silent"] = silent_stats

        traded_sorted = sorted(traded, key=lambda o: o["date"])
        mid = len(traded_sorted) // 2
        report[f"{r_key}__traded_first_half"] = _bucket_stats(traded_sorted[:mid], r_key)
        report[f"{r_key}__traded_second_half"] = _bucket_stats(traded_sorted[mid:], r_key)

    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS insider_silence_validate_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_json TEXT,
            generated_at TEXT
        )
    """)
    conn.execute(
        "INSERT INTO insider_silence_validate_results (report_json, generated_at) VALUES (?,?)",
        (json.dumps(report), report["generated_at"])
    )
    conn.commit()
    conn.close()
    log.info("Analysis stored in insider_silence_validate_results.")


def print_stats() -> None:
    conn = get_connection()
    row = conn.execute(
        "SELECT report_json FROM insider_silence_validate_results ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        print("No results yet — run without --stats first.")
        return
    report = json.loads(row["report_json"])

    print("\n" + "=" * 95)
    print("  FACTOR-LIBRARY ITEM 5 VALIDATION — insider silence/traded (Ma 2013)")
    print("=" * 95)
    print(f"  Total gate-cleared signals: {report.get('total_signals', 0)}")
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
