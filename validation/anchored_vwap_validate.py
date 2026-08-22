"""
validation/anchored_vwap_validate.py — Factor-library validation

Tests AnchoredVWAPAgent (agents/anchored_vwap_agent.py) — the 72-bar
support-anchored VWAP distance from prorealcode.com's "Auto Midas
Anchored VWAP" indicator — against the same gate-cleared signal set
every other backlog item used (pattern + RS + liquidity + risk +
asymmetry + VCP, price_history_deep, 2007-2026), reusing the real agent
classes directly. Skeleton copied from validation/turnover_spike_validate.py,
now with the walk-forward check built in from the start (permanent
recipe as of tonight, not a retrofit).

Thesis: does a stock's distance above its own 72-bar support-anchored
VWAP (a "value zone" reference) carry a standalone edge on gate-cleared
signals? Direction NOT assumed -- this codebase's own 52-week-high
proximity test (a conceptually similar "distance from a reference
level" factor) INVERTED, so both high and low terciles are tested.

HONEST SCOPE NOTE: never independently validated on NSE microstructure
before this run, on either side. Only the support/"Bottom VWAP" side and
only the 72-bar level are tested -- the resistance side and the other 3
hierarchical levels (17/305/1292 bars) are explicitly out of scope for
this first, cheap test.

Usage:
    python validation/anchored_vwap_validate.py
    python validation/anchored_vwap_validate.py RELIANCE
    python validation/anchored_vwap_validate.py --stats
    python validation/anchored_vwap_validate.py --fresh
"""

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
from agents.anchored_vwap_agent import compute_stock_support_avwap_components, compute_universe_avwap_index
from agents.liquidity_agent import LiquidityAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
from agents.vcp_gate import VCPContractionGate
from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from validation.backtest import ROUND_TRIP_COST_PCT, _print_cost_model
from validation.monte_carlo_significance import bootstrap_test, permutation_test
from validation.walk_forward_significance import walk_forward_check, print_walk_forward_report

log = logging.getLogger(__name__)

_LOG_DIR = BASE_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = _LOG_DIR / f"anchored_vwap_validate_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"

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
HOLDOUT_FRAC = 0.30
MIN_HOLDOUT_BUCKET_N = 30

REGIME_PENALTIES = {"A": 0, "B": 0, "C": -5, "D": -12, "E": -20}

_PROGRESS_DDL = """
CREATE TABLE IF NOT EXISTS anchored_vwap_validate_progress (
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
    conn.execute("DELETE FROM anchored_vwap_validate_progress")
    conn.commit()
    conn.close()


def _load_completed_tickers() -> set:
    conn = get_connection()
    rows = conn.execute("SELECT ticker FROM anchored_vwap_validate_progress").fetchall()
    conn.close()
    return {r["ticker"] for r in rows}


def _save_ticker_progress(ticker: str, res: dict) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO anchored_vwap_validate_progress
        (ticker, signals, results_json, completed_at)
        VALUES (?,?,?,?)
    """, (ticker, res["signals"], json.dumps(res["results"]), datetime.today().isoformat()))
    conn.commit()
    conn.close()


def _load_all_progress() -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT ticker, signals, results_json FROM anchored_vwap_validate_progress"
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


def _precompute_avwap_index_by_date(dates: list, components_by_ticker: dict, tickers: list) -> dict:
    log.info("  Precomputing per-date cross-sectional AVWAP distance...")
    result = {}
    sampled_dates = dates[::STEP]
    for idx, date in enumerate(sampled_dates):
        date_str = date.strftime("%Y-%m-%d")
        result[date_str] = compute_universe_avwap_index(components_by_ticker, date, tickers)
        if idx % 20 == 0:
            n_valid = len(result[date_str])
            log.info(f"    {idx}/{len(sampled_dates)} dates ranked ({n_valid} valid members)")
    return result


def replay_ticker(ticker: str, tier: str, nifty_df: pd.DataFrame,
                   universe_ranks_by_date: DateIndexedRanks,
                   avwap_index_by_date: DateIndexedRanks,
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

        avwap_day = avwap_index_by_date.get_day(date_str)
        pct_above_support_avwap = avwap_day.get(ticker)  # None if missing — not zero-filled

        regime = _regime_for_date(regime_map, date_str)
        penalty = int(REGIME_PENALTIES.get(regime, -5) * cfg["regime_penalty_mult"])

        out = _forward_test_net(df, i, entry)
        if out is None:
            continue

        result = {
            "date": date_str, "pattern": pa.pattern, "regime": regime,
            "penalty_applied": penalty, "pct_above_support_avwap": pct_above_support_avwap,
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

    log.info("Precomputing rolling support-AVWAP distance per ticker (vectorized, once each)...")
    components_by_ticker = {
        t: compute_stock_support_avwap_components(df) for t, df in stock_data.items()
    }
    avwap_index_raw = _precompute_avwap_index_by_date(all_dates, components_by_ticker, list(stock_data.keys()))
    avwap_index_by_date = DateIndexedRanks(avwap_index_raw)

    log.info(f"Running anchored-VWAP validation on {len(remaining)} stocks...")
    total_signals = 0

    for i, (ticker, name, sector, tier) in enumerate(remaining, 1):
        try:
            res = replay_ticker(ticker, tier, nifty_df, universe_ranks_by_date,
                                 avwap_index_by_date, regime_map)
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


def _holdout_check(with_avwap: list, rng: np.random.Generator) -> dict:
    ordered = sorted(with_avwap, key=lambda o: o["date"])
    split_idx = int(round(len(ordered) * (1 - HOLDOUT_FRAC)))
    discovery, holdout = ordered[:split_idx], ordered[split_idx:]

    out = {"n_discovery": len(discovery), "n_holdout": len(holdout)}
    if len(discovery) < 30 or len(holdout) < 30:
        out["verdict"] = f"TOO THIN (discovery={len(discovery)}, holdout={len(holdout)}, need >=30 each)"
        return out

    disc_values = np.array([o["pct_above_support_avwap"] for o in discovery])
    p33, p67 = np.percentile(disc_values, [33, 67])
    out["discovery_tercile_cutoffs"] = {"p33": round(float(p33), 4), "p67": round(float(p67), 4)}

    hold_low = [o for o in holdout if o["pct_above_support_avwap"] <= p33]
    hold_high = [o for o in holdout if o["pct_above_support_avwap"] >= p67]
    hold_pool_r = np.array([o["net_r"] for o in holdout], dtype=float)

    for label, bucket in (("low_avwap_holdout", hold_low), ("high_avwap_holdout", hold_high)):
        if len(bucket) < MIN_HOLDOUT_BUCKET_N:
            out[label] = {"n": len(bucket), "verdict": f"TOO THIN (N < floor {MIN_HOLDOUT_BUCKET_N})"}
            continue
        r_values = [o["net_r"] for o in bucket]
        boot = bootstrap_test(r_values, n_boot=10_000, seed=RANDOM_SEED)
        _obs, _null_mean, _null_std, perm_p = permutation_test(
            np.array(r_values, dtype=float), hold_pool_r, N_PERM, rng
        )
        stats = _bucket_stats(bucket)
        stats["ci_low"] = round(boot["ci_low"], 3)
        stats["ci_high"] = round(boot["ci_high"], 3)
        stats["permutation_p_value"] = round(perm_p, 4)
        out[label] = stats

    return out


def analyze_and_store() -> None:
    all_progress = _load_all_progress()
    all_outcomes = [o for res in all_progress for o in res["results"]]
    with_avwap = [o for o in all_outcomes if o.get("pct_above_support_avwap") is not None]

    if len(with_avwap) < 30:
        log.warning(f"Only {len(with_avwap)} signals have a pct_above_support_avwap — too few to analyze.")
        return

    report = {"generated_at": datetime.today().isoformat(),
              "total_signals": len(all_outcomes), "with_avwap": len(with_avwap)}
    rng = np.random.default_rng(RANDOM_SEED)

    pool_r = np.array([o["net_r"] for o in with_avwap], dtype=float)
    report["pool"] = _bucket_stats(with_avwap)

    values = np.array([o["pct_above_support_avwap"] for o in with_avwap])
    p33, p67 = np.percentile(values, [33, 67])
    low_avwap  = [o for o in with_avwap if o["pct_above_support_avwap"] <= p33]
    mid        = [o for o in with_avwap if p33 < o["pct_above_support_avwap"] < p67]
    high_avwap = [o for o in with_avwap if o["pct_above_support_avwap"] >= p67]

    report["tercile_cutoffs"] = {"p33": round(float(p33), 4), "p67": round(float(p67), 4)}

    low_stats = _bucket_stats(low_avwap)
    high_stats = _bucket_stats(high_avwap)
    mid_stats = _bucket_stats(mid)
    if len(low_avwap) >= 10:
        low_stats["p_value_vs_pool"] = round(
            _permutation_test(np.array([o["net_r"] for o in low_avwap]), pool_r, N_PERM, rng), 4)
    if len(high_avwap) >= 10:
        high_stats["p_value_vs_pool"] = round(
            _permutation_test(np.array([o["net_r"] for o in high_avwap]), pool_r, N_PERM, rng), 4)
    report["low_avwap_tercile"] = low_stats     # close to / at the support level
    report["mid_tercile"] = mid_stats
    report["high_avwap_tercile"] = high_stats   # well above the support level

    high_sorted = sorted(high_avwap, key=lambda o: o["date"])
    mid_split = len(high_sorted) // 2
    report["high_avwap_first_half"] = _bucket_stats(high_sorted[:mid_split])
    report["high_avwap_second_half"] = _bucket_stats(high_sorted[mid_split:])

    report["holdout_check"] = _holdout_check(with_avwap, rng)

    wf_report = walk_forward_check(with_avwap, value_key="pct_above_support_avwap",
                                    n_windows=5, bucket_direction="high")
    report["walk_forward"] = wf_report

    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS anchored_vwap_validate_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_json TEXT,
            generated_at TEXT
        )
    """)
    conn.execute(
        "INSERT INTO anchored_vwap_validate_results (report_json, generated_at) VALUES (?,?)",
        (json.dumps(report), report["generated_at"])
    )
    conn.commit()
    conn.close()
    log.info("Analysis stored in anchored_vwap_validate_results.")


def print_stats() -> None:
    conn = get_connection()
    row = conn.execute(
        "SELECT report_json FROM anchored_vwap_validate_results ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        print("No results yet — run without --stats first.")
        return
    report = json.loads(row["report_json"])

    print("\n" + "=" * 95)
    print("  ANCHORED VWAP — 72-bar support-anchor distance (prorealcode.com, Auto Midas AVWAP)")
    print("=" * 95)
    print(f"  Total gate-cleared signals: {report.get('total_signals', 0)} "
          f"({report.get('with_avwap', 0)} with a computable AVWAP distance)")
    if "tercile_cutoffs" in report:
        print(f"  Tercile cutoffs (post-hoc, this run): {report['tercile_cutoffs']}")
    for key, stats in report.items():
        if not isinstance(stats, dict) or key in ("tercile_cutoffs", "holdout_check", "walk_forward"):
            continue
        n = stats.get("n", 0)
        wr = stats.get("win_rate", 0) * 100
        ar = stats.get("avg_r", 0)
        pf = stats.get("profit_factor", 0)
        pv = stats.get("p_value_vs_pool")
        pv_str = f" p={pv}" if pv is not None else ""
        print(f"  {key:<28} N={n:>5}  WR={wr:>5.1f}%  AvgR={ar:>7.2f}  PF={pf:>5.2f}{pv_str}")

    hc = report.get("holdout_check", {})
    print("  " + "-" * 91)
    print(f"  HOLDOUT CHECK (discovery N={hc.get('n_discovery', '?')}, holdout N={hc.get('n_holdout', '?')})")
    if "verdict" in hc:
        print(f"    {hc['verdict']}")
    for key in ("low_avwap_holdout", "high_avwap_holdout"):
        stats = hc.get(key)
        if not stats:
            continue
        if "verdict" in stats:
            print(f"    {key:<22} {stats['verdict']}")
        else:
            print(f"    {key:<22} N={stats['n']:>4}  AvgR={stats['avg_r']:>7.2f}  "
                  f"CI=[{stats.get('ci_low','?')}, {stats.get('ci_high','?')}]  "
                  f"p={stats.get('permutation_p_value','?')}")

    wf = report.get("walk_forward")
    if wf:
        print_walk_forward_report(wf, "AVWAP high tercile (well above support)")
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
