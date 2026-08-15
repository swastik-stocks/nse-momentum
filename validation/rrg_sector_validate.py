"""
validation/rrg_sector_validate.py — Factor-library backlog item validation

Tests RRGSectorAgent (agents/rrg_sector_agent.py) — Relative Rotation
Graph sector quadrant (LEADING/WEAKENING/LAGGING/IMPROVING) — against the
same gate-cleared signal set every other backlog item used (pattern + RS +
liquidity + risk + asymmetry + VCP, price_history_deep, 2007-2026),
reusing the real agent classes directly.

ARCHITECTURALLY DIFFERENT from every other item in this harness: RRG is a
sector-level signal, not a per-stock one. Each gate-cleared signal is
attributed the RRG quadrant of the SECTOR its stock belongs to on the
signal date (via sector_breadth.py's official NSE sector taxonomy), then
outcomes are bucketed by quadrant. The RRG literature's thesis: LEADING
(and to a lesser extent IMPROVING) sectors should produce stronger
forward returns than LAGGING ones.

HONEST SCOPE NOTE: the RS-Ratio/RS-Momentum formula here is an open-source
approximation of the proprietary JdK/StockCharts RRG algorithm (see the
agent's own docstring) — never validated on NSE sector composition before
this run, and the underlying sector index is itself a simple equal-
weighted synthetic construction, not an official NSE sector index.

Usage:
    python validation/rrg_sector_validate.py
    python validation/rrg_sector_validate.py RELIANCE
    python validation/rrg_sector_validate.py --stats
    python validation/rrg_sector_validate.py --fresh
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
from agents.rrg_sector_agent import build_sector_price_index, compute_rrg_quadrants
from agents.liquidity_agent import LiquidityAgent
from agents.risk_agent import RiskAgent
from agents.asymmetry_gate import AsymmetryGate
from agents.vcp_gate import VCPContractionGate
from sector_breadth import build_ticker_sector_mapping
from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from validation.backtest import ROUND_TRIP_COST_PCT, _print_cost_model

log = logging.getLogger(__name__)

_LOG_DIR = BASE_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = _LOG_DIR / f"rrg_sector_validate_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"

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
CREATE TABLE IF NOT EXISTS rrg_sector_validate_progress (
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
    conn.execute("DELETE FROM rrg_sector_validate_progress")
    conn.commit()
    conn.close()


def _load_completed_tickers() -> set:
    conn = get_connection()
    rows = conn.execute("SELECT ticker FROM rrg_sector_validate_progress").fetchall()
    conn.close()
    return {r["ticker"] for r in rows}


def _save_ticker_progress(ticker: str, res: dict) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO rrg_sector_validate_progress
        (ticker, signals, results_json, completed_at)
        VALUES (?,?,?,?)
    """, (ticker, res["signals"], json.dumps(res["results"]), datetime.today().isoformat()))
    conn.commit()
    conn.close()


def _load_all_progress() -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT ticker, signals, results_json FROM rrg_sector_validate_progress"
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


class SectorQuadrantLookup:
    """Per-sector quadrant series -> {date_str: quadrant} lookup, nearest
    date on-or-before the requested date (same bisect convention as
    DateIndexedRanks, since sector-index alignment can leave sparse gaps
    on days a sector's member set was thin)."""

    def __init__(self, quadrants_by_sector: dict):
        self._by_sector = {}
        for sector, series in quadrants_by_sector.items():
            clean = series.dropna()
            date_strs = [d.strftime("%Y-%m-%d") for d in clean.index]
            self._by_sector[sector] = (date_strs, clean.values.tolist())

    def get(self, sector: str, date_str: str):
        entry = self._by_sector.get(sector)
        if entry is None:
            return None
        date_strs, values = entry
        idx = bisect.bisect_right(date_strs, date_str) - 1
        if idx < 0:
            return None
        return values[idx]


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


def replay_ticker(ticker: str, tier: str, sector: str, nifty_df: pd.DataFrame,
                   universe_ranks_by_date: DateIndexedRanks,
                   quadrant_lookup: SectorQuadrantLookup,
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

        quadrant = quadrant_lookup.get(sector, date_str)  # None if missing — not zero-filled

        regime = _regime_for_date(regime_map, date_str)
        penalty = int(REGIME_PENALTIES.get(regime, -5) * cfg["regime_penalty_mult"])

        out = _forward_test_net(df, i, entry)
        if out is None:
            continue

        result = {
            "date": date_str, "pattern": pa.pattern, "regime": regime,
            "penalty_applied": penalty, "sector": sector, "quadrant": quadrant,
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

    log.info("Loading official NSE sector taxonomy...")
    ticker_to_sector, n_official, n_fallback = build_ticker_sector_mapping()
    log.info(f"  Sector source: {n_official} from official NSE list, {n_fallback} fallback")

    log.info(f"Loading deep price history for {len(universe_items)} stocks...")
    stock_data = {}
    for ticker, *_ in universe_items:
        df = _load_price_history_deep(ticker)
        if not df.empty:
            stock_data[ticker] = df

    all_dates_idx = nifty_df.index
    all_dates = sorted(all_dates_idx)
    universe_ranks_raw = _precompute_universe_ranks_by_date(all_dates, stock_data, nifty_df)
    universe_ranks_by_date = DateIndexedRanks(universe_ranks_raw)

    log.info("Building synthetic equal-weighted sector indices...")
    sector_index = build_sector_price_index(stock_data, ticker_to_sector, all_dates_idx)
    log.info(f"  {len(sector_index)} sector indices built")

    log.info("Computing RRG quadrants per sector (RS-Ratio / RS-Momentum)...")
    quadrants_by_sector = {}
    for sector, idx_series in sector_index.items():
        rrg_df = compute_rrg_quadrants(idx_series, nifty_df["Close"])
        quadrants_by_sector[sector] = rrg_df["quadrant"]
    quadrant_lookup = SectorQuadrantLookup(quadrants_by_sector)

    log.info(f"Running RRG sector-rotation validation on {len(remaining)} stocks...")
    total_signals = 0

    for i, (ticker, name, sector_nse_universe, tier) in enumerate(remaining, 1):
        sector = ticker_to_sector.get(ticker, sector_nse_universe)
        try:
            res = replay_ticker(ticker, tier, sector, nifty_df, universe_ranks_by_date,
                                 quadrant_lookup, regime_map)
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
    with_quad = [o for o in all_outcomes if o.get("quadrant") is not None]

    if len(with_quad) < 30:
        log.warning(f"Only {len(with_quad)} signals have a quadrant — too few to analyze.")
        return

    report = {"generated_at": datetime.today().isoformat(),
              "total_signals": len(all_outcomes), "with_quadrant": len(with_quad)}
    rng = np.random.default_rng(RANDOM_SEED)

    pool_r = np.array([o["net_r"] for o in with_quad], dtype=float)
    report["pool"] = _bucket_stats(with_quad)

    for q in ("LEADING", "IMPROVING", "WEAKENING", "LAGGING"):
        bucket = [o for o in with_quad if o["quadrant"] == q]
        stats = _bucket_stats(bucket)
        if len(bucket) >= 10:
            stats["p_value_vs_pool"] = round(
                _permutation_test(np.array([o["net_r"] for o in bucket]), pool_r, N_PERM, rng), 4)
        report[f"quadrant_{q}"] = stats

    leading = [o for o in with_quad if o["quadrant"] == "LEADING"]
    leading_sorted = sorted(leading, key=lambda o: o["date"])
    mid_split = len(leading_sorted) // 2
    report["leading_first_half"] = _bucket_stats(leading_sorted[:mid_split])
    report["leading_second_half"] = _bucket_stats(leading_sorted[mid_split:])

    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rrg_sector_validate_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_json TEXT,
            generated_at TEXT
        )
    """)
    conn.execute(
        "INSERT INTO rrg_sector_validate_results (report_json, generated_at) VALUES (?,?)",
        (json.dumps(report), report["generated_at"])
    )
    conn.commit()
    conn.close()
    log.info("Analysis stored in rrg_sector_validate_results.")


def print_stats() -> None:
    conn = get_connection()
    row = conn.execute(
        "SELECT report_json FROM rrg_sector_validate_results ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        print("No results yet — run without --stats first.")
        return
    report = json.loads(row["report_json"])

    print("\n" + "=" * 95)
    print("  FACTOR-LIBRARY BACKLOG (4) — RRG Sector Rotation (LEADING/WEAKENING/LAGGING/IMPROVING)")
    print("=" * 95)
    print(f"  Total gate-cleared signals: {report.get('total_signals', 0)} "
          f"({report.get('with_quadrant', 0)} with a computable quadrant)")
    for key, stats in report.items():
        if not isinstance(stats, dict):
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
