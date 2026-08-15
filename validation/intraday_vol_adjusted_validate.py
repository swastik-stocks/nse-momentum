"""
validation/intraday_vol_adjusted_validate.py — Intraday vol-adjusted move
validation (10:05 / 12:30 / 3:15 BTST checkpoint intelligence)

Tests IntradayVolAdjustedAgent (agents/intraday_vol_adjusted_agent.py) —
see the conversation that scoped this out: an intraday analog of
factor-library item 1 (vol-adjusted momentum, already live and validated
in agents/rs_agent.py), combined with a path-efficiency read, against the
REAL historical population that matters for this question: signals that
would actually have reached classify_btst()'s CONFIRMED state.

Deliberately reuses validation/btst_backtest.py's replay_ticker_btst()
UNCHANGED (imported directly, not reimplemented) to identify that
population and its two outcome labels (gap_return_net, t1exit_return_net)
— this script only ADDS the intraday checkpoint reads on top of trades
that function already found, using price_history_hourly (531 tickers,
2015-02 to 2026-04, 7 bars/session at 09:00-15:00 IST) for real intraday
granularity instead of btst_backtest.py's own daily-OHLC proxy.

HONEST APPROXIMATIONS:
  - "12:30" is approximated by the 12:00 hourly bar — price_history_hourly
    only has bars on the hour, no 12:30 bar exists. A live check at 12:30
    would use a real live snapshot; this backtest is necessarily coarser.
  - Only trades where would_pass_btst == True are analyzed — this script
    answers "given a stock already reaching BTST-CONFIRMED, does this
    intraday read add information," not "should this gate the initial
    BTST confirmation itself."
  - trailing_daily_vol_pct uses the 20 trading days STRICTLY BEFORE the
    trade date (price_history_deep), avoiding any look-ahead into the
    trade day's own move.

Usage:
    python validation/intraday_vol_adjusted_validate.py
    python validation/intraday_vol_adjusted_validate.py RELIANCE
    python validation/intraday_vol_adjusted_validate.py --stats
    python validation/intraday_vol_adjusted_validate.py --fresh
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
from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from validation.backtest import _print_cost_model
from validation.pipeline_replay_deep import (
    _load_price_history_deep, _load_nifty_history, _load_regime_map,
    _build_ticker_meta, _all_deep_tickers, _precompute_universe_ranks_by_date,
    DEFAULT_TIER,
)
from validation.btst_backtest import replay_ticker_btst
from agents.intraday_vol_adjusted_agent import IntradayVolAdjustedAgent

log = logging.getLogger(__name__)

_LOG_DIR = BASE_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = _LOG_DIR / f"intraday_vol_adjusted_validate_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_file, encoding="utf-8"),
    ]
)
log.info(f"Log file: {_log_file}")

TRAILING_VOL_DAYS = 20
CHECKPOINTS = {"10:05": "10:00:00", "12:30": "12:00:00", "15:15": "15:00:00"}
N_PERM = 10_000
RANDOM_SEED = 44

_PROGRESS_DDL = """
CREATE TABLE IF NOT EXISTS intraday_vol_adjusted_validate_progress (
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
    conn.execute("DELETE FROM intraday_vol_adjusted_validate_progress")
    conn.commit()
    conn.close()


def _load_completed_tickers() -> set:
    conn = get_connection()
    rows = conn.execute("SELECT ticker FROM intraday_vol_adjusted_validate_progress").fetchall()
    conn.close()
    return {r["ticker"] for r in rows}


def _save_ticker_progress(ticker: str, res: dict) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO intraday_vol_adjusted_validate_progress
        (ticker, signals, results_json, completed_at)
        VALUES (?,?,?,?)
    """, (ticker, res["signals"], json.dumps(res["results"]), datetime.today().isoformat()))
    conn.commit()
    conn.close()


def _load_all_progress() -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT ticker, signals, results_json FROM intraday_vol_adjusted_validate_progress"
    ).fetchall()
    conn.close()
    return [
        {"ticker": r["ticker"], "signals": r["signals"], "results": json.loads(r["results_json"])}
        for r in rows
    ]


def _load_hourly_bars(ticker: str) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql(
        "SELECT datetime, open, high, low, close, volume FROM price_history_hourly "
        "WHERE ticker=? ORDER BY datetime ASC",
        conn, params=(ticker,)
    )
    conn.close()
    if df.empty:
        return df
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    return df


def _trailing_daily_vol(daily_df: pd.DataFrame, trade_date: str) -> float:
    """Stddev of daily returns over the TRAILING_VOL_DAYS trading days
    strictly before trade_date — no look-ahead into the trade day itself."""
    prior = daily_df.loc[:trade_date]
    if len(prior) > 0 and prior.index[-1].strftime("%Y-%m-%d") == trade_date:
        prior = prior.iloc[:-1]
    if len(prior) < TRAILING_VOL_DAYS + 1:
        return 0.0
    closes = prior["Close"].to_numpy(dtype=float)[-TRAILING_VOL_DAYS - 1:]
    rets = closes[1:] / closes[:-1] - 1
    return float(np.std(rets, ddof=1))


def analyze_ticker(ticker: str, tier: str, nifty_df: pd.DataFrame,
                    universe_ranks_by_date: dict, regime_map: dict,
                    earliest_snapshot_date: str) -> dict:
    trades = replay_ticker_btst(ticker, tier, nifty_df, universe_ranks_by_date,
                                 regime_map, earliest_snapshot_date)
    confirmed = [t for t in trades if t["would_pass_btst"]]
    if not confirmed:
        return {"ticker": ticker, "signals": 0, "results": []}

    daily_df = _load_price_history_deep(ticker)
    hourly_df = _load_hourly_bars(ticker)
    if hourly_df.empty:
        return {"ticker": ticker, "signals": 0, "results": []}
    hourly_by_date = {
        d: g.sort_values("Datetime") for d, g in
        hourly_df.groupby(hourly_df["Datetime"].dt.strftime("%Y-%m-%d"))
    }

    results = []
    for trade in confirmed:
        date_str = trade["date"]
        day_bars = hourly_by_date.get(date_str)
        if day_bars is None or day_bars.empty:
            continue
        day_open = float(day_bars["Open"].iloc[0])
        trailing_vol = _trailing_daily_vol(daily_df, date_str)

        checkpoint_reads = {}
        for label, bar_time in CHECKPOINTS.items():
            cutoff = pd.to_datetime(f"{date_str} {bar_time}")
            bars_upto = day_bars[day_bars["Datetime"] <= cutoff]
            if bars_upto.empty:
                continue
            agent = IntradayVolAdjustedAgent(
                bars_upto[["Close"]], day_open=day_open,
                trailing_daily_vol_pct=trailing_vol
            )
            if agent.get_bars_available() < 2:
                continue
            checkpoint_reads[label] = {
                "z": agent.get_z(),
                "efficiency": agent.get_efficiency(),
                "gate_pass": agent.passes_gate(),
            }

        if not checkpoint_reads:
            continue

        results.append({
            "date": date_str,
            "gap_return_net": trade["gap_return_net"],
            "t1exit_return_net": trade["t1exit_return_net"],
            "checkpoints": checkpoint_reads,
        })

    return {"ticker": ticker, "signals": len(results), "results": results}


def run_validate(tickers: list = None, fresh: bool = False) -> None:
    _ensure_progress_table()
    if fresh:
        _clear_progress()
    _print_cost_model()

    meta = _build_ticker_meta()
    all_tickers = _all_deep_tickers()
    if tickers:
        all_tickers = [t for t in all_tickers if t.replace(".NS", "") in tickers or t in tickers]

    completed = _load_completed_tickers()
    remaining = [t for t in all_tickers if t not in completed]
    log.info(f"{len(completed)} tickers already checkpointed, {len(remaining)} remaining.")

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

    log.info(f"Loading deep daily price history for {len(all_tickers)} tickers "
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

    log.info(f"Running intraday vol-adjusted validation on {len(remaining)} stocks...")
    total_signals = 0

    for i, ticker in enumerate(remaining, 1):
        _, _, tier = meta.get(ticker, (ticker.replace(".NS", ""), "Unknown", DEFAULT_TIER))
        try:
            res = analyze_ticker(ticker, tier, nifty_df, universe_ranks_by_date,
                                  regime_map, earliest_snapshot_date)
        except Exception as e:
            log.warning(f"  {ticker} CRASHED ({type(e).__name__}: {e}) — skipping, continuing run")
            continue
        _save_ticker_progress(ticker, res)
        total_signals += res["signals"]
        if i % 25 == 0:
            log.info(f"  {i}/{len(remaining)} | {total_signals:,} BTST-confirmed signals analyzed so far")

    log.info(f"Validation replay complete: {total_signals:,} BTST-confirmed signals with intraday reads")
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


def _bucket_stats(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"n": 0, "mean_pct": 0.0, "win_rate": 0.0}
    return {
        "n": len(values),
        "mean_pct": round(float(np.mean(values)) * 100, 3),
        "win_rate": round(float(np.mean(values > 0)), 3),
    }


def analyze_and_store() -> None:
    all_progress = _load_all_progress()
    all_signals = [s for res in all_progress for s in res["results"]]
    if not all_signals:
        log.warning("No BTST-confirmed signals with intraday reads found — nothing to analyze.")
        return

    report = {"generated_at": datetime.today().isoformat(), "total_signals": len(all_signals)}
    rng = np.random.default_rng(RANDOM_SEED)

    for outcome_key in ("gap_return_net", "t1exit_return_net"):
        pool = np.array([s[outcome_key] for s in all_signals], dtype=float)
        report[f"pool_{outcome_key}"] = _bucket_stats(pool)

        for label in CHECKPOINTS:
            with_reads = [s for s in all_signals if label in s["checkpoints"]]
            if len(with_reads) < 30:
                continue
            zs = np.array([s["checkpoints"][label]["z"] for s in with_reads])
            effs = np.array([s["checkpoints"][label]["efficiency"] for s in with_reads])
            outs = np.array([s[outcome_key] for s in with_reads], dtype=float)

            # Quadrant split: high |Z| (top half) x high efficiency (top half)
            abs_z = np.abs(zs)
            z_med = np.median(abs_z)
            e_med = np.median(effs)

            clean_strong = outs[(abs_z >= z_med) & (effs >= e_med)]
            choppy_loud  = outs[(abs_z >= z_med) & (effs < e_med)]
            quiet        = outs[(abs_z < z_med) & (effs < e_med)]

            for qname, qvals in (("clean_strong", clean_strong),
                                  ("choppy_loud", choppy_loud),
                                  ("quiet", quiet)):
                stats = _bucket_stats(qvals)
                if stats["n"] >= 10:
                    stats["p_value_vs_pool"] = round(
                        _permutation_test(qvals, pool, N_PERM, rng), 4)
                report[f"{outcome_key}__{label}__{qname}"] = stats

    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS intraday_vol_adjusted_validate_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_json TEXT,
            generated_at TEXT
        )
    """)
    conn.execute(
        "INSERT INTO intraday_vol_adjusted_validate_results (report_json, generated_at) VALUES (?,?)",
        (json.dumps(report), report["generated_at"])
    )
    conn.commit()
    conn.close()
    log.info("Analysis stored in intraday_vol_adjusted_validate_results.")


def print_stats() -> None:
    conn = get_connection()
    row = conn.execute(
        "SELECT report_json FROM intraday_vol_adjusted_validate_results ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        print("No results yet — run without --stats first.")
        return
    report = json.loads(row["report_json"])

    print("\n" + "=" * 100)
    print("  INTRADAY VOL-ADJUSTED MOVE VALIDATION — BTST-confirmed population, 3 checkpoints")
    print("=" * 100)
    print(f"  Total BTST-confirmed signals with intraday reads: {report.get('total_signals', 0)}")
    for key, stats in report.items():
        if not isinstance(stats, dict):
            continue
        n = stats.get("n", 0)
        mp = stats.get("mean_pct", 0)
        wr = stats.get("win_rate", 0) * 100
        pv = stats.get("p_value_vs_pool")
        pv_str = f" p={pv}" if pv is not None else ""
        print(f"  {key:<45} N={n:>5}  mean={mp:>+7.3f}%  WR={wr:>5.1f}%{pv_str}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", nargs="?", default=None)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    elif args.ticker:
        run_validate(tickers=[args.ticker.upper()], fresh=args.fresh)
    else:
        run_validate(fresh=args.fresh)
