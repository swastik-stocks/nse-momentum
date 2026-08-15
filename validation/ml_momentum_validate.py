"""
validation/ml_momentum_validate.py — Factor-library backlog item validation

Tests MLMomentumModel (agents/ml_momentum_agent.py) — Beaudan & He (2019)'s
logistic-regression momentum classifier — against the same gate-cleared
signal set items 1-5 used (pattern + RS + liquidity + risk + asymmetry +
VCP, price_history_deep, 2007-2026), reusing the real agent classes
directly.

METHODOLOGY (ported from the paper's Section 3.2.3, adapted to a pooled
cross-sectional design):
    1. Generate a flat signal table: every gate-cleared candidate across
       the whole universe, with its 12 raw momentum/drawdown features and
       forward outcomes, chronologically ordered.
    2. WALK-FORWARD fit/refit: initial training window = first 40% of the
       date range (paper's own default allocation). Refit every
       RETRAIN_EVERY_DAYS (~2 calendar years) sliding the training window
       forward — a FIXED cadence, not the paper's cost-convergence
       autonomous trigger (that trigger is a materially harder thing to
       implement correctly; the paper itself reports that fixed
       frequencies up to ~8 years capture most of the value of the
       autonomous version — see its Section 6.3 / Figure 26). This is
       documented as a simplification, not silently substituted.
    3. Predictions on each out-of-sample block use ONLY the model fitted
       on data strictly before that block (no look-ahead) — mirrors the
       paper's train/test split discipline exactly.
    4. Two kinds of evaluation, reported side by side:
       a) Academic (paper's own framing): precision/recall/F-score of the
          classifier's own label (forward annualized profitability >=5%
          over H business days).
       b) Trading-outcome (this codebase's standing convention): among
          out-of-sample rows, bucket by predicted class (1 vs 0), compute
          WR/AvgR/PF using the REAL cost-adjusted net_r/is_win outcome
          (FORWARD_BARS=20, same convention as items 1-5), plus a
          permutation test of the "predicted positive" bucket vs pool.

HONEST SCOPE NOTE: primary model trains on H=20 (this codebase's own
holding horizon, matching every other item's FORWARD_BARS), not the
paper's own H=3 (found optimal for SPX but explicitly flagged by the
authors as unlikely to transfer to single stocks). An H=3 diagnostic run
is also computed for academic comparability, but its "trading outcome" is
not meaningful in this codebase's context (nothing in the live gate chain
re-evaluates a position every 3 days) so it's reported as classifier
error metrics only, not bucketed against net_r_20.

Usage:
    python validation/ml_momentum_validate.py --generate --fresh
    python validation/ml_momentum_validate.py --evaluate
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
from agents.ml_momentum_agent import (
    compute_raw_features, annualized_forward_profitability, MLMomentumModel,
    N_BASE_FEATURES, DELTA_ANNUAL,
)
from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
from validation.backtest import ROUND_TRIP_COST_PCT, _print_cost_model

log = logging.getLogger(__name__)

_LOG_DIR = BASE_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_log_file = _LOG_DIR / f"ml_momentum_validate_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(_log_file, encoding="utf-8")]
)
log.info(f"Log file: {_log_file}")

FORWARD_BARS = 20      # this codebase's own trading horizon (also used as H for the primary label)
H_DIAGNOSTIC = 3       # paper's own SPX-optimal H, academic comparison only
WIN_THRESHOLD = 0.05
STEP = 5
DEFAULT_TIER = "MID"
N_PERM = 10_000
RANDOM_SEED = 48
TRAIN_FRACTION = 0.40         # paper's own default initial-training allocation
RETRAIN_EVERY_DAYS = 500      # ~2 calendar years; fixed cadence, see module docstring

REGIME_PENALTIES = {"A": 0, "B": 0, "C": -5, "D": -12, "E": -20}

_PROGRESS_DDL = """
CREATE TABLE IF NOT EXISTS ml_momentum_validate_progress (
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
    conn.execute("DELETE FROM ml_momentum_validate_progress")
    conn.commit()
    conn.close()


def _load_completed_tickers() -> set:
    conn = get_connection()
    rows = conn.execute("SELECT ticker FROM ml_momentum_validate_progress").fetchall()
    conn.close()
    return {r["ticker"] for r in rows}


def _save_ticker_progress(ticker: str, res: dict) -> None:
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO ml_momentum_validate_progress
        (ticker, signals, results_json, completed_at)
        VALUES (?,?,?,?)
    """, (ticker, res["signals"], json.dumps(res["results"]), datetime.today().isoformat()))
    conn.commit()
    conn.close()


def _load_all_progress() -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT ticker, signals, results_json FROM ml_momentum_validate_progress"
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


def replay_ticker(ticker: str, tier: str, nifty_df: pd.DataFrame,
                   universe_ranks_by_date: dict, regime_map: dict) -> dict:
    df = _load_price_history_deep(ticker)
    if len(df) < 400:  # need enough history for the 360-day momentum feature + buffer
        return {"ticker": ticker, "signals": 0, "results": []}

    close = df["Close"].squeeze().to_numpy(dtype=float)
    cfg = UNIVERSE_CONFIG.get(tier, UNIVERSE_CONFIG[DEFAULT_TIER])
    results = []

    for i in range(360, len(df) - FORWARD_BARS, STEP):
        window = df.iloc[:i].copy()
        window.index = range(len(window))
        date_str = df.index[i].strftime("%Y-%m-%d")

        raw_feats = compute_raw_features(close, i)
        if raw_feats is None:
            continue

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

        p_h20 = annualized_forward_profitability(close, i, FORWARD_BARS)
        p_h3 = annualized_forward_profitability(close, i, H_DIAGNOSTIC)
        if p_h20 is None:
            continue

        regime = _regime_for_date(regime_map, date_str)
        penalty = int(REGIME_PENALTIES.get(regime, -5) * cfg["regime_penalty_mult"])
        out = _forward_test_net(df, i, entry)
        if out is None:
            continue

        results.append({
            "date": date_str, "pattern": pa.pattern, "regime": regime,
            "penalty_applied": penalty,
            "features": raw_feats.tolist(),
            "y_h20": int(p_h20 >= DELTA_ANNUAL),
            "y_h3": int(p_h3 >= DELTA_ANNUAL) if p_h3 is not None else None,
            "net_r": out["net_r"], "is_win": out["is_win"],
        })

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


def generate_signals(tickers: list = None, fresh: bool = False) -> None:
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
    universe_ranks_by_date = _precompute_universe_ranks_by_date(all_dates, stock_data, nifty_df)

    log.info(f"Generating ML-momentum feature/label signals for {len(remaining)} stocks...")
    total_signals = 0
    for i, (ticker, name, sector, tier) in enumerate(remaining, 1):
        try:
            res = replay_ticker(ticker, tier, nifty_df, universe_ranks_by_date, regime_map)
        except Exception as e:
            log.warning(f"  {ticker} CRASHED ({type(e).__name__}: {e}) — skipping, continuing run")
            continue
        _save_ticker_progress(ticker, res)
        total_signals += res["signals"]
        if i % 25 == 0:
            log.info(f"  {i}/{len(remaining)} | {total_signals:,} signals so far")

    log.info(f"Signal generation complete: {total_signals:,} signals.")


# ---------------------------------------------------------------------------
# Walk-forward evaluation
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


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f_score": round(f1, 3)}


def run_walk_forward() -> dict:
    all_progress = _load_all_progress()
    rows = []
    for res in all_progress:
        for o in res["results"]:
            if o.get("features") is None:
                continue
            rows.append(o)
    if len(rows) < 200:
        raise RuntimeError(f"Only {len(rows)} usable signals — too few for a walk-forward split.")

    rows.sort(key=lambda o: o["date"])
    dates = [o["date"] for o in rows]
    unique_dates = sorted(set(dates))
    n_dates = len(unique_dates)
    train_cutoff_date = unique_dates[int(n_dates * TRAIN_FRACTION)]
    log.info(f"{len(rows)} total usable signals, {n_dates} unique dates, "
             f"initial train cutoff at {train_cutoff_date} ({TRAIN_FRACTION:.0%} of range)")

    X_all = np.array([o["features"] for o in rows])
    y20_all = np.array([o["y_h20"] for o in rows])
    date_all = np.array(dates)

    # Build retrain checkpoints: train_cutoff, then every RETRAIN_EVERY_DAYS
    # trading days after that (approximated via unique-date index steps,
    # since STEP=5 sampling means each "day" here is really a 5-trading-day tick).
    idx_cutoff = int(n_dates * TRAIN_FRACTION)
    retrain_step_ticks = max(1, RETRAIN_EVERY_DAYS // STEP)
    checkpoints = list(range(idx_cutoff, n_dates, retrain_step_ticks)) + [n_dates]
    log.info(f"Walk-forward: {len(checkpoints)-1} retrain windows "
             f"(~{RETRAIN_EVERY_DAYS} trading days each)")

    oos_predictions = []  # (row, predicted_class)
    for k in range(len(checkpoints) - 1):
        train_end_date = unique_dates[checkpoints[k] - 1]
        test_start_idx = checkpoints[k]
        test_end_idx = checkpoints[k + 1]
        test_start_date = unique_dates[test_start_idx] if test_start_idx < n_dates else None
        test_end_date = unique_dates[test_end_idx - 1] if test_end_idx <= n_dates else unique_dates[-1]
        if test_start_date is None:
            break

        train_mask = date_all <= train_end_date
        test_mask = (date_all >= test_start_date) & (date_all <= test_end_date)
        n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
        if n_train < 100 or n_test == 0:
            continue

        model = MLMomentumModel()
        model.fit(X_all[train_mask], y20_all[train_mask])
        proba = model.predict_proba(X_all[test_mask])
        pred_class = (proba >= 0.5).astype(int)

        test_rows = [r for r, m in zip(rows, test_mask) if m]
        for r, p, pc in zip(test_rows, proba, pred_class):
            oos_predictions.append({**r, "proba": float(p), "pred_class": int(pc)})

        log.info(f"  window {k+1}/{len(checkpoints)-1}: train_end={train_end_date} "
                 f"(n={n_train}) -> test {test_start_date}..{test_end_date} (n={n_test})")

    log.info(f"Walk-forward complete: {len(oos_predictions)} out-of-sample predictions")

    y_true = np.array([o["y_h20"] for o in oos_predictions])
    y_pred = np.array([o["pred_class"] for o in oos_predictions])
    report = {
        "generated_at": datetime.today().isoformat(),
        "total_signals": len(rows),
        "oos_predictions": len(oos_predictions),
        "classifier_metrics_h20": _precision_recall_f1(y_true, y_pred),
    }

    # H=3 academic-comparison metrics (paper's own optimal H for SPX) — no
    # trading-outcome bucketing, see module docstring for why.
    h3_rows = [o for o in oos_predictions if o.get("y_h3") is not None]
    if len(h3_rows) >= 30:
        y3_true = np.array([o["y_h3"] for o in h3_rows])
        y3_pred = np.array([o["pred_class"] for o in h3_rows])  # same model/prediction, different label
        report["classifier_metrics_h3_diagnostic"] = _precision_recall_f1(y3_true, y3_pred)
        report["classifier_metrics_h3_diagnostic"]["note"] = (
            "predicted class is still from the H20-trained model; this measures how well an "
            "H20-trained signal happens to align with the paper's own H3 label, not a separately "
            "trained H3 classifier"
        )

    # Trading-outcome bucketing (this codebase's own standard)
    pool_r = np.array([o["net_r"] for o in oos_predictions])
    report["pool"] = _bucket_stats(oos_predictions)
    predicted_positive = [o for o in oos_predictions if o["pred_class"] == 1]
    predicted_negative = [o for o in oos_predictions if o["pred_class"] == 0]
    pos_stats = _bucket_stats(predicted_positive)
    neg_stats = _bucket_stats(predicted_negative)
    rng = np.random.default_rng(RANDOM_SEED)
    if len(predicted_positive) >= 10:
        pos_stats["p_value_vs_pool"] = round(
            _permutation_test(np.array([o["net_r"] for o in predicted_positive]), pool_r, N_PERM, rng), 4)
    if len(predicted_negative) >= 10:
        neg_stats["p_value_vs_pool"] = round(
            _permutation_test(np.array([o["net_r"] for o in predicted_negative]), pool_r, N_PERM, rng), 4)
    report["predicted_positive"] = pos_stats
    report["predicted_negative"] = neg_stats

    pos_sorted = sorted(predicted_positive, key=lambda o: o["date"])
    mid = len(pos_sorted) // 2
    report["predicted_positive_first_half"] = _bucket_stats(pos_sorted[:mid])
    report["predicted_positive_second_half"] = _bucket_stats(pos_sorted[mid:])

    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ml_momentum_validate_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_json TEXT,
            generated_at TEXT
        )
    """)
    conn.execute(
        "INSERT INTO ml_momentum_validate_results (report_json, generated_at) VALUES (?,?)",
        (json.dumps(report), report["generated_at"])
    )
    conn.commit()
    conn.close()
    log.info("Analysis stored in ml_momentum_validate_results.")
    return report


def print_stats() -> None:
    conn = get_connection()
    row = conn.execute(
        "SELECT report_json FROM ml_momentum_validate_results ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        print("No results yet — run --evaluate first.")
        return
    report = json.loads(row["report_json"])

    print("\n" + "=" * 95)
    print("  FACTOR-LIBRARY BACKLOG — ML momentum classifier (Beaudan & He 2019)")
    print("=" * 95)
    print(f"  Total usable signals: {report.get('total_signals', 0)} "
          f"| Out-of-sample predictions: {report.get('oos_predictions', 0)}")
    cm = report.get("classifier_metrics_h20", {})
    print(f"  Classifier metrics (H=20, this codebase's own horizon): "
          f"precision={cm.get('precision')} recall={cm.get('recall')} F={cm.get('f_score')}")
    cm3 = report.get("classifier_metrics_h3_diagnostic")
    if cm3:
        print(f"  Classifier metrics (H=3, academic diagnostic only): "
              f"precision={cm3.get('precision')} recall={cm3.get('recall')} F={cm3.get('f_score')}")
    print("  " + "-" * 91)
    for key in ("pool", "predicted_positive", "predicted_negative",
                "predicted_positive_first_half", "predicted_positive_second_half"):
        stats = report.get(key)
        if not stats:
            continue
        n = stats.get("n", 0)
        wr = stats.get("win_rate", 0) * 100
        ar = stats.get("avg_r", 0)
        pf = stats.get("profit_factor", 0)
        pv = stats.get("p_value_vs_pool")
        pv_str = f" p={pv}" if pv is not None else ""
        print(f"  {key:<32} N={n:>5}  WR={wr:>5.1f}%  AvgR={ar:>7.2f}  PF={pf:>5.2f}{pv_str}")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", nargs="?", default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if args.stats:
        print_stats()
    elif args.evaluate:
        run_walk_forward()
        print_stats()
    elif args.generate:
        tickers = [args.ticker.upper() if args.ticker.endswith(".NS") else args.ticker.upper() + ".NS"] \
            if args.ticker else None
        generate_signals(tickers=tickers, fresh=args.fresh)
    else:
        print("Usage:\n"
              "  python validation/ml_momentum_validate.py --generate --fresh\n"
              "  python validation/ml_momentum_validate.py --evaluate\n")
