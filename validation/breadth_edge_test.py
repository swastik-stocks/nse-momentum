"""
breadth_edge_test.py — P3-07 Stage 2

Tags every existing gate-cleared trade (from pipeline_replay_deep_progress,
the 6,106-trade Aug-05 confirmation run) with point-in-time sector_score,
industry_score, and composite_score computed from price_history_deep as of
that trade's own signal date. Formula locked 05 Aug 2026:
  - sector_score / industry_score: leave-one-out mean of the group's 5
    price flags (above_sma20/50/100, above_rsi50, above_rs55), excluding
    the scored stock itself -- otherwise a stock's own momentum leaks into
    its own "independent" confirmation signal.
  - industry_score falls back to None if fewer than MIN_INDUSTRY_REMAINING
    (5) other stocks remain in that industry after exclusion -- confirmed
    via verify_point_in_time_breadth_loo.py that only 25/120 industries
    clear a >=6-member bar; 44% of stocks fall back to sector-only, an
    accepted, deliberate coverage cost for genuine independence.
  - composite_score = 0.5*sector_score + 0.5*industry_score, or just
    sector_score if industry_score is None. 50/50 locked after confirming
    100% ticker_industry_map coverage (see verification log).

Point-in-time NIFTY 500 universe aware (via load_deep_history's
get_point_in_time_universe, already internally cached per pipeline_
replay_deep.py's own v6.1 perf fix) -- a stock not yet in the index on a
given historical date is excluded from that date's sector/industry
aggregates, same rigor as the rest of this replay pipeline.

For a point-in-time-universe ticker not present in the CURRENT
nse_universe.py / official sector list (delisted/renamed since), sector
is tagged "Unknown" rather than silently dropped -- mirrors the same
documented approximation pipeline_replay_deep.py already uses for tier
defaults on such tickers.

PERFORMANCE: each ticker's full price_history_deep series is loaded ONCE
into memory (dates + closes as numpy arrays) at startup, then sliced
per-date via np.searchsorted. This deliberately avoids the exact
per-date-per-ticker DB-query-storm pattern pipeline_replay_deep.py's own
v6.1 changelog already identified and fixed for RS-rank precomputation
("previously ran ~250,000 times per full replay... the dominant cost").

Checkpointed per trade-date to breadth_by_date_cache -- same "resume
after kill" pattern as pipeline_replay_deep.py's per-ticker checkpoint,
since the per-date computation (even in-memory) across ~500 tickers x
potentially thousands of unique dates is still real work worth not
losing to an interrupted run.

Output: breadth_tagged_trades table (one row per original trade, plus
the 3 scores) -- input for Stage 3's formal Monte Carlo tertile
significance test (deliberately NOT built here -- see module docstring
in the P3-07 design notes for why: no point building full permutation
machinery before seeing whether the tertile split shows any shape worth
testing rigorously). This script prints a descriptive tertile summary
(N, WinRate, AvgR per bucket) as a first look only.

Usage:
    python validation/breadth_edge_test.py            # resumes from checkpoint
    python validation/breadth_edge_test.py --fresh    # ignore checkpoint, start over
"""

import sys
import json
import argparse
import logging
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.schema import get_connection
from sector_breadth import build_ticker_sector_mapping, _compute_stock_flags
from industry_breadth import load_ticker_industry_map
from load_deep_history import get_point_in_time_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

MIN_INDUSTRY_REMAINING = 5
FLAG_KEYS = ("above_sma20", "above_sma50", "above_sma100", "above_rsi50", "above_rs55")
UNKNOWN_SECTOR = "Unknown"

_CACHE_DDL = """
CREATE TABLE IF NOT EXISTS breadth_by_date_cache (
    date          TEXT PRIMARY KEY,
    payload_json  TEXT,
    computed_at   TEXT
)
"""

_TAGGED_DDL = """
CREATE TABLE IF NOT EXISTS breadth_tagged_trades (
    ticker           TEXT,
    pattern          TEXT,
    date             TEXT,
    is_win           INTEGER,
    net_r            REAL,
    sector_score     REAL,
    industry_score   REAL,
    composite_score  REAL
)
"""


def _ensure_tables():
    conn = get_connection()
    conn.execute(_CACHE_DDL)
    conn.execute(_TAGGED_DDL)
    conn.commit()
    conn.close()


def _load_all_trades() -> list:
    """Flatten pipeline_replay_deep_progress (per-ticker, per-pattern
    outcome lists) into one row per individual trade."""
    conn = get_connection()
    rows = conn.execute("SELECT ticker, results_json FROM pipeline_replay_deep_progress").fetchall()
    conn.close()
    trades = []
    for ticker, results_json in rows:
        results = json.loads(results_json)
        for pattern, outcomes in results.items():
            for o in outcomes:
                trades.append({
                    "ticker": ticker,
                    "pattern": pattern,
                    "date": o.get("date"),
                    "is_win": bool(o["is_win"]),
                    "net_r": float(o["net_r"]),
                })
    return trades


def _preload_all_price_series(tickers: list) -> dict:
    """{ticker: (dates_array, closes_array)}, both sorted ascending.
    One query per ticker, ONCE -- see module docstring perf note."""
    conn = get_connection()
    series = {}
    for i, ticker in enumerate(tickers, 1):
        rows = conn.execute(
            "SELECT date, close FROM price_history_deep WHERE ticker = ? ORDER BY date",
            (ticker,),
        ).fetchall()
        if rows:
            dates = np.array([r[0] for r in rows])
            closes = np.array([r[1] for r in rows], dtype=float)
            series[ticker] = (dates, closes)
        if i % 100 == 0:
            log.info(f"  Preloaded {i}/{len(tickers)} tickers' price series...")
    conn.close()
    log.info(f"  Preloaded {len(series)}/{len(tickers)} tickers with usable price series")
    return series


def _flags_as_of(series: dict, ticker: str, date_str: str):
    """In-memory slice + flag computation -- no DB call."""
    if ticker not in series:
        return None
    dates, closes = series[ticker]
    idx = int(np.searchsorted(dates, date_str, side="right"))
    if idx == 0:
        return None
    return _compute_stock_flags(closes[:idx])


def _aggregate(ticker_flags: dict, ticker_to_group: dict):
    agg_sum = defaultdict(lambda: {k: 0 for k in FLAG_KEYS})
    agg_count = defaultdict(int)
    for ticker, flags in ticker_flags.items():
        group = ticker_to_group.get(ticker, UNKNOWN_SECTOR)
        agg_count[group] += 1
        for k in FLAG_KEYS:
            if flags[k]:
                agg_sum[group][k] += 1
    return agg_sum, agg_count


def _loo_score(flags: dict, group: str, agg_sum: dict, agg_count: dict, min_remaining: int):
    if not group:
        return None
    c = agg_count.get(group, 0)
    remaining = c - 1
    if remaining < min_remaining:
        return None
    pct_sum = 0.0
    for k in FLAG_KEYS:
        own = 1 if flags[k] else 0
        loo_sum = agg_sum[group][k] - own
        pct_sum += (loo_sum / remaining * 100) if remaining > 0 else 0.0
    return round(pct_sum / 5, 1)


def build_breadth_for_date(series: dict, ticker_to_sector: dict, ticker_to_industry: dict,
                            date_str: str) -> dict:
    """{ticker: {sector_score, industry_score, composite_score}} for every
    point-in-time-universe ticker with usable history as of date_str."""
    pit_universe = set(get_point_in_time_universe(date_str))

    ticker_flags = {}
    for ticker in pit_universe:
        flags = _flags_as_of(series, ticker, date_str)
        if flags is not None:
            ticker_flags[ticker] = flags

    sector_sum, sector_count = _aggregate(ticker_flags, ticker_to_sector)
    industry_sum, industry_count = _aggregate(ticker_flags, ticker_to_industry)

    per_ticker_scores = {}
    for ticker, flags in ticker_flags.items():
        sector = ticker_to_sector.get(ticker, UNKNOWN_SECTOR)
        industry = ticker_to_industry.get(ticker)

        ss = _loo_score(flags, sector, sector_sum, sector_count, min_remaining=1)
        isc = _loo_score(flags, industry, industry_sum, industry_count,
                          min_remaining=MIN_INDUSTRY_REMAINING) if industry else None
        if ss is None:
            continue
        composite = round(0.5 * ss + 0.5 * isc, 1) if isc is not None else ss
        per_ticker_scores[ticker] = {
            "sector_score": ss, "industry_score": isc, "composite_score": composite,
        }
    return per_ticker_scores


def _load_cached_dates() -> set:
    conn = get_connection()
    rows = conn.execute("SELECT date FROM breadth_by_date_cache").fetchall()
    conn.close()
    return {r[0] for r in rows}


def _save_date_cache(date_str: str, payload: dict):
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO breadth_by_date_cache (date, payload_json, computed_at) VALUES (?,?,?)",
        (date_str, json.dumps(payload), dt.datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def _load_date_cache(date_str: str):
    conn = get_connection()
    row = conn.execute("SELECT payload_json FROM breadth_by_date_cache WHERE date = ?", (date_str,)).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def print_tertile_summary(tagged: list):
    for score_key in ("sector_score", "industry_score", "composite_score"):
        scored = [t for t in tagged if t.get(score_key) is not None]
        if len(scored) < 30:
            log.warning(f"{score_key}: only {len(scored)} trades have this score — skipping tertile summary")
            continue
        scored.sort(key=lambda t: t[score_key])
        n = len(scored)
        buckets = [
            ("BOTTOM", scored[: n // 3]),
            ("MID",    scored[n // 3: 2 * n // 3]),
            ("TOP",    scored[2 * n // 3:]),
        ]
        print(f"\n{score_key} tertiles ({n} trades with this score):")
        print(f"  {'BUCKET':<8}{'N':>7}{'WIN%':>8}{'AVG R':>9}")
        for label, bucket in buckets:
            wr = sum(1 for x in bucket if x["is_win"]) / len(bucket) * 100
            avg_r = sum(x["net_r"] for x in bucket) / len(bucket)
            print(f"  {label:<8}{len(bucket):>7}{wr:>7.1f}%{avg_r:>9.3f}")


def run(fresh: bool = False):
    _ensure_tables()
    if fresh:
        conn = get_connection()
        conn.execute("DELETE FROM breadth_by_date_cache")
        conn.commit()
        conn.close()
        log.info("--fresh: cleared breadth_by_date_cache")

    ticker_to_sector, n_official, n_fallback = build_ticker_sector_mapping()
    ticker_to_industry = load_ticker_industry_map()
    log.info(f"Sector map: {len(ticker_to_sector)} tickers | Industry map: {len(ticker_to_industry)} tickers")

    trades = _load_all_trades()
    log.info(f"Loaded {len(trades)} trades from pipeline_replay_deep_progress")

    trades_with_date = [t for t in trades if t.get("date")]
    n_missing_date = len(trades) - len(trades_with_date)
    if n_missing_date:
        log.warning(f"{n_missing_date} trades have no 'date' field (older checkpoint schema?) — excluded")

    unique_dates = sorted({t["date"] for t in trades_with_date})
    log.info(f"{len(unique_dates)} unique trade dates need breadth computed")

    log.info("Preloading full price series for all universe tickers (one-time cost)...")
    all_tickers = sorted(set(ticker_to_sector.keys()) | set(ticker_to_industry.keys()))
    series = _preload_all_price_series(all_tickers)

    cached_dates = _load_cached_dates()
    remaining_dates = [d for d in unique_dates if d not in cached_dates]
    log.info(f"{len(unique_dates) - len(remaining_dates)} dates already cached, "
             f"{len(remaining_dates)} remaining to compute")

    for i, date_str in enumerate(remaining_dates, 1):
        payload = build_breadth_for_date(series, ticker_to_sector, ticker_to_industry, date_str)
        _save_date_cache(date_str, payload)
        if i % 25 == 0 or i == len(remaining_dates):
            log.info(f"  {i}/{len(remaining_dates)} dates computed (latest: {date_str})")

    log.info("All dates cached. Tagging trades...")
    tagged = []
    n_no_score = 0
    for t in trades_with_date:
        scores = _load_date_cache(t["date"])
        s = scores.get(t["ticker"]) if scores else None
        if s is None:
            n_no_score += 1
            continue
        tagged.append({**t, **s})
    log.info(f"Tagged {len(tagged)}/{len(trades_with_date)} trades with breadth scores "
             f"({n_no_score} had no score — ticker not in point-in-time universe or "
             f"insufficient history on that date)")

    print_tertile_summary(tagged)

    conn = get_connection()
    conn.execute("DELETE FROM breadth_tagged_trades")
    for t in tagged:
        conn.execute(
            """INSERT INTO breadth_tagged_trades
               (ticker, pattern, date, is_win, net_r, sector_score, industry_score, composite_score)
               VALUES (?,?,?,?,?,?,?,?)""",
            (t["ticker"], t["pattern"], t["date"], 1 if t["is_win"] else 0, t["net_r"],
             t.get("sector_score"), t.get("industry_score"), t.get("composite_score")),
        )
    conn.commit()
    conn.close()
    log.info(f"Persisted {len(tagged)} tagged trades to breadth_tagged_trades table (Stage 3 input)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="Ignore breadth_by_date_cache checkpoint, recompute all dates")
    args = ap.parse_args()
    run(fresh=args.fresh)
