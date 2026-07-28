"""
NSE Momentum v6 — Out-of-Sample Split-Period Significance Test
Follow-up to validation/monte_carlo_significance.py. That test asks "is
this pattern's pooled 10-year edge distinguishable from zero?" — this
test asks the harder, more important question: "does the edge hold up if
tested on two independent, non-overlapping time periods, or does the
pooled result actually come from one unusually good/bad stretch?"

WHY THIS MATTERS
    A pattern can pass the pooled bootstrap test with N=1000+ and still
    be an artifact of a single regime (e.g. the 2020-21 recovery rally)
    that happened to dominate the sample. Splitting the decade in half
    and requiring the edge to show up in BOTH halves independently is a
    much stronger bar — closer to genuine walk-forward validation than a
    single in-sample bootstrap can ever be.

WHAT THIS DOES
    Reads per-trade outcomes (now including signal date — schema v2) from
    pipeline_replay_deep_progress. Splits each pattern's trades into two
    periods at --split-date (default 2021-01-01, roughly the midpoint of
    the 2016-2026 point-in-time-universe coverage window). Runs the same
    bootstrap significance test independently on each half, then reports
    whether the pattern's edge (or lack of one) is CONSISTENT across both
    periods or only shows up in one.

REQUIRES
    A checkpoint built with the current pipeline_replay_deep.py (schema
    v2, which stores "date" per trade). If your checkpoint predates this,
    re-run: python validation/pipeline_replay_deep.py --all-patterns --fresh

USAGE
    python validation/split_period_significance.py
    python validation/split_period_significance.py --split-date 2020-06-01
    python validation/split_period_significance.py --resamples 20000 --min-n 15
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.schema import get_connection
from validation.monte_carlo_significance import bootstrap_test, permutation_test, CONFIDENCE

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

DEFAULT_RESAMPLES = 10_000
DEFAULT_MIN_N = 15          # lower than the pooled test's 20 — each half has fewer trades by construction
DEFAULT_SPLIT_DATE = "2021-01-01"

_SPLIT_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS split_period_significance (
    pattern         TEXT PRIMARY KEY,
    split_date      TEXT,
    n1              INTEGER,
    avg_r1          REAL,
    verdict1        TEXT,
    n2              INTEGER,
    avg_r2          REAL,
    verdict2        TEXT,
    consistent      TEXT,
    last_updated    TEXT
)
"""


def _ensure_table():
    conn = get_connection()
    conn.execute(_SPLIT_TABLE_DDL)
    conn.commit()
    conn.close()


def load_pattern_outcomes_with_dates() -> dict:
    """{pattern: [(date_str, net_r), ...]} — requires schema v2 checkpoint
    (per-trade "date" field). Rows missing "date" are skipped with a
    warning rather than silently treated as one period or the other."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT ticker, results_json FROM pipeline_replay_deep_progress"
    ).fetchall()
    conn.close()

    if not rows:
        raise RuntimeError(
            "pipeline_replay_deep_progress is empty. Run "
            "validation/pipeline_replay_deep.py --all-patterns --fresh first."
        )

    combined: dict = {}
    missing_date_count = 0
    for r in rows:
        results = json.loads(r["results_json"])
        for pattern, outcomes in results.items():
            for o in outcomes:
                if "date" not in o:
                    missing_date_count += 1
                    continue
                combined.setdefault(pattern, []).append((o["date"], o["net_r"]))

    if missing_date_count:
        raise RuntimeError(
            f"{missing_date_count} stored trade outcomes are missing a 'date' field — "
            f"this checkpoint predates schema v2. Re-run with --fresh to rebuild: "
            f"python validation/pipeline_replay_deep.py --all-patterns --fresh"
        )

    return combined


def _verdict_short(ci_low: float, ci_high: float, p_ge_0: float) -> str:
    if ci_high < 0:
        return "NEGATIVE (sig)"
    if ci_low > 0:
        return "POSITIVE (sig)"
    if p_ge_0 < 0.10:
        return "leans negative"
    if p_ge_0 > 0.90:
        return "leans positive"
    return "inconclusive"


def _consistency(v1: str, v2: str) -> str:
    sig1 = "sig" in v1
    sig2 = "sig" in v2
    pos1 = "POSITIVE" in v1 or v1 == "leans positive"
    pos2 = "POSITIVE" in v2 or v2 == "leans positive"
    neg1 = "NEGATIVE" in v1 or v1 == "leans negative"
    neg2 = "NEGATIVE" in v2 or v2 == "leans negative"

    if sig1 and sig2 and pos1 and pos2:
        return "CONSISTENT — significant positive in both periods"
    if sig1 and sig2 and neg1 and neg2:
        return "CONSISTENT — significant negative in both periods"
    if pos1 and pos2 and not (neg1 or neg2):
        return "consistent direction (positive), not both significant"
    if neg1 and neg2 and not (pos1 or pos2):
        return "consistent direction (negative), not both significant"
    if (pos1 and neg2) or (neg1 and pos2):
        return "FLIPPED — opposite direction across periods, treat pooled result with caution"
    return "mixed / at least one period inconclusive"


def run(split_date: str, resamples: int, min_n: int) -> None:
    _ensure_table()
    outcomes_by_pattern = load_pattern_outcomes_with_dates()

    log.info(f"Loaded {len(outcomes_by_pattern)} patterns. Splitting at {split_date}, "
             f"running {resamples:,} bootstrap resamples per half...")

    today = datetime.today().strftime("%Y-%m-%d")
    conn = get_connection()
    report_rows = []

    for pattern, dated_outcomes in sorted(outcomes_by_pattern.items(),
                                           key=lambda kv: -len(kv[1])):
        period1 = [r for d, r in dated_outcomes if d < split_date]
        period2 = [r for d, r in dated_outcomes if d >= split_date]

        n1, n2 = len(period1), len(period2)

        if n1 < min_n or n2 < min_n:
            verdict1 = verdict2 = "TOO THIN"
            avg1 = float(np.mean(period1)) if period1 else float("nan")
            avg2 = float(np.mean(period2)) if period2 else float("nan")
            consistency = f"TOO THIN TO SPLIT (N1={n1}, N2={n2}, need >= {min_n} each)"
        else:
            boot1 = bootstrap_test(period1, resamples, seed=42)
            boot2 = bootstrap_test(period2, resamples, seed=43)
            verdict1 = _verdict_short(boot1["ci_low"], boot1["ci_high"], boot1["p_value_mean_ge_0"])
            verdict2 = _verdict_short(boot2["ci_low"], boot2["ci_high"], boot2["p_value_mean_ge_0"])
            avg1, avg2 = boot1["observed_avg_r"], boot2["observed_avg_r"]
            consistency = _consistency(verdict1, verdict2)

        conn.execute("""
            INSERT OR REPLACE INTO split_period_significance
            (pattern, split_date, n1, avg_r1, verdict1, n2, avg_r2, verdict2, consistent, last_updated)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (pattern, split_date, n1, avg1, verdict1, n2, avg2, verdict2, consistency, today))

        report_rows.append({
            "pattern": pattern, "n1": n1, "avg1": avg1, "v1": verdict1,
            "n2": n2, "avg2": avg2, "v2": verdict2, "consistency": consistency,
        })

    conn.commit()
    conn.close()
    print_report(report_rows, split_date)


def print_report(rows: list, split_date: str) -> None:
    print("\n" + "=" * 130)
    print(f"  OUT-OF-SAMPLE SPLIT-PERIOD TEST  (before vs on/after {split_date}, {CONFIDENCE*100:.0f}% CI bootstrap each half)")
    print("=" * 130)
    print(f"  {'PATTERN':<20}{'N1':>6}{'AVG R1':>9}{'VERDICT1':>16}   |  {'N2':>6}{'AVG R2':>9}{'VERDICT2':>16}   CONSISTENCY")
    print("  " + "-" * 126)
    for r in rows:
        a1 = f"{r['avg1']:.2f}R" if r['avg1'] == r['avg1'] else "  n/a"   # NaN check
        a2 = f"{r['avg2']:.2f}R" if r['avg2'] == r['avg2'] else "  n/a"
        print(f"  {r['pattern']:<20}{r['n1']:>6}{a1:>9}{r['v1']:>16}   |  "
              f"{r['n2']:>6}{a2:>9}{r['v2']:>16}   {r['consistency']}")
    print("=" * 130)
    print("  Trust CONSISTENT results most. FLIPPED results mean the pooled 10yr number in")
    print("  monte_carlo_significance.py may be driven by one period, not a durable edge.")
    print("=" * 130 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-date", default=DEFAULT_SPLIT_DATE,
                         help=f"Date to split periods at, YYYY-MM-DD (default {DEFAULT_SPLIT_DATE})")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N,
                         help=f"Minimum signals required IN EACH HALF to test (default {DEFAULT_MIN_N})")
    args = parser.parse_args()
    run(split_date=args.split_date, resamples=args.resamples, min_n=args.min_n)
