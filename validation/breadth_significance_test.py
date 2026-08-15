"""
breadth_significance_test.py — P3-07 Stage 3

Formal Monte Carlo significance test for the P3-07 breadth scores, same
statistical parameters as validation/monte_carlo_significance.py (the
script that decided VCP's fate earlier today): 10,000 bootstrap
resamples, 10,000 permutation draws, seed=42. Adapted here to a
two-sample TOP-tertile-vs-BOTTOM-tertile comparison rather than
pattern-vs-pooled-baseline, since that's the natural test for a
continuous score split into buckets.

Reads breadth_tagged_trades (Stage 2 output: 6,011 trades, each tagged
with sector_score, industry_score, composite_score as of its own
point-in-time signal date). For each of the 3 scores independently:
  1. Split into tertiles (bottom/mid/top) by that score.
  2. Bootstrap 95% CI on AvgR for each bucket independently.
  3. One-tailed permutation test: pools TOP+BOTTOM trades, reshuffles
     into same-sized groups 10,000 times, and asks how often a random
     split produces a gap >= the ACTUAL observed TOP-minus-BOTTOM AvgR
     gap. One-tailed because the hypothesis is directional (stronger
     breadth -> better returns), not merely "the two differ" -- matches
     the directional framing monte_carlo_significance.py already uses
     ("SIGNIFICANT: positive expectancy...").

Buckets with fewer than MIN_BUCKET_N (20) trades are marked TOO THIN TO
TEST rather than given a misleading verdict -- same guard
monte_carlo_significance.py used for Symmetrical Triangle (N=13).

Usage:
    python validation/breadth_significance_test.py
"""

import sys
import logging
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.schema import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

N_BOOT = 10000
N_PERM = 10000
SEED = 42
MIN_BUCKET_N = 20


def load_tagged_trades(score_col: str) -> list:
    conn = get_connection()
    rows = conn.execute(
        f"SELECT net_r, is_win FROM breadth_tagged_trades "
        f"WHERE {score_col} IS NOT NULL ORDER BY {score_col} ASC"
    ).fetchall()
    conn.close()
    return [{"net_r": r[0], "is_win": bool(r[1])} for r in rows]


def bootstrap_ci(trades: list, rng: np.random.Generator, n_boot: int = N_BOOT):
    r = np.array([t["net_r"] for t in trades])
    n = len(r)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(r, size=n, replace=True)
        means[i] = sample.mean()
    return float(np.mean(r)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def profit_factor(trades: list) -> float:
    wins = sum(t["net_r"] for t in trades if t["is_win"])
    losses = abs(sum(t["net_r"] for t in trades if not t["is_win"]))
    if losses > 0:
        return wins / losses
    return float("inf") if wins > 0 else 0.0


def permutation_test_diff(top: list, bottom: list, rng: np.random.Generator, n_perm: int = N_PERM):
    """One-tailed: fraction of random reshuffles of the pooled TOP+BOTTOM
    trades whose group-A-minus-group-B gap >= the ACTUAL observed
    TOP-minus-BOTTOM gap. Low p = the real split is not just luck."""
    r_top = np.array([t["net_r"] for t in top])
    r_bottom = np.array([t["net_r"] for t in bottom])
    observed_diff = float(r_top.mean() - r_bottom.mean())

    pooled = np.concatenate([r_top, r_bottom])
    n_top = len(r_top)

    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        diff = perm[:n_top].mean() - perm[n_top:].mean()
        if diff >= observed_diff:
            count_ge += 1
    p_value = count_ge / n_perm
    return observed_diff, p_value


def run_for_score(score_col: str):
    trades = load_tagged_trades(score_col)
    n = len(trades)
    print(f"\n{'='*90}")
    print(f"  {score_col}  (N={n} tagged trades)")
    print(f"{'='*90}")
    if n < MIN_BUCKET_N * 3:
        print(f"  TOO THIN TO TEST (N={n} < minimum {MIN_BUCKET_N * 3} for 3 buckets)")
        return

    bottom = trades[: n // 3]
    mid = trades[n // 3: 2 * n // 3]
    top = trades[2 * n // 3:]

    rng = np.random.default_rng(SEED)

    print(f"  {'BUCKET':<8}{'N':>7}{'WIN%':>8}{'AVG R':>9}{'95% CI (AVG R)':>20}{'PF':>7}")
    for label, bucket in (("BOTTOM", bottom), ("MID", mid), ("TOP", top)):
        if len(bucket) < MIN_BUCKET_N:
            print(f"  {label:<8} TOO THIN TO TEST (N={len(bucket)} < {MIN_BUCKET_N})")
            continue
        mean_r, ci_lo, ci_hi = bootstrap_ci(bucket, rng)
        wr = sum(1 for t in bucket if t["is_win"]) / len(bucket) * 100
        pf = profit_factor(bucket)
        ci_str = f"[{ci_lo:>6.3f}, {ci_hi:>6.3f}]"
        print(f"  {label:<8}{len(bucket):>7}{wr:>7.1f}%{mean_r:>9.3f}{ci_str:>20}{pf:>7.2f}")

    if len(top) >= MIN_BUCKET_N and len(bottom) >= MIN_BUCKET_N:
        observed_diff, p_value = permutation_test_diff(top, bottom, rng)
        sig = "SIGNIFICANT (p<0.05)" if p_value < 0.05 else "NOT SIGNIFICANT (p>=0.05)"
        strong = " -- STRONG (p<0.01)" if p_value < 0.01 else ""
        print(f"\n  TOP vs BOTTOM: observed AvgR gap = {observed_diff:+.3f}R   "
              f"one-tailed permutation p = {p_value:.4f}   -> {sig}{strong}")


def run_matched_sample_comparison():
    """
    Fairness check before freezing the industry-first-sector-fallback
    design: compares sector_score vs industry_score on the EXACT SAME
    trade subset (the 2,632 trades where industry_score is available),
    not sector's full 6,011-trade population vs industry's 2,632-trade
    subset. Isolates whether industry's edge is genuinely about
    industry-specific information, or just an artifact of which trades
    happen to have industry coverage (e.g. larger/more liquid/better-
    classified stocks that might simply perform better regardless of
    which score is used).
    """
    conn = get_connection()
    rows = conn.execute(
        "SELECT sector_score, industry_score, net_r, is_win FROM breadth_tagged_trades "
        "WHERE industry_score IS NOT NULL"
    ).fetchall()
    conn.close()

    trades = [{"sector_score": r[0], "industry_score": r[1], "net_r": r[2], "is_win": bool(r[3])}
              for r in rows]
    n = len(trades)

    print(f"\n{'#' * 90}")
    print(f"  MATCHED-SAMPLE COMPARISON — sector_score vs industry_score, "
          f"SAME {n} trades for both")
    print(f"{'#' * 90}")

    results = {}
    for score_key in ("sector_score", "industry_score"):
        sorted_trades = sorted(trades, key=lambda t: t[score_key])
        bottom = sorted_trades[: n // 3]
        top = sorted_trades[2 * n // 3:]

        rng = np.random.default_rng(SEED)
        top_mean, top_lo, top_hi = bootstrap_ci(top, rng)
        bottom_mean, bottom_lo, bottom_hi = bootstrap_ci(bottom, rng)
        top_pf = profit_factor(top)
        observed_diff, p_value = permutation_test_diff(top, bottom, rng)

        results[score_key] = {
            "top_mean": top_mean, "top_ci": (top_lo, top_hi), "top_pf": top_pf,
            "bottom_mean": bottom_mean, "gap": observed_diff, "p": p_value,
        }

        print(f"\n  {score_key} (matched sample, N={n}):")
        print(f"    BOTTOM  N={len(bottom):>5}  AvgR={bottom_mean:>7.3f}")
        print(f"    TOP     N={len(top):>5}  AvgR={top_mean:>7.3f}  "
              f"95% CI=[{top_lo:.3f}, {top_hi:.3f}]  PF={top_pf:.2f}")
        sig = "SIGNIFICANT (p<0.05)" if p_value < 0.05 else "NOT SIGNIFICANT"
        strong = " -- STRONG (p<0.01)" if p_value < 0.01 else ""
        print(f"    TOP vs BOTTOM gap = {observed_diff:+.3f}R   p = {p_value:.4f}   -> {sig}{strong}")

    print(f"\n  HEAD-TO-HEAD (matched sample, same {n} trades both times):")
    print(f"  {'':<16}{'TOP AvgR':>12}{'TOP PF':>10}{'TOP-BOTTOM gap':>18}{'p-value':>10}")
    for score_key in ("sector_score", "industry_score"):
        r = results[score_key]
        print(f"  {score_key:<16}{r['top_mean']:>12.3f}{r['top_pf']:>10.2f}"
              f"{r['gap']:>18.3f}{r['p']:>10.4f}")

    winner = max(results, key=lambda k: results[k]["top_mean"])
    print(f"\n  On this matched sample, {winner} has the higher TOP-tertile AvgR.")


if __name__ == "__main__":
    log.info(f"Monte Carlo breadth significance test — {N_BOOT:,} bootstrap resamples, "
              f"{N_PERM:,} permutation draws, seed={SEED}")
    for score_col in ("sector_score", "industry_score", "composite_score"):
        run_for_score(score_col)
    run_matched_sample_comparison()
    print()
