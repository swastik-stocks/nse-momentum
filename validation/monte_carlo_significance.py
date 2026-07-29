"""
validation/monte_carlo_significance.py — NSE Momentum v6

Monte Carlo significance testing on pattern trade edges from
pipeline_replay_deep.py output.

Answers: "Is this pattern's edge (avg R, PF) distinguishable from
what pure chance would produce, given the same trade count?"

Two tests are run per pattern:

1. BOOTSTRAP CONFIDENCE INTERVALS
   Resample the pattern's own trades (with replacement) N_BOOT times,
   recompute avg R and PF each time -> gives a 95% CI on both stats.
   If the CI for avg R excludes 0, the edge is unlikely to be a fluke
   of this particular sample.

2. PERMUTATION / LABEL-SHUFFLE TEST (the stronger test)
   Pool ALL trades across ALL patterns (the full gate-cleared universe).
   Repeatedly draw a random sample of the same size as the pattern's
   N, from the pooled pool, without regard to pattern label, and
   compute avg R. This builds a null distribution: "what avg R would
   a random N-sized subset of all gate-cleared trades produce, if
   pattern identity didn't matter?"
   p-value = fraction of null draws with avg R >= observed avg R.
   This is the more honest test because it doesn't ask "is this
   pattern's edge different from zero" (which is easy for skewed R
   distributions) — it asks "is this pattern's edge different from
   what you'd get by picking trades at random from the same gate
   chain," which directly tests whether the PATTERN LABEL carries
   information.

Usage:
    python validation\\monte_carlo_significance.py --patterns "Cup & Handle,Swing High Breakout,VCP,Falling Wedge,Bull Flag"
    python validation\\monte_carlo_significance.py --all-patterns

Requires: numpy, pandas (already in your venv per SKILL stack)
"""

import argparse
import json
import sqlite3
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG — adjust to match your environment
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "momentum_v4.db"  # confirmed 2026-07-27
N_BOOT = 10_000          # bootstrap resamples per pattern
N_PERM = 10_000          # permutation draws per pattern
CONFIDENCE = 0.95
RANDOM_SEED = 42
ROUND_TRIP_COST_PCT = 0.363  # from your v6 cost model, informational only


# ---------------------------------------------------------------------------
# DATA LOADING — ADAPT THIS FUNCTION to your actual schema
# ---------------------------------------------------------------------------
def load_trades(db_path: Path, min_completed_at: str = None) -> pd.DataFrame:
    """
    Parses per-trade results out of pipeline_replay_deep_progress.results_json.

    Schema (confirmed 2026-07-27 against momentum_v4.db):
        pipeline_replay_deep_progress:
            ticker TEXT, signals INTEGER, results_json TEXT, completed_at TEXT
        results_json is a dict keyed by pattern name, each value a list of
        trade dicts: {is_win, net_r, regime, penalty_applied, date}

    net_r is already the cost-adjusted realized R (the "net" in net_r
    reflects the round-trip transaction cost model, confirmed by
    comparing against the printed pipeline_replay_deep.py summary table
    for the 2026-07-27 --all-patterns --fresh run). No further cost
    adjustment is applied here.

    min_completed_at: if given (ISO string), only rows completed at or
    after this timestamp are included. Use this to exclude any stale
    rows left over from a prior (e.g. pre-brokerage-fix) run in case
    the checkpoint wasn't fully cleared by --fresh. Default None = take
    everything currently in the table.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"{db_path} not found. Edit DB_PATH at the top of this script."
        )

    conn = sqlite3.connect(str(db_path))
    try:
        query = "SELECT ticker, signals, results_json, completed_at FROM pipeline_replay_deep_progress WHERE signals > 0"
        if min_completed_at:
            query += f" AND completed_at >= '{min_completed_at}'"
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()

    records = []
    for ticker, signals, results_json, completed_at in rows:
        try:
            parsed = json.loads(results_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, dict):
            continue
        for pattern, trades in parsed.items():
            if not isinstance(trades, list):
                continue
            for trade in trades:
                if "net_r" not in trade:
                    continue
                records.append({
                    "ticker": ticker,
                    "pattern": pattern,
                    "r_multiple": trade["net_r"],
                    "is_win": trade.get("is_win"),
                    "regime": trade.get("regime"),
                    "penalty_applied": trade.get("penalty_applied", 0),
                    "date": trade.get("date"),
                    "completed_at": completed_at,
                })

    df = pd.DataFrame.from_records(records)
    return df


def load_trades_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Fallback loader if you export per-trade results to CSV instead
    (e.g. if pipeline_replay_deep.py has a --export-trades flag, or
    you add one). Expects columns: pattern, r_multiple.
    """
    df = pd.read_csv(csv_path)
    required = {"pattern", "r_multiple"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# STATS
# ---------------------------------------------------------------------------
def profit_factor(r_values: np.ndarray) -> float:
    gains = r_values[r_values > 0].sum()
    losses = -r_values[r_values < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def bootstrap_ci(r_values: np.ndarray, n_boot: int, confidence: float, rng: np.random.Generator):
    n = len(r_values)
    boot_avg_r = np.empty(n_boot)
    boot_pf = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(r_values, size=n, replace=True)
        boot_avg_r[i] = sample.mean()
        boot_pf[i] = profit_factor(sample)

    alpha = (1 - confidence) / 2
    avg_r_ci = (np.quantile(boot_avg_r, alpha), np.quantile(boot_avg_r, 1 - alpha))
    pf_finite = boot_pf[np.isfinite(boot_pf)]
    pf_ci = (
        (np.quantile(pf_finite, alpha), np.quantile(pf_finite, 1 - alpha))
        if len(pf_finite) > 0
        else (np.nan, np.nan)
    )
    p_value_mean_ge_0 = (boot_avg_r >= 0).mean()
    return avg_r_ci, pf_ci, p_value_mean_ge_0


def bootstrap_test(r_values, n_boot: int, seed: int = 42) -> dict:
    """
    Thin dict-returning wrapper around bootstrap_ci(), added for
    validation/split_period_significance.py, which expects a single
    dict result {observed_avg_r, ci_low, ci_high, p_value_mean_ge_0}
    keyed by name rather than the tuple bootstrap_ci() returns (which
    this module's own CLI in main() uses directly). Takes a plain
    integer seed rather than a pre-built np.random.Generator, since
    split_period_significance.py calls this once per period with a
    fixed seed (42 / 43) for reproducibility across the two halves.

    Does not duplicate any bootstrap logic -- delegates straight to
    bootstrap_ci() so there's exactly one implementation of the actual
    resampling math in this file.
    """
    r_values = np.asarray(r_values, dtype=float)
    rng = np.random.default_rng(seed)
    avg_r_ci, _pf_ci, p_value_mean_ge_0 = bootstrap_ci(r_values, n_boot, CONFIDENCE, rng)
    return {
        "observed_avg_r": float(r_values.mean()),
        "ci_low": avg_r_ci[0],
        "ci_high": avg_r_ci[1],
        "p_value_mean_ge_0": p_value_mean_ge_0,
    }


def permutation_test(pattern_r: np.ndarray, pooled_r: np.ndarray, n_perm: int, rng: np.random.Generator):
    """
    Null hypothesis: this pattern's trades are indistinguishable from a
    random same-sized draw of ALL gate-cleared trades (i.e. pattern
    label carries no information about R).
    Returns (observed_avg_r, null_mean, null_std, p_value_one_sided).
    """
    n = len(pattern_r)
    observed = pattern_r.mean()

    null_avgs = np.empty(n_perm)
    for i in range(n_perm):
        draw = rng.choice(pooled_r, size=n, replace=False)
        null_avgs[i] = draw.mean()

    p_value = (null_avgs >= observed).mean()
    return observed, null_avgs.mean(), null_avgs.std(), p_value


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Monte Carlo significance test on pattern edges")
    parser.add_argument("--patterns", type=str, default=None,
                         help="Comma-separated pattern names to test. Default: the 5 candidates "
                              "(Cup & Handle, Swing High Breakout, VCP, Falling Wedge, Bull Flag)")
    parser.add_argument("--all-patterns", action="store_true",
                         help="Test every pattern present in the loaded data")
    parser.add_argument("--csv", type=str, default=None,
                         help="Load per-trade data from CSV instead of the DB")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--min-completed-at", type=str, default=None,
                         help="ISO timestamp filter, e.g. 2026-07-27T00:00:00 — excludes any "
                              "stale rows completed before this time (safety valve in case a "
                              "checkpoint wasn't fully cleared)")
    parser.add_argument("--no-db-write", action="store_true",
                         help="Skip writing results back into monte_carlo_significance table; CSV only")
    args = parser.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)

    if args.csv:
        df = load_trades_from_csv(Path(args.csv))
    else:
        df = load_trades(DB_PATH, min_completed_at=args.min_completed_at)

    if df.empty:
        print("No trade data loaded — check load_trades()/load_trades_from_csv() wiring.", file=sys.stderr)
        sys.exit(1)

    if args.all_patterns:
        target_patterns = sorted(df["pattern"].unique())
    elif args.patterns:
        target_patterns = [p.strip() for p in args.patterns.split(",")]
    else:
        target_patterns = ["Cup & Handle", "Swing High Breakout", "VCP", "Falling Wedge", "Bull Flag"]

    pooled_r = df["r_multiple"].to_numpy()

    print("=" * 95)
    print("  MONTE CARLO SIGNIFICANCE TEST — pattern edge vs. random draw from pooled gate-cleared trades")
    print(f"  Pooled universe: {len(pooled_r):,} trades across {df['pattern'].nunique()} patterns")
    print(f"  Bootstrap resamples: {args.n_boot:,} | Permutation draws: {args.n_perm:,} | Seed: {RANDOM_SEED}")
    print("=" * 95)
    header = f"  {'PATTERN':<24}{'N':>6}{'AVG R':>9}{'95% CI (avg R)':>22}{'PF':>7}{'95% CI (PF)':>16}{'p-value':>10}  VERDICT"
    print(header)
    print("-" * 95)

    results = []
    for pattern in target_patterns:
        sub = df.loc[df["pattern"] == pattern, "r_multiple"].to_numpy()
        if len(sub) == 0:
            print(f"  {pattern:<24}  -- no trades found, skipping --")
            continue

        avg_r = sub.mean()
        pf = profit_factor(sub)
        avg_r_ci, pf_ci, p_value_mean_ge_0 = bootstrap_ci(sub, args.n_boot, CONFIDENCE, rng)
        observed, null_mean, null_std, permutation_p_value = permutation_test(sub, pooled_r, args.n_perm, rng)

        if len(sub) < 20:
            verdict = f"TOO THIN TO TEST (N={len(sub)} < min 20) — do not draw conclusions"
        elif avg_r_ci[0] > 0:
            verdict = "SIGNIFICANT: positive expectancy — 95% CI entirely above zero"
        elif avg_r_ci[1] < 0:
            verdict = "SIGNIFICANT: negative expectancy — 95% CI entirely below zero"
        elif p_value_mean_ge_0 >= 0.90 and avg_r > 0:
            verdict = "LIKELY POSITIVE: CI straddles zero but most resamples stay positive"
        elif p_value_mean_ge_0 <= 0.10 and avg_r < 0:
            verdict = "LIKELY NEGATIVE: CI straddles zero but most resamples stay negative"
        else:
            verdict = "INCONCLUSIVE: CI straddles zero, resamples split — cannot distinguish from no-edge"

        ci_str = f"[{avg_r_ci[0]:.2f}, {avg_r_ci[1]:.2f}]"
        pf_ci_str = f"[{pf_ci[0]:.2f}, {pf_ci[1]:.2f}]" if np.isfinite(pf_ci[0]) else "n/a"

        print(f"  {pattern:<24}{len(sub):>6}{avg_r:>9.2f}{ci_str:>22}{pf:>7.2f}{pf_ci_str:>16}{permutation_p_value:>10.4f}  {verdict}")

        results.append({
            "pattern": pattern, "n_signals": len(sub), "observed_avg_r": avg_r,
            "ci_low": avg_r_ci[0], "ci_high": avg_r_ci[1], "pf": pf,
            "p_value_mean_ge_0": p_value_mean_ge_0,
            "permutation_p_value": permutation_p_value,
            "verdict": verdict, "resamples": args.n_boot,
        })

    print("=" * 95)
    print("  Interpretation:")
    print("  p-value = fraction of random N-sized draws from the ENTIRE pooled trade set that")
    print("  matched or beat this pattern's avg R. Low p-value = the pattern label is doing real")
    print("  work, not just riding the overall gate chain's baseline edge.")
    print("  p < 0.05 is the conventional bar; p < 0.01 is a strong result worth weighting heavily.")
    print("=" * 95)

    out_df = pd.DataFrame(results)
    out_path = Path(__file__).resolve().parent / "monte_carlo_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nCSV backup written to {out_path}")

    if not args.no_db_write:
        write_back_to_db(DB_PATH, results)


def write_back_to_db(db_path: Path, results: list):
    """
    Upserts results into the existing monte_carlo_significance table
    (same schema already present in momentum_v4.db), replacing any
    stale rows for the patterns just tested. Rows for patterns NOT
    included in this run's --patterns list are left untouched.
    """
    import datetime
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    for r in results:
        cur.execute("""
            INSERT INTO monte_carlo_significance
                (pattern, n_signals, observed_avg_r, ci_low, ci_high,
                 p_value_mean_ge_0, permutation_p_value, verdict, resamples, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pattern) DO UPDATE SET
                n_signals=excluded.n_signals,
                observed_avg_r=excluded.observed_avg_r,
                ci_low=excluded.ci_low,
                ci_high=excluded.ci_high,
                p_value_mean_ge_0=excluded.p_value_mean_ge_0,
                permutation_p_value=excluded.permutation_p_value,
                verdict=excluded.verdict,
                resamples=excluded.resamples,
                last_updated=excluded.last_updated
        """, (r["pattern"], r["n_signals"], r["observed_avg_r"], r["ci_low"], r["ci_high"],
              r["p_value_mean_ge_0"], r["permutation_p_value"], r["verdict"], r["resamples"], now))
    conn.commit()
    conn.close()
    print(f"Updated {len(results)} rows in monte_carlo_significance table (last_updated={now}).")
    print("NOTE: if 'pattern' isn't a unique/primary key on that table, the ON CONFLICT upsert")
    print("will fail with an OperationalError — in that case run with --no-db-write and manually")
    print("DELETE the stale rows before re-inserting from the CSV, or add a UNIQUE constraint.")


if __name__ == "__main__":
    main()
