"""
Regime breadth-input validation (2026-11-17)
============================================

QUESTION: the live RegimeClassifier scores the single-day A/D ratio twice
(directly, and again via a breadth_score that on the bhavcopy path is just a
lookup of the same A/D). Does replacing that second vote help or hurt?

ENGINES compared, all through the REAL RegimeClassifier code:
    LIVE_NOW   breadth_score = A/D lookup (compute_breadth_score, bhavcopy path)
    OPT2       breadth_score = None -> neutral (A/D counted once)
    OPT1       breadth_score = structural score (agents/structural_breadth.py)
    BACKFILL   market_regime_history.regime -- the regime the existing deep
               replays (and thus the score gates / -12 D penalty) were run with

EVIDENCE: the 6,106 gate-cleared signals already produced by
validation/pipeline_replay_deep.py (real gate chain, point-in-time universe,
net of costs). A regime engine is useful if the days it labels D/E
(T1 cap 0 = blocked) produce WORSE signals than the days it allows.
Reported per engine: mean net R allowed vs blocked, permutation p-value
(one-sided: allowed > blocked), the same split on two halves
(2016-2020 / 2021-2026) for stability, and regime flips per 100 sessions.

KNOWN LIMITS (stated, not hidden):
  - Daily A/D is the point-in-time NIFTY 500 universe's, not NSE-wide
    (~3,000 symbols) -- NSE-wide history isn't stored. Same proxy for every
    engine, so the comparison is fair even if levels differ from live.
  - Bank Nifty history isn't stored locally: every engine gets the
    classifier's neutral 1 pt for it. Equal across engines.
  - Regime only. MacroAgent (FII flows etc.) has no stored history.
  - price_history_deep ends 2026-02-23.

Usage:  python validation/regime_breadth_validate.py
"""

import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from agents.market_breadth_agent import compute_breadth_score
from agents.regime_classifier import RegimeClassifier
from agents.structural_breadth import structural_breadth_series

logging.basicConfig(level=logging.WARNING, format="%(message)s")
DB = BASE_DIR / "data" / "momentum_v4.db"
LIVE_PATTERNS = ("Cup & Handle", "Swing High Breakout")   # the two patterns live today
WARMUP_START = "2014-06-01"   # 252-day H/L window + 50-EMA before the first snapshot
PERMUTATIONS = 5000
BLOCKED = ("D", "E")

# The replay must not spam logs/regime_calibration.log with ~2,500 synthetic rows.
RegimeClassifier._log_calibration = lambda self: None


def load_prices(conn):
    df = pd.read_sql("SELECT ticker, date, high, low, close FROM price_history_deep WHERE date >= ?",
                     conn, params=(WARMUP_START,), parse_dates=["date"])
    wide = {c: df.pivot(index="date", columns="ticker", values=c).sort_index() for c in ("close", "high", "low")}
    return wide["close"], wide["high"], wide["low"]


def membership_mask(conn, index, columns):
    snaps = conn.execute("SELECT effective_date, symbols FROM universe_snapshots ORDER BY effective_date").fetchall()
    mask = pd.DataFrame(np.nan, index=index, columns=columns, dtype=float)
    for eff, syms in snaps:
        members = [s for s in syms.split(",") if s in mask.columns]
        row = pd.Series(0.0, index=columns); row[members] = 1.0
        mask.loc[pd.Timestamp(eff)] = row  # snapshot date may be a non-trading day
    mask = mask.sort_index().ffill().reindex(index)
    return mask.fillna(0).astype(bool), pd.Timestamp(snaps[0][0])


def legacy_breadth(adv_rate_1d):
    """Exact production path: bhavcopy-sourced A/D -> compute_breadth_score()."""
    if pd.isna(adv_rate_1d):
        return np.nan, np.nan
    adv, dec = int(round(adv_rate_1d * 1000)), int(round((1 - adv_rate_1d) * 1000))
    r = compute_breadth_score({"advances": adv, "declines": dec, "source": "BHAVCOPY_FULL"})
    return r["breadth_score"], r["ad_ratio"]


def classify(nifty, vix, ad_ratio, breadth, above50):
    return RegimeClassifier(nifty50_data=nifty, banknifty_data=None, vix=vix, ad_ratio=ad_ratio,
                            breadth_score=breadth, above_50_ema_pct=above50).classify()["regime"]


def perm_p(allowed, blocked, rng):
    """One-sided: P(mean(allowed) - mean(blocked) >= observed) under random labels."""
    if len(allowed) < 20 or len(blocked) < 20:
        return np.nan
    obs = allowed.mean() - blocked.mean()
    pool = np.concatenate([allowed, blocked]); n = len(allowed); hits = 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(pool)
        hits += (pool[:n].mean() - pool[n:].mean()) >= obs
    return (hits + 1) / (PERMUTATIONS + 1)


def report(name, sig, regimes, rng):
    lab = sig["date"].map(regimes)
    s = sig.assign(regime=lab).dropna(subset=["regime"])
    print(f"\n=== {name} ===")
    dist = regimes.value_counts(normalize=True).reindex(list("ABCDE")).fillna(0)
    flips = (regimes != regimes.shift()).iloc[1:].mean() * 100
    print("days by regime: " + "  ".join(f"{k} {v:.0%}" for k, v in dist.items()) + f" | flips per 100 sessions {flips:.1f}")
    g = s.groupby("regime")["net_r"].agg(["count", "mean"])
    g["win"] = s.groupby("regime")["is_win"].mean()
    print("signals by regime: " + " | ".join(f"{r}: n={int(row['count'])} R={row['mean']:+.2f} win={row['win']:.0%}" for r, row in g.iterrows()))
    out = {}
    for label, part in (("ALL", s), ("2016-20", s[s.date < "2021-01-01"]), ("2021-26", s[s.date >= "2021-01-01"])):
        blk = part[part.regime.isin(BLOCKED)].net_r.to_numpy(); alw = part[~part.regime.isin(BLOCKED)].net_r.to_numpy()
        p = perm_p(alw.copy(), blk.copy(), rng)
        diff = (alw.mean() if len(alw) else np.nan) - (blk.mean() if len(blk) else np.nan)
        print(f"  {label:<8} allowed n={len(alw):>4} R={alw.mean() if len(alw) else np.nan:+.3f} | "
              f"blocked n={len(blk):>4} R={blk.mean() if len(blk) else np.nan:+.3f} | edge {diff:+.3f}R  p={p:.4f}")
        out[label] = (diff, p, len(blk))
    pos_blocked = s[s.regime.isin(BLOCKED) & (s.net_r > 0)].net_r.sum() / s[s.net_r > 0].net_r.sum()
    print(f"  share of all winning R that this engine blocks: {pos_blocked:.0%}")
    return out


def main():
    conn = sqlite3.connect(DB)
    print("loading prices...", flush=True)
    close, high, low = load_prices(conn)
    mask, first_snap = membership_mask(conn, close.index, close.columns)
    print("structural breadth series...", flush=True)
    sb = structural_breadth_series(close, high, low, member_mask=mask)

    mrh = pd.read_sql("SELECT date, regime, nifty_close, vix FROM market_regime_history ORDER BY date",
                      conn, parse_dates=["date"]).set_index("date")
    nifty_all = mrh[["nifty_close"]].rename(columns={"nifty_close": "Close"})

    rows = []
    for raw in json.loads("[" + ",".join(r[0] for r in conn.execute(
            "SELECT results_json FROM pipeline_replay_deep_progress WHERE signals > 0")) + "]"):
        for pat, results in raw.items():
            for res in results:
                rows.append({"pattern": pat, "date": res["date"], "net_r": res["net_r"], "is_win": res["is_win"]})
    sig = pd.DataFrame(rows)
    print(f"signals: {len(sig)} total, {sig.pattern.isin(LIVE_PATTERNS).sum()} from live patterns")

    dates = [d for d in sb.index if d >= first_snap and d in mrh.index]
    print(f"classifying {len(dates)} sessions {dates[0].date()} -> {dates[-1].date()}...", flush=True)
    eng = {"LIVE_NOW": {}, "OPT2": {}, "OPT1": {}, "BACKFILL": {}}
    for d in dates:
        r = sb.loc[d]; key = d.strftime("%Y-%m-%d")
        nifty = nifty_all.loc[:d]; vix = float(mrh.at[d, "vix"]) if pd.notna(mrh.at[d, "vix"]) else 15.0
        lb, ad = legacy_breadth(r["adv_rate_1d"])
        if pd.isna(ad) or pd.isna(r["above_50_pct"]):
            continue
        a50 = float(r["above_50_pct"])
        eng["LIVE_NOW"][key] = classify(nifty, vix, ad, lb, a50)
        eng["OPT2"][key] = classify(nifty, vix, ad, None, a50)
        eng["OPT1"][key] = classify(nifty, vix, ad, None if pd.isna(r["score"]) else float(r["score"]), a50)
        eng["BACKFILL"][key] = mrh.at[d, "regime"]

    rng = np.random.default_rng(20261117)
    summary = {}
    for subset, s in (("LIVE PATTERNS (Cup & Handle, Swing High Breakout)", sig[sig.pattern.isin(LIVE_PATTERNS)]),
                      ("ALL REPLAYED PATTERNS", sig)):
        print(f"\n################ {subset} ################")
        for name, m in eng.items():
            summary[(subset[:4], name)] = report(name, s, pd.Series(m), rng)

    # Today's-style check: how often does each engine call D/E on a day the
    # 10-day structure is healthy (structural score >= 5)?
    sbk = sb.copy(); sbk.index = sbk.index.strftime("%Y-%m-%d")
    healthy = set(sbk.index[sbk.score >= 5])
    print("\nD/E calls on structurally healthy days (score >= 5):")
    for name, m in eng.items():
        ser = pd.Series(m); h = ser[ser.index.isin(healthy)]
        print(f"  {name:<9} {h.isin(BLOCKED).mean():.1%} of {len(h)} healthy sessions")


if __name__ == "__main__":
    main()
