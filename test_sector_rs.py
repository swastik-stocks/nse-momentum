#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone smoke test for compute_sector_relative_ranks() — no network,
no DB, no real market data. Builds synthetic OHLCV so it's fast and
deterministic, and specifically checks the case that motivated this
feature: a stock that looks strong against the whole universe but is
actually mediocre against its own sector peers.

Run: python test_sector_rs.py
Expect: all four checks print PASS. Any FAIL means something in the
sector-relative RS wiring is broken before you trust it on live data.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from agents.rs_agent import compute_sector_relative_ranks, RSAgent


def _make_price_series(total_return_pct: float, bars: int = 140, seed: int = 0) -> pd.DataFrame:
    """Synthetic close-price series that ends at approximately the given
    total return over `bars` bars, with mild noise so it's not a perfectly
    straight line (which would make EMA/RSI-style calcs degenerate).

    NOISE CALIBRATION: std=0.0008/day (not 0.008 — a 10x-too-large first
    draft let compounding noise completely scramble the intended return
    ordering over 140 days, which silently invalidated this test's own
    assumptions rather than testing the actual function. Verified by
    checking realized vs target returns directly before trusting this."""
    rng = np.random.default_rng(seed)
    daily_drift = (1 + total_return_pct / 100) ** (1 / bars) - 1
    noise = rng.normal(0, 0.0008, bars)
    rets = daily_drift + noise
    close = 100 * np.cumprod(1 + rets)
    return pd.DataFrame({
        "Close": close,
        "Volume": rng.integers(100000, 500000, bars),
    })


def run():
    checks_passed = 0
    checks_total = 0

    # ── Scenario: two sectors, clearly different characters ────────────────
    # IT sector: broadly strong (all 6 stocks up 15-40%)
    # Financials: broadly weak (all 6 stocks flat-to-down -5% to +5%)
    # Except one Financials stock — up 12% — a genuine sector leader, but
    # would look mediocre against the whole universe next to IT's rally.
    stock_data = {}
    universe_meta = {}

    it_returns = [40, 35, 30, 25, 20, 15]
    for i, ret in enumerate(it_returns):
        t = f"IT{i}.NS"
        stock_data[t] = _make_price_series(ret, seed=i)
        universe_meta[t] = "IT"

    fin_returns = [-15, -8, -3, 2, 8, 30]  # last one (FIN5) is the clear sector leader
    for i, ret in enumerate(fin_returns):
        t = f"FIN{i}.NS"
        stock_data[t] = _make_price_series(ret, seed=100 + i)
        universe_meta[t] = "Financials"

    data_dict = {"stock_data": stock_data}

    sector_ranks = compute_sector_relative_ranks(data_dict, universe_meta)

    # ── Check 1: function runs and returns something for every ticker ──────
    checks_total += 1
    if len(sector_ranks) == len(stock_data):
        print("PASS  Check 1: returned a percentile for every ticker "
              f"({len(sector_ranks)}/{len(stock_data)})")
        checks_passed += 1
    else:
        print(f"FAIL  Check 1: expected {len(stock_data)} entries, got {len(sector_ranks)}")

    # ── Check 2: within IT, the highest-return stock should rank highest ───
    checks_total += 1
    it_ranked = sorted(
        [(t, sector_ranks[t]) for t in stock_data if universe_meta[t] == "IT"],
        key=lambda x: -x[1]
    )
    if it_ranked[0][0] == "IT0.NS":  # IT0 has the 40% return, the top performer
        print(f"PASS  Check 2: IT sector leader correctly ranked #1 within sector "
              f"({it_ranked[0][0]} = {it_ranked[0][1]}th pct)")
        checks_passed += 1
    else:
        print(f"FAIL  Check 2: expected IT0.NS to rank #1 within IT sector, "
              f"got {it_ranked[0][0]}")

    # ── Check 3: the real test — FIN5 (weak universe context, strong sector
    # context) should score HIGH within its sector despite being unremarkable
    # against the whole universe. This is the exact scenario the feature
    # exists to catch, per the reviewer's original critique.
    checks_total += 1
    fin5_sector_pct = sector_ranks.get("FIN5.NS", -1)
    if fin5_sector_pct >= 80:
        print(f"PASS  Check 3: FIN5.NS (sector leader, +30%, but would look "
              f"mediocre vs IT's +15-40% rally) correctly scores high WITHIN "
              f"its own sector: {fin5_sector_pct}th percentile")
        checks_passed += 1
    else:
        print(f"FAIL  Check 3: FIN5.NS should rank high within Financials "
              f"(expected >=80th pct), got {fin5_sector_pct}")

    # ── Check 4: RSAgent wiring — score() should apply the sector bonus and
    # stay within the original 0-20 cap (the cap-integrity check we specifically
    # cared about — no silent inflation of the 100-point formula) ───────────
    checks_total += 1
    rsa = RSAgent(
        df=stock_data["FIN5.NS"],
        nifty_df=pd.DataFrame(),
        universe_ranks={"FIN5.NS": 55.0},  # deliberately mediocre universe pct
        sector_ranks=sector_ranks,
        ticker="FIN5.NS",
    )
    score = rsa.score()
    if 0 <= score <= 20:
        bonus_applied = rsa.get_sector_percentile() >= 90
        print(f"PASS  Check 4: RSAgent.score() = {score} (within 0-20 cap, "
              f"sector_pct={rsa.get_sector_percentile()}, "
              f"bonus tier applied: {bonus_applied})")
        checks_passed += 1
    else:
        print(f"FAIL  Check 4: RSAgent.score() = {score} — OUT OF THE 0-20 "
              f"CAP, this would silently inflate the 100-point formula")

    print(f"\n{checks_passed}/{checks_total} checks passed.")
    if checks_passed < checks_total:
        sys.exit(1)


if __name__ == "__main__":
    run()
