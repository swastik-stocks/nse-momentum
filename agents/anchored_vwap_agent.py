"""
NSE Momentum vNext - Anchored VWAP Agent   [PROPOSED -- NOT WIRED INTO
orchestrator.py]

STATUS: prototype only. Same evidence-first discipline as every other
prototype in this codebase -- must clear validation/anchored_vwap_validate.py's
in-sample + holdout + walk-forward significance tests before this touches
live scoring.

WHY THIS EXISTS
    From the broader prorealcode.com search (Anchored VWAP / Order Flow /
    Volume Profile) the user asked for: "Auto Midas Anchored VWAP" --
    auto-anchors volume-weighted price to a recent swing extreme across 4
    hierarchical lookback windows (17/72/305 bars, 1292 -- roughly a
    month, a quarter, a year, five years), producing dynamic support/
    resistance bands. Stated uses: the macro band as trend confirmation,
    the intermediate bands as "value zone" pullback-entry references, and
    extension from the shortest band as a mean-reversion exhaustion flag.

WHAT'S GENUINELY DIFFERENT FROM A PLAIN ROLLING VWAP: the anchor point
    isn't a fixed N bars back -- it's the bar with the LOWEST LOW (for a
    support/"Bottom" VWAP) within the trailing lookback window, and the
    VWAP is computed from THAT anchor bar forward to the current bar (a
    variable-length window, not a fixed one). This module implements
    only the support-side anchor (matching the source's "Bottom VWAP");
    the resistance-side ("Top VWAP", anchored to the highest high) is a
    mechanical mirror, not built here to keep the first test scoped and
    cheap -- expand only if this MVP shows promise, same discipline
    applied to Hameed's turnover-spike proxy earlier tonight.

LEVEL CHOSEN FOR THIS FIRST TEST: 72 bars (~1 quarter), the source's own
    "value zone" intermediate level -- long enough to be a real
    structural reference, short enough to still be relevant to this
    codebase's weeks-to-months pattern-formation and holding horizons
    (unlike the 1292-bar macro level, which describes multi-year trend
    context closer to a regime signal than a per-stock ranking factor).

RISK FLAGGED IN ADVANCE (from the research-to-validation plan): this
    codebase already tested a conceptually similar "proximity to a
    reference level" factor -- 52-week-high proximity, from the original
    Excel library -- and it INVERTED (the bottom tercile, further from
    the level, beat the top tercile). No assumption is made here about
    which direction (close-to-support vs far-above-support) wins; the
    validator tests both terciles, same as every other factor tonight.

DATA SOURCE: OHLCV only, already loaded everywhere in this codebase.
    Typical price (H+L+C)/3 used for the VWAP price input, the standard
    convention, not Close alone.
"""

import numpy as np
import pandas as pd

AVWAP_LOOKBACK = 72  # ~1 quarter, the source indicator's "value zone" intermediate level


def compute_stock_support_avwap_components(df: pd.DataFrame, lookback: int = AVWAP_LOOKBACK) -> pd.Series:
    """
    Vectorized (except one O(n) pass using precomputed cumulative sums,
    not a nested loop), once per ticker: for each bar, finds the anchor
    = the bar with the lowest Low within the trailing `lookback` window,
    then computes VWAP (typical-price-weighted) from that anchor bar
    through the current bar. Returns a Series aligned to df's index:
    (close - support_avwap) / support_avwap -- the pct-above-support
    distance, NOT the raw AVWAP level itself (this is what gets
    percentile-ranked across the universe).
    """
    high, low, close, vol = df["High"], df["Low"], df["Close"], df["Volume"]
    typical_price = (high + low + close) / 3.0
    pv = typical_price * vol

    cum_pv = pv.cumsum().to_numpy()
    cum_vol = vol.cumsum().to_numpy()
    low_arr = low.to_numpy()
    close_arr = close.to_numpy()
    n = len(df)

    # Rolling argmin of Low within the trailing `lookback` window (raw=True
    # for numpy-level speed, not a Python-level loop per bar).
    anchor_offset = low.rolling(lookback).apply(lambda x: x.argmin(), raw=True)
    anchor_offset_arr = anchor_offset.to_numpy()

    pct_above_support = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        offset = anchor_offset_arr[i]
        if np.isnan(offset):
            continue
        window_start = i - lookback + 1
        anchor_idx = int(window_start + offset)
        prior_pv = cum_pv[anchor_idx - 1] if anchor_idx > 0 else 0.0
        prior_vol = cum_vol[anchor_idx - 1] if anchor_idx > 0 else 0.0
        denom = cum_vol[i] - prior_vol
        if denom <= 0:
            continue
        support_avwap = (cum_pv[i] - prior_pv) / denom
        if support_avwap > 0:
            pct_above_support[i] = (close_arr[i] - support_avwap) / support_avwap

    result = pd.Series(pct_above_support, index=df.index, name="pct_above_support_avwap")
    return result


def compute_universe_avwap_index(components_by_ticker: dict, date, tickers: list) -> dict:
    """Cross-sectional step for the backtest: {ticker: pct_above_support_avwap}
    for a specific date. Missing/NaN omitted, not zero-filled."""
    rows = {}
    for t in tickers:
        s = components_by_ticker.get(t)
        if s is None or date not in s.index:
            continue
        v = s.loc[date]
        if pd.isna(v):
            continue
        rows[t] = float(v)
    return rows


def compute_universe_avwap_percentiles(data_dict: dict, lookback: int = AVWAP_LOOKBACK) -> dict:
    """
    [PROTOTYPE -- not yet validated] Live cross-sectional pct-above-
    support-AVWAP percentile for every ticker, computed once per scan --
    same {ticker: percentile_0_to_100} shape as every other live
    modifier. Higher percentile = further above the support-anchored
    VWAP -- direction of any bonus/penalty is decided by the validator,
    not assumed here (see the 52-week-high inversion risk noted above).
    """
    stock_data = data_dict.get("stock_data", {})
    raw = {}
    for ticker, df in stock_data.items():
        if df.empty or len(df) < lookback + 1:
            continue
        components = compute_stock_support_avwap_components(df, lookback=lookback)
        val = components.iloc[-1]
        if pd.isna(val):
            continue
        raw[ticker] = float(val)

    if len(raw) < 10:
        return {}

    sorted_v = np.sort(list(raw.values()))
    return {
        t: round(int(np.searchsorted(sorted_v, v, side="left")) / max(len(sorted_v), 1) * 100, 1)
        for t, v in raw.items()
    }


class AnchoredVWAPAgent:
    """
    Classifies a precomputed pct_above_support_avwap value into
    gate/bonus outputs. Thresholds are terciles observed in validation,
    not a priori guesses -- see validation/anchored_vwap_validate.py.
    """

    def __init__(self, pct_above_support_avwap: float = None):
        self.pct_above_support_avwap = pct_above_support_avwap

    def passes_gate(self) -> bool:
        return True  # diagnostic-only agent -- never a hard gate pending validation

    def score_bonus(self) -> float:
        return 0.0  # 0.0 until validation says otherwise

    def get_pct_above_support_avwap(self):
        return self.pct_above_support_avwap
