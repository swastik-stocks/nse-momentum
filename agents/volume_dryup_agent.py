"""
NSE Momentum vNext - Volume Dry-Up Pre-Breakout Agent   [PROPOSED -- NOT
WIRED INTO orchestrator.py]

STATUS: prototype only. Same evidence-first discipline as every other
prototype in this codebase -- must clear validation/volume_dryup_validate.py's
permutation-significance test before this touches live scoring.

WHY THIS EXISTS
    Second-round backlog item, sourced from a public GitHub repo review
    (PKScreener, an NSE-specific open-source screener). PKScreener flags
    stocks currently trading at an N-day volume low as an early-warning
    precursor to a breakout -- the thesis being that a volume dry-up
    (participants losing interest, float tightening into fewer hands)
    often precedes an expansion move, the same "quiet before the storm"
    intuition behind Bollinger Band squeezes and this codebase's own
    VCP-contraction pattern detection, but measured on VOLUME instead of
    PRICE range.

    Conceptually distinct from the existing VCPContractionGate
    (agents/vcp_gate.py), which measures price-range contraction (the W4
    tightening percentage) -- this measures the same "coiling" intuition
    on the volume axis instead, which could carry independent information
    even for stocks whose price pattern doesn't show a clean VCP.

FORMULA: today's volume / 20-day rolling average volume -- literally the
    same ratio this codebase already calls RVOL everywhere else
    (confirm_picks.py's own intraday RVOL uses a 20-day average baseline;
    this reuses that exact convention at end-of-day granularity rather
    than inventing a new baseline window). Lower ratio = "dryer" =
    PKScreener's thesis predicts a stronger precursor signal. This module
    keeps the raw ratio (not inverted) so "lower = more validated
    interesting" reads the same way "lower RVOL" already does everywhere
    else in this codebase -- the validator decides direction, not this
    module.

WINDOW: 20 trading days, matching confirm_picks.py's own RVOL baseline
    convention exactly (not this codebase's w26=130 momentum window --
    volume dry-up is a short-horizon microstructure signal in the source
    screener, not a momentum measure, so borrowing the momentum window
    would misrepresent what's being tested).

DATA SOURCE: price_history_deep's Volume column only -- no benchmark, no
    new external data.

INTERFACE: dual .passes_gate() / .score_bonus() interface, fails open,
    diagnostic-only until validated -- same as every other prototype here.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

VOLUME_WINDOW = 20  # matches confirm_picks.py's own RVOL 20-day baseline


def compute_stock_volume_ratio_components(df: pd.DataFrame, window: int = VOLUME_WINDOW) -> pd.Series:
    """
    Vectorized, once per ticker: rolling volume-ratio series (today's
    volume / trailing N-day average volume, EXCLUDING today itself so a
    single huge-volume day doesn't inflate its own baseline) aligned to
    df's own date index. Rows before `window` bars of history are NaN.
    """
    vol = df["Volume"]
    avg_vol = vol.shift(1).rolling(window).mean()
    ratio = vol / avg_vol.replace(0, np.nan)
    ratio.name = "volume_ratio"
    return ratio


def compute_universe_volume_ratio_index(components_by_ticker: dict, date, tickers: list) -> dict:
    """
    Cross-sectional step for the backtest: {ticker: volume_ratio_float}
    for a specific date. Missing/NaN omitted, not zero-filled.
    """
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


def compute_universe_volume_ratio_percentiles(data_dict: dict, window: int = VOLUME_WINDOW) -> dict:
    """
    [PROTOTYPE -- not yet validated] Live cross-sectional volume-ratio
    percentile for every ticker, computed once per scan -- same
    {ticker: percentile_0_to_100} shape as every other live modifier, so
    orchestrator.py could wire this in identically IF
    validation/volume_dryup_validate.py clears it. Higher percentile =
    HIGHER volume ratio (more active than usual) -- a low percentile is
    the "dry-up" PKScreener's thesis is about; direction of any bonus/
    penalty is decided by the validator, not assumed here.
    """
    stock_data = data_dict.get("stock_data", {})

    raw = {}
    for ticker, df in stock_data.items():
        if df.empty or len(df) < window + 1:
            continue
        vol = df["Volume"].squeeze().to_numpy(dtype=float)
        baseline = vol[-(window + 1):-1]  # trailing window, excluding today
        avg_vol = np.mean(baseline)
        if avg_vol <= 0:
            continue
        ratio = vol[-1] / avg_vol
        raw[ticker] = float(ratio)

    if len(raw) < 10:
        return {}

    sorted_v = np.sort(list(raw.values()))
    return {
        t: round(int(np.searchsorted(sorted_v, v, side="left")) / max(len(sorted_v), 1) * 100, 1)
        for t, v in raw.items()
    }


class VolumeDryupAgent:
    """
    Classifies a precomputed volume_ratio value into gate/bonus outputs.
    Thresholds are terciles observed in validation, not a priori guesses
    -- see validation/volume_dryup_validate.py.
    """

    def __init__(self, volume_ratio: float = None):
        self.volume_ratio = volume_ratio

    def passes_gate(self) -> bool:
        """Fails open (True) when no volume_ratio was computed -- never
        blocks on a data gap."""
        return True  # diagnostic-only agent -- never a hard gate pending validation

    def score_bonus(self) -> float:
        """0.0 until validation says otherwise."""
        return 0.0

    def get_volume_ratio(self):
        return self.volume_ratio
