"""
NSE Momentum vNext - Exponential Regression Slope x R^2 Agent   [PROPOSED
-- NOT WIRED INTO orchestrator.py]

STATUS: prototype only. Same evidence-first discipline as every other
prototype in this codebase -- must clear validation/slope_r2_validate.py's
permutation-significance test before this touches live scoring.

WHY THIS EXISTS
    Second-round backlog item, sourced from a public GitHub repo review
    (skyte/momentum), implementing the momentum ranking formula from
    Andreas Clenow's "Stocks on the Move": fit an exponential (log-price
    OLS) regression over a trailing window, annualize the slope, and
    weight it by the regression's own R^2 (goodness of fit). Two stocks
    with identical total return over the window rank differently -- the
    one whose gain came from a smooth, consistent trend (high R^2)
    outranks the one whose gain came from a few violent, choppy jumps
    (low R^2). This is a genuinely different momentum FORMULA than
    everything else already live in this codebase (RS percentile is a
    multi-window relative-return composite; vol-adjusted momentum divides
    that same composite by realized volatility) -- worth testing as an
    independent ranking input, not a modifier layered onto RS.

    Conceptually adjacent to Frog-in-the-Pan (already SHELVED, no signal
    -- see FACTOR_LIBRARY_IMPLEMENTATION_PLAN.md) in that both reward
    "smooth" price paths over "choppy" ones of equal magnitude, but a
    different mechanism: FIP counts the fraction of up/down days, this
    measures literal trendline fit quality via R^2. Worth testing
    independently since FIP's null result doesn't necessarily predict
    this one's outcome -- they're measuring path quality via different
    math.

FORMULA (Clenow's own, unmodified):
    1. OLS-fit ln(Close) against a 0..N-1 day index over the trailing
       window.
    2. annualized_slope = (exp(slope_per_day * 252) - 1) * 100
    3. momentum_score = annualized_slope * R^2

WINDOW: 90 trading days -- Clenow's own book/paper default, used as-is
    rather than substituted for this codebase's existing w26=130 window
    (unlike Frog-in-the-Pan, which reused w26 because it was isolating a
    piece of an existing computation). This factor is testing a distinct
    formula end-to-end, so using its own source's validated parameter is
    a more faithful test of the actual published technique than forcing
    it onto an unrelated window chosen for a different measure.

DATA SOURCE: price_history_deep only (stock's own OHLCV, no benchmark
    needed -- this is an absolute-momentum measure, not relative to
    Nifty) -- no new external data.

INTERFACE: dual .passes_gate() / .score_bonus() interface, fails open,
    diagnostic-only until validated -- same as every other prototype here.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FORMATION_WINDOW = 90  # Clenow's own "Stocks on the Move" default
TRADING_DAYS_PER_YEAR = 252


def _fit_slope_r2(log_price_window: np.ndarray) -> float:
    """
    Closed-form OLS of log_price_window against a fixed 0..N-1 index,
    returns annualized_slope * R^2 in one shot (no need for slope and R^2
    separately downstream -- the product is the only thing ever scored).
    NaN if the window is degenerate (zero variance in x, which can't
    actually happen for N>1 fixed integers, kept as a defensive guard).
    """
    n = len(log_price_window)
    x = np.arange(n, dtype=float)
    x_dev = x - x.mean()
    y_dev = log_price_window - log_price_window.mean()
    ss_xx = float(np.sum(x_dev * x_dev))
    if ss_xx == 0:
        return np.nan
    ss_xy = float(np.sum(x_dev * y_dev))
    ss_yy = float(np.sum(y_dev * y_dev))
    slope = ss_xy / ss_xx
    r2 = (ss_xy * ss_xy) / (ss_xx * ss_yy) if ss_yy > 0 else 0.0
    annualized = (np.exp(slope * TRADING_DAYS_PER_YEAR) - 1.0) * 100.0
    return annualized * r2


def compute_stock_slope_r2_components(df: pd.DataFrame, window: int = FORMATION_WINDOW) -> pd.Series:
    """
    Vectorized (via rolling().apply(raw=True), one closed-form regression
    per window position -- no external regression library needed), once
    per ticker: rolling momentum_score series aligned to df's own date
    index. Rows before `window` bars of history are NaN.
    """
    log_close = np.log(df["Close"].clip(lower=1e-6))
    score = log_close.rolling(window).apply(_fit_slope_r2, raw=True)
    score.name = "slope_r2_score"
    return score


def compute_universe_slope_r2_index(components_by_ticker: dict, date, tickers: list) -> dict:
    """
    Cross-sectional step for the backtest: {ticker: slope_r2_score_float}
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


def compute_universe_slope_r2_percentiles(data_dict: dict, window: int = FORMATION_WINDOW) -> dict:
    """
    [PROTOTYPE -- not yet validated] Live cross-sectional slope*R^2
    percentile for every ticker, computed once per scan -- same
    {ticker: percentile_0_to_100} shape as every other live modifier, so
    orchestrator.py could wire this in identically IF
    validation/slope_r2_validate.py clears it. Higher percentile = a
    smoother, more consistent uptrend (or a less-bad downtrend) over the
    trailing 90 sessions.
    """
    stock_data = data_dict.get("stock_data", {})

    raw = {}
    for ticker, df in stock_data.items():
        if df.empty or len(df) < window:
            continue
        close = df["Close"].squeeze().to_numpy(dtype=float)
        log_close_win = np.log(np.clip(close[-window:], 1e-6, None))
        score = _fit_slope_r2(log_close_win)
        if np.isnan(score):
            continue
        raw[ticker] = score

    if len(raw) < 10:
        return {}

    sorted_v = np.sort(list(raw.values()))
    return {
        t: round(int(np.searchsorted(sorted_v, v, side="left")) / max(len(sorted_v), 1) * 100, 1)
        for t, v in raw.items()
    }


class SlopeR2Agent:
    """
    Classifies a precomputed slope_r2_score into gate/bonus outputs.
    Thresholds are terciles observed in validation, not a priori guesses
    -- see validation/slope_r2_validate.py.
    """

    def __init__(self, slope_r2_score: float = None):
        self.slope_r2_score = slope_r2_score

    def passes_gate(self) -> bool:
        """Fails open (True) when no slope_r2_score was computed --
        never blocks on a data gap."""
        return True  # diagnostic-only agent -- never a hard gate pending validation

    def score_bonus(self) -> float:
        """0.0 until validation says otherwise."""
        return 0.0

    def get_slope_r2_score(self):
        return self.slope_r2_score
