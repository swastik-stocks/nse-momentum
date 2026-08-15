"""
NSE Momentum vNext - Skewness-Penalized Composite Agent   [PROPOSED -- NOT
WIRED INTO orchestrator.py]

STATUS: prototype only. Same evidence-first discipline as every other
prototype in this codebase -- must clear validation/skew_penalty_validate.py's
permutation-significance test before this touches live scoring.

WHY THIS EXISTS
    Second-round backlog item, sourced from a public GitHub repo review
    (tanish35/Momentum-Investing) rather than the original 8-paper set.
    That repo's composite ranking formula includes a linear penalty on
    90-day return skewness alongside momentum and "Frog-in-the-Pan" path
    consistency: score = w_m*Momentum + w_f*FIP - w_s*Skewness.

    Deliberately tested in ISOLATION here rather than as a blended
    composite: this codebase already validated and wired in the Lottery
    Index (agents/lottery_index_agent.py), whose z-score average BLENDS
    idiosyncratic skewness together with MAX and IVOL and found the top
    tercile (high skew among other things) outperforms. A standalone
    "penalize high skew" factor tests the OPPOSITE direction from what
    that blended result implied -- worth checking on its own terms
    instead of assuming either literature thesis (Chung's lottery-demand
    story vs. the skewness-preference overpricing story) transfers to
    NSE, since the two disagree with each other about the expected sign.

FORMULA: idiosyncratic skewness of CAPM residuals (stock return regressed
    on Nifty return) over a trailing window -- literally the same
    computation as the ISKEW component of agents/lottery_index_agent.py's
    compute_stock_lottery_components(), reused directly rather than
    reimplemented (DRY: this module isolates that one column instead of
    duplicating the rolling-beta/residual math).

WINDOW: 21 trading days (~1 month), matching lottery_index_agent.py's
    ROLLING_WINDOW so this is a true apples-to-apples isolation of one
    piece of an already-tested blend, not a differently-scoped new window.

DATA SOURCE: price_history_deep only -- no new external data.

INTERFACE: dual .passes_gate() / .score_bonus() interface, fails open,
    diagnostic-only until validated -- same as every other prototype here.
"""

import logging

import numpy as np
import pandas as pd

from agents.lottery_index_agent import compute_stock_lottery_components, ROLLING_WINDOW

log = logging.getLogger(__name__)


def compute_stock_skew_components(df: pd.DataFrame, nifty_df: pd.DataFrame,
                                   window: int = ROLLING_WINDOW) -> pd.Series:
    """
    Thin wrapper: reuses compute_stock_lottery_components() (already
    computes rolling idiosyncratic skew as part of its 3-column output)
    and isolates just the iskew column, so the underlying CAPM-residual
    math is never duplicated between the two modules.
    """
    comp = compute_stock_lottery_components(df, nifty_df, window=window)
    return comp["iskew"]


def compute_universe_skew_index(components_by_ticker: dict, date, tickers: list) -> dict:
    """
    Cross-sectional step for the backtest: {ticker: iskew_float} for a
    specific date, raw (not z-scored -- unlike the Lottery Index this is a
    single measure, not a 3-way blend, so no cross-sectional standardization
    is needed to combine it with anything else). Missing/NaN omitted.
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


def compute_universe_skew_percentiles(data_dict: dict, window: int = ROLLING_WINDOW) -> dict:
    """
    [PROTOTYPE -- not yet validated] Live cross-sectional idiosyncratic-skew
    percentile for every ticker, computed once per scan -- same
    {ticker: percentile_0_to_100} shape as every other live modifier, so
    orchestrator.py could wire this in identically IF
    validation/skew_penalty_validate.py clears it. Higher percentile =
    more positively skewed (bigger/more frequent upside outliers relative
    to the residual distribution) -- direction of any bonus/penalty is
    decided by the validator, not assumed here.
    """
    nifty = data_dict.get("nifty50_data", pd.DataFrame())
    stock_data = data_dict.get("stock_data", {})
    if nifty.empty or len(nifty) < window + 1:
        return {}
    nifty_ret = nifty["Close"].squeeze().pct_change().to_numpy(dtype=float)
    if len(nifty_ret) < window:
        return {}
    mkt_win = nifty_ret[-window:]

    raw = {}
    for ticker, df in stock_data.items():
        if df.empty or len(df) < window + 1:
            continue
        ret = df["Close"].squeeze().pct_change().to_numpy(dtype=float)
        if len(ret) < window:
            continue
        stock_win = ret[-window:]

        cov = float(np.cov(stock_win, mkt_win)[0, 1])
        var = float(np.var(mkt_win))
        beta = cov / var if var > 0 else 0.0
        resid = stock_win - beta * mkt_win

        iskew = float(pd.Series(resid).skew())
        if np.isnan(iskew):
            continue
        raw[ticker] = iskew

    if len(raw) < 10:
        return {}

    sorted_v = np.sort(list(raw.values()))
    return {
        t: round(int(np.searchsorted(sorted_v, v, side="left")) / max(len(sorted_v), 1) * 100, 1)
        for t, v in raw.items()
    }


class SkewPenaltyAgent:
    """
    Classifies a precomputed idiosyncratic-skew value into gate/bonus
    outputs. Thresholds are terciles observed in validation, not a priori
    guesses -- see validation/skew_penalty_validate.py.
    """

    def __init__(self, iskew: float = None):
        self.iskew = iskew

    def passes_gate(self) -> bool:
        """Fails open (True) when no iskew was computed -- never blocks
        on a data gap."""
        return True  # diagnostic-only agent -- never a hard gate pending validation

    def score_bonus(self) -> float:
        """0.0 until validation says otherwise."""
        return 0.0

    def get_iskew(self):
        return self.iskew
