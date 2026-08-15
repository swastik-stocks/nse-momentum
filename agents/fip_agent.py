"""
NSE Momentum vNext - Frog-in-the-Pan (FIP) Agent   [PROPOSED -- NOT WIRED
INTO orchestrator.py]

STATUS: prototype only. Same evidence-first discipline as every other
prototype in this codebase -- must clear validation/fip_validate.py's
permutation-significance test before this touches live scoring.

WHY THIS EXISTS
    Factor-library backlog item, Da, Gurun & Warachka (2014) "Frog in the
    Pan: Continuous Information and Momentum" (J. Finance). Their finding:
    momentum continuation is stronger following price moves built from
    many small, same-direction daily increments ("continuous information")
    than following moves dominated by a few large jumps ("discrete
    information"), even when the two moves have identical total magnitude.
    A trader watching a frog jump out of a slowly heated pan (continuous)
    reacts differently than one watching a frog dropped straight into
    boiling water (discrete) -- their metaphor for under/over-reaction.

FORMULA (the paper's own "Information Discreteness" measure):
    ID = sign(PRET) x (%negative days - %positive days)
    over a trailing formation window, where PRET is the cumulative return
    over that same window and %pos/%neg are the fraction of days with
    positive/negative daily returns (flat days count toward neither).

    Lower ID = smoother, continuous trend = paper's thesis predicts
    STRONGER continuation. This module exposes the sign-flipped version,
    fip_score = -ID = sign(PRET) x (%positive days - %negative days), so
    "higher percentile = smoother = hypothesized better" matches the
    direction convention every other live modifier in rs_agent.py already
    uses (RS percentile, vol-adjusted percentile, lottery percentile).

WINDOW CHOICE: 130 trading days (~26 weeks) -- deliberately reuses the
    same w26 window compute_vol_adjusted_universe_ranks() already uses as
    its volatility-estimation window, rather than the paper's literal
    12-month formation period. Same rationale as that module's own
    docstring: this keeps every momentum-family measure in this codebase
    on one apples-to-apples lookback instead of a new arbitrary window per
    item.

HONEST SCOPE NOTE: the paper's own tests are monthly-rebalanced US-equity
    decile portfolios over 1926-2011 -- a very different sampling scheme
    from this codebase's event-driven (breakout-triggered) NSE signals.
    Never validated on NSE microstructure before this round.

DATA SOURCE: price_history_deep only (stock's own OHLCV) -- no new
    external data needed, same as items 1-3 and the Lottery Index.

INTERFACE: dual .passes_gate() / .score_bonus() interface, fails open,
    diagnostic-only until validated -- same division of labour as
    LotteryIndexAgent (agents/lottery_index_agent.py): this class just
    classifies a precomputed fip_score, the expensive cross-sectional
    percentile computation lives in the module-level functions.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FORMATION_WINDOW = 130  # ~26 weeks, matches compute_vol_adjusted_universe_ranks()'s w26


def compute_stock_fip_components(df: pd.DataFrame, window: int = FORMATION_WINDOW) -> pd.Series:
    """
    Vectorized, once per ticker: rolling fip_score series aligned to df's
    own date index. Rows before `window` bars of history are NaN
    (insufficient data).
    """
    ret = df["Close"].pct_change()
    pct_pos = (ret > 0).astype(float).rolling(window).mean()
    pct_neg = (ret < 0).astype(float).rolling(window).mean()
    pret = df["Close"] / df["Close"].shift(window) - 1.0
    sign_pret = np.sign(pret)
    fip_score = sign_pret * (pct_pos - pct_neg)
    fip_score.name = "fip_score"
    return fip_score


def compute_universe_fip_index(components_by_ticker: dict, date, tickers: list) -> dict:
    """
    Cross-sectional step for the backtest: given each ticker's precomputed
    rolling fip_score (from compute_stock_fip_components) and a specific
    date, return {ticker: fip_score_float} for that date. Tickers with NaN
    on this date (insufficient history, or a flat-PRET window where
    sign()==0) are omitted, not zero-filled.
    """
    rows = {}
    for t in tickers:
        s = components_by_ticker.get(t)
        if s is None or date not in s.index:
            continue
        v = s.loc[date]
        if pd.isna(v) or v == 0.0:
            continue
        rows[t] = float(v)
    return rows


def compute_universe_fip_percentiles(data_dict: dict, window: int = FORMATION_WINDOW) -> dict:
    """
    [PROTOTYPE -- not yet validated] Live cross-sectional fip_score
    percentile for every ticker, computed once per scan -- same
    {ticker: percentile_0_to_100} shape as compute_universe_ranks() /
    compute_vol_adjusted_universe_ranks() / compute_universe_lottery_percentiles(),
    so orchestrator.py could wire this in identically to those IF
    validation/fip_validate.py clears it.

    Uses only the LAST `window` bars of each series (today's live
    cross-section), unlike compute_stock_fip_components()'s full rolling
    history (built for the backtest's per-date walk).
    """
    stock_data = data_dict.get("stock_data", {})

    raw = {}
    for ticker, df in stock_data.items():
        if df.empty or len(df) < window + 1:
            continue
        close = df["Close"].squeeze().to_numpy(dtype=float)
        ret = np.diff(close) / close[:-1]
        win_ret = ret[-window:]
        pret = close[-1] / close[-1 - window] - 1.0
        sign_pret = np.sign(pret)
        if sign_pret == 0:
            continue
        pct_pos = float(np.mean(win_ret > 0))
        pct_neg = float(np.mean(win_ret < 0))
        fip_score = sign_pret * (pct_pos - pct_neg)
        raw[ticker] = fip_score

    if len(raw) < 10:
        return {}

    sorted_v = np.sort(list(raw.values()))
    return {
        t: round(int(np.searchsorted(sorted_v, v, side="left")) / max(len(sorted_v), 1) * 100, 1)
        for t, v in raw.items()
    }


class FIPAgent:
    """
    Classifies a precomputed fip_score (bounded in [-1, 1], sign-flipped
    Information Discreteness) into gate/bonus outputs. Thresholds are
    terciles observed in validation, not a priori guesses -- see
    validation/fip_validate.py for how they were derived and whether they
    actually predict anything.
    """

    def __init__(self, fip_score: float = None):
        self.fip_score = fip_score

    def passes_gate(self) -> bool:
        """Fails open (True) when no fip_score was computed (insufficient
        history / flat PRET) -- never blocks on a data gap."""
        return True  # diagnostic-only agent -- never a hard gate pending validation

    def score_bonus(self) -> float:
        """0.0 until validation says otherwise -- see the harness's own
        docstring for why no direction is assumed a priori."""
        return 0.0

    def get_fip_score(self):
        return self.fip_score
