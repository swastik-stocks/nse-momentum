"""
NSE Momentum vNext - Lottery Index Agent   [PROPOSED -- NOT WIRED INTO
orchestrator.py]

STATUS: prototype only. Same evidence-first discipline as every other
prototype in this codebase -- must clear validation/lottery_index_validate.py's
permutation-significance test before this touches live scoring.

WHY THIS EXISTS
    Factor-library backlog item, Chung (2019) "Retail Trading and Momentum
    Profitability" -- see FACTOR_LIBRARY_IMPLEMENTATION_PLAN.md. Chung's
    core finding: momentum returns increase monotonically across quintiles
    of retail-trading participation (top-minus-bottom spread 1.42%,
    t=3.46) -- but since retail trading data doesn't exist for most of his
    sample (or for NSE, in this codebase's case), he proxies it with a
    "lottery index": stocks retail investors are known to gravitate toward
    (low price, high idiosyncratic vol/skew, high past max return).

FORMULA (confirmed against the primary source this round, not assumed):
    LotteryIndex = average of cross-sectional z-scores of:
        1. MAX  -- max daily return over the trailing window
        2. IVOL -- idiosyncratic volatility (std dev of CAPM residuals,
           stock return regressed on Nifty return, trailing window)
        3. ISKEW -- idiosyncratic skewness of those same residuals
    Higher = more "lottery-like" = proxies heavier retail participation.

HONEST SCOPE NOTE (same one flagged in the plan doc before this was built):
    Chung's proxy exists because HIS sample lacks direct retail data for
    most of a 77-year backtest -- not because the proxy is known to be
    BETTER than direct data. NSE also has no direct retail-participation
    feed, so this is buildable, but it inherits the same "proxy, not
    ground truth" caveat Chung's own paper carries, and it has never been
    tested on NSE microstructure (very different retail base, tick sizes,
    circuit filters vs 1940s-2010s US markets Chung's sample spans).

DATA SOURCE: price_history_deep only (stock's own OHLCV + Nifty OHLCV) --
    no new external data needed, same as items 1-3.

INTERFACE
    Mirrors every other prototype in this repo: dual .passes_gate() /
    .score_bonus() interface, fails open on insufficient/no data. Unlike
    InsiderSilenceAgent/PromoterFeedbackAgent, the expensive part (rolling
    MAX/IVOL/ISKEW computation + cross-sectional z-scoring against the
    point-in-time universe) happens in the validator/caller, same division
    of labour as compute_universe_ranks() in rs_agent.py -- this class
    just classifies/scores a precomputed lottery_index value.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROLLING_WINDOW = 21  # ~1 trading month, standard MAX/IVOL/ISKEW lookback in the literature


def compute_stock_lottery_components(df: pd.DataFrame, nifty_df: pd.DataFrame,
                                      window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Vectorized, once per ticker: rolling MAX / IVOL / ISKEW series aligned
    to df's own date index. IVOL/ISKEW use a rolling-window CAPM residual
    (beta estimated via rolling cov/var over the same window, applied to
    that window's own returns -- a standard, cheap approximation, not a
    true time-varying-beta model).

    Returns a DataFrame indexed like df, columns [max_ret, ivol, iskew].
    Rows before `window` bars of history are NaN (insufficient data).
    """
    ret = df["Close"].pct_change()
    nifty_aligned = nifty_df["Close"].reindex(df.index, method="ffill")
    mkt_ret = nifty_aligned.pct_change()

    roll_cov = ret.rolling(window).cov(mkt_ret)
    roll_var = mkt_ret.rolling(window).var()
    beta = (roll_cov / roll_var.replace(0, np.nan))

    resid = ret - beta * mkt_ret

    out = pd.DataFrame(index=df.index)
    out["max_ret"] = ret.rolling(window).max()
    out["ivol"]     = resid.rolling(window).std()
    out["iskew"]    = resid.rolling(window).skew()
    return out


def compute_universe_lottery_index(components_by_ticker: dict, date, tickers: list) -> dict:
    """
    Cross-sectional step: given each ticker's precomputed rolling
    components (from compute_stock_lottery_components) and a specific
    date + point-in-time universe member list, z-score each of the three
    raw components across that day's universe and average them.

    Returns {ticker: lottery_index_float}. Tickers with NaN components on
    this date (insufficient history) are omitted, not zero-filled -- a
    missing entry means "no signal," not "average lottery-ness."
    """
    rows = {}
    for t in tickers:
        comp = components_by_ticker.get(t)
        if comp is None or date not in comp.index:
            continue
        r = comp.loc[date]
        if r.isna().any():
            continue
        rows[t] = (float(r["max_ret"]), float(r["ivol"]), float(r["iskew"]))

    if len(rows) < 10:  # too few cross-sectional members to z-score meaningfully
        return {}

    arr = np.array(list(rows.values()))  # N x 3
    mu  = arr.mean(axis=0)
    sd  = arr.std(axis=0)
    sd[sd == 0] = np.nan
    z = (arr - mu) / sd
    lottery = np.nanmean(z, axis=1)

    return {t: float(v) for t, v in zip(rows.keys(), lottery) if not np.isnan(v)}


def compute_universe_lottery_percentiles(data_dict: dict, window: int = ROLLING_WINDOW) -> dict:
    """
    [LIVE, v6 -- wired into RSAgent.score(), 2026-08-14] Cross-sectional
    lottery-index percentile for every ticker, computed once per scan --
    same {ticker: percentile_0_to_100} shape as compute_universe_ranks()/
    compute_vol_adjusted_universe_ranks() in rs_agent.py, so orchestrator.py
    wires it in identically to item 1.

    Validated via validation/lottery_index_validate.py: top tercile N=825,
    WR=65.9%, Avg R 1.39 vs pool 0.84, p=0.0, consistent (though decaying:
    1.78->1.00) across both halves of a 2007-2026, 2,500-signal replay.
    Bottom tercile underperformed the pool (WR=53.7%, p=1.0 -- i.e.
    reliably BELOW almost every random pool draw, not just "not better").

    Uses only the LAST `window` bars of each series (today's live
    cross-section), unlike compute_stock_lottery_components()'s full
    rolling history (built for the backtest's per-date walk) -- simpler
    and sufficient since live scoring only ever needs "today."
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

        max_ret = float(np.max(stock_win))
        ivol = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
        iskew = float(pd.Series(resid).skew())
        if np.isnan(iskew):
            continue
        raw[ticker] = (max_ret, ivol, iskew)

    if len(raw) < 10:
        return {}

    arr = np.array(list(raw.values()))
    mu, sd = arr.mean(axis=0), arr.std(axis=0)
    sd[sd == 0] = np.nan
    z = (arr - mu) / sd
    lottery = np.nanmean(z, axis=1)

    valid = {t: v for t, v in zip(raw.keys(), lottery) if not np.isnan(v)}
    if not valid:
        return {}
    sorted_v = np.sort(list(valid.values()))
    return {
        t: round(int(np.searchsorted(sorted_v, v, side="left")) / max(len(sorted_v), 1) * 100, 1)
        for t, v in valid.items()
    }


class LotteryIndexAgent:
    """
    Classifies a precomputed lottery_index value (cross-sectional z-score
    average, roughly centered at 0, unbounded) into gate/bonus outputs.
    Thresholds are terciles observed in validation, not a priori guesses
    -- see validation/lottery_index_validate.py for how they were derived
    and whether they actually predict anything.
    """

    def __init__(self, lottery_index: float = None):
        self.lottery_index = lottery_index

    def passes_gate(self) -> bool:
        """Fails open (True) when no lottery_index was computed (insufficient
        history / too-thin cross-section) -- never blocks on a data gap."""
        if self.lottery_index is None:
            return True
        return True  # diagnostic-only agent -- never a hard gate pending validation

    def score_bonus(self) -> float:
        """0.0 until validation says otherwise -- see the harness's own
        docstring for why no direction is assumed a priori."""
        return 0.0

    def get_lottery_index(self):
        return self.lottery_index
