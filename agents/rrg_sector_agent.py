"""
NSE Momentum vNext - RRG Sector Rotation Agent   [PROPOSED -- NOT WIRED
INTO orchestrator.py]

STATUS: prototype only. Same evidence-first discipline as every other
prototype in this codebase -- must clear validation/rrg_sector_validate.py's
permutation-significance test before this touches live scoring.

WHY THIS EXISTS
    Second-round backlog item, sourced from a public GitHub repo review
    (AdroitAnandAI/RRG-Sector-Rotation-India). A Relative Rotation Graph
    plots each sector's relative strength against the market (RS-Ratio)
    on one axis and that RS-Ratio's own rate of change (RS-Momentum) on
    the other, splitting sectors into four quadrants:
        LEADING    (RS-Ratio >=100, RS-Momentum >=100) -- outperforming and accelerating
        WEAKENING  (RS-Ratio >=100, RS-Momentum  <100) -- outperforming but decelerating
        LAGGING    (RS-Ratio  <100, RS-Momentum  <100) -- underperforming and decelerating
        IMPROVING  (RS-Ratio  <100, RS-Momentum >=100) -- underperforming but accelerating

    ARCHITECTURALLY DIFFERENT from every other backlog item: this is a
    SECTOR-level regime/rotation overlay, not a stock-level ranking input.
    A stock's own score doesn't change here -- what's being tested is
    whether the SECTOR a stock belongs to being in a favorable rotation
    quadrant on the signal date predicts anything about that stock's
    forward outcome. Complements the market-breadth detector (regime.py /
    macro_agent.py) conceptually, not RSAgent's per-stock composite --
    though the validation harness still attributes quadrant to individual
    gate-cleared signals (via each stock's sector) to reuse this
    codebase's one established evidence chain rather than inventing a
    separate sector-level backtest methodology.

FORMULA (approximation, not the proprietary JdK/StockCharts algorithm --
    that exact double-smoothing formula was never fully published; this
    is a commonly-used open-source approximation, same one the surveyed
    repo implements):
    1. Build a synthetic equal-weighted sector index: daily cross-
       sectional mean of member-stock returns, compounded.
    2. RS = sector_index / benchmark_index (Nifty).
    3. RS-Ratio = 100 * RS / RS.rolling(RATIO_WINDOW).mean()
    4. RS-Momentum = 100 * RS-Ratio / RS-Ratio.shift(MOMENTUM_WINDOW)

WINDOWS: RATIO_WINDOW=55 (~11 weeks), MOMENTUM_WINDOW=10 (~2 weeks) --
    daily-data equivalents of the ~10-13 week smoothing conventional RRG
    implementations use on weekly charts (StockCharts/TradingView), scaled
    down since this codebase runs on daily bars, not weekly.

SECTOR TAXONOMY: reuses sector_breadth.py's build_ticker_sector_mapping()
    (NSE's own official Nifty 500 constituent sector list, with the same
    fallback-to-nse_universe.py behavior for any unmapped ticker) rather
    than inventing a second sector taxonomy -- one source of truth.

DATA SOURCE: price_history_deep (member-stock OHLCV) + the same Nifty
    benchmark series every other item already loads -- no new external
    data.

INTERFACE: dual .passes_gate() / .score_bonus() interface, fails open,
    diagnostic-only until validated -- same as every other prototype here.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RATIO_WINDOW = 55      # ~11 weeks, daily-data analogue of standard RRG smoothing
MOMENTUM_WINDOW = 10   # ~2 weeks, rate-of-change window on RS-Ratio


def build_sector_price_index(stock_data: dict, ticker_to_sector: dict, all_dates: pd.DatetimeIndex) -> dict:
    """
    Equal-weighted synthetic sector index per sector, aligned to
    `all_dates` (the benchmark's own date index, same canonical date list
    every other backtest item loops over). For each date, the sector's
    return is the cross-sectional mean of that day's pct_change() across
    every member stock that has a price on that date (nanmean -- a stock
    missing data on a given day is simply excluded from that day's
    average, not zero-filled). Compounded into an index starting at 100.

    Returns {sector: pd.Series indexed like all_dates}.
    """
    sector_members = {}
    for ticker, sector in ticker_to_sector.items():
        if ticker in stock_data:
            sector_members.setdefault(sector, []).append(ticker)

    sector_index = {}
    for sector, tickers in sector_members.items():
        rets = []
        for t in tickers:
            df = stock_data[t]
            close = df["Close"].reindex(all_dates, method="ffill")
            rets.append(close.pct_change())
        if not rets:
            continue
        ret_matrix = pd.concat(rets, axis=1)
        sector_ret = ret_matrix.mean(axis=1, skipna=True)
        sector_ret = sector_ret.fillna(0.0)
        index_level = 100.0 * (1.0 + sector_ret).cumprod()
        sector_index[sector] = index_level

    return sector_index


def compute_rrg_quadrants(sector_index: pd.Series, benchmark_close: pd.Series,
                           ratio_window: int = RATIO_WINDOW,
                           momentum_window: int = MOMENTUM_WINDOW) -> pd.DataFrame:
    """
    Given one sector's index series and the benchmark's close series
    (both aligned to the same date index), returns a DataFrame indexed
    the same way with columns [rs_ratio, rs_momentum, quadrant]. Rows
    before enough history for both rolling windows are NaN/None.
    """
    bench_aligned = benchmark_close.reindex(sector_index.index, method="ffill")
    rs = sector_index / bench_aligned.replace(0, np.nan)

    rs_ratio = 100.0 * rs / rs.rolling(ratio_window).mean()
    rs_momentum = 100.0 * rs_ratio / rs_ratio.shift(momentum_window)

    def _quadrant(row):
        r, m = row["rs_ratio"], row["rs_momentum"]
        if pd.isna(r) or pd.isna(m):
            return None
        if r >= 100 and m >= 100:
            return "LEADING"
        if r >= 100 and m < 100:
            return "WEAKENING"
        if r < 100 and m < 100:
            return "LAGGING"
        return "IMPROVING"

    out = pd.DataFrame({"rs_ratio": rs_ratio, "rs_momentum": rs_momentum})
    out["quadrant"] = out.apply(_quadrant, axis=1)
    return out


class RRGSectorAgent:
    """
    Classifies a precomputed sector RRG quadrant into gate/bonus outputs.
    Which quadrants (if any) get a bonus/penalty is decided by
    validation/rrg_sector_validate.py, not assumed here -- LEADING/
    IMPROVING are the RRG literature's favorable quadrants, but the point
    of validating is checking whether that transfers to NSE sector
    composition and this codebase's signal population.
    """

    def __init__(self, quadrant: str = None):
        self.quadrant = quadrant

    def passes_gate(self) -> bool:
        """Fails open (True) when no quadrant was computed (insufficient
        sector-index history) -- never blocks on a data gap."""
        return True  # diagnostic-only agent -- never a hard gate pending validation

    def score_bonus(self) -> float:
        """0.0 until validation says otherwise."""
        return 0.0

    def get_quadrant(self):
        return self.quadrant
