"""
NSE Momentum v5.0 - RS Agent
RS percentile vs full universe. Runs once per scan.
Weights: 4w=40%, 12w=40%, 26w=20%

v5 changes:
- RS gate lowered from 40th to 30th percentile
- Added rs_sector: stock vs sector peers
- Added rs_persistence: weeks in top quartile (last 13 weeks)
- Composite stored for RankingFunnel prioritisation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict

log = logging.getLogger(__name__)

RS_GATE = 30   # v5: lowered from 40 to catch recovering leaders like Polycab


def compute_sector_relative_ranks(data_dict: Dict, universe_meta: Dict[str, str]) -> Dict[str, float]:
    """
    Mansfield-style sector-relative RS — stock's return vs its OWN sector's
    peer-average return, expressed as a percentile WITHIN that sector's stock
    pool (not the whole universe).

    WHY THIS EXISTS: compute_universe_ranks() above answers "how does this
    stock rank against the whole ~401-stock universe." That has a known bias
    — a stock can be a 90th-percentile mover in this constrained pool while
    only being middling against its own sector or the broader Nifty 500. A
    Financials stock beating the whole universe mostly means it's beating a
    lot of non-Financials stocks, which isn't the comparison a sector-rotation
    trader actually cares about. This function answers the narrower, more
    relevant question: is this stock a LEADER OR LAGGARD WITHIN ITS OWN
    SECTOR — the same logic SectorAgent already uses to rank sectors
    themselves, just applied one level down to individual stocks.

    This was documented as a v5 change in this file's own changelog
    ("Added rs_sector: stock vs sector peers") but never actually built —
    this is that implementation.

    Args:
        data_dict: same dict passed to compute_universe_ranks (needs
            'stock_data' and ideally 'nifty500_data' or 'nifty50_data')
        universe_meta: {ticker: sector_name} — same mapping SectorAgent uses,
            sourced from nse_universe.py's per-stock sector tags

    Returns:
        {ticker: percentile_0_to_100} — percentile is computed against ONLY
        that stock's sector peers, not the full universe. Sectors with fewer
        than 5 members return 50.0 (neutral) for all members — too small a
        pool for a percentile rank to mean anything.
    """
    stock_data = data_dict.get("stock_data", {})
    if not stock_data or not universe_meta:
        return {}

    # Group tickers by sector first
    sector_members: Dict[str, list] = {}
    for ticker in stock_data:
        sector = universe_meta.get(ticker, "Unknown")
        sector_members.setdefault(sector, []).append(ticker)

    result: Dict[str, float] = {}

    for sector, tickers in sector_members.items():
        if len(tickers) < 5:
            # Too few peers for a percentile to be meaningful — neutral default
            for t in tickers:
                result[t] = 50.0
            continue

        sector_scores = {}
        for ticker in tickers:
            df = stock_data.get(ticker)
            if df is None or df.empty or len(df) < 60:
                continue
            c = df["Close"].squeeze().to_numpy(dtype=float)

            def _ret(bars):
                return float(c[-1] / c[-bars] - 1) if len(c) >= bars and c[-bars] > 0 else 0.0

            # Same 40/40/20 weighting as the universe-wide calc, for consistency
            s4, s12, s26 = _ret(20), _ret(60), _ret(130) if len(c) >= 130 else _ret(60)
            sector_scores[ticker] = 0.40 * s4 + 0.40 * s12 + 0.20 * s26

        if not sector_scores:
            for t in tickers:
                result[t] = 50.0
            continue

        vals = np.array(list(sector_scores.values()))
        sorted_v = np.sort(vals)
        for ticker, raw in sector_scores.items():
            rank = int(np.searchsorted(sorted_v, raw, side="left"))
            result[ticker] = round(rank / max(len(sorted_v), 1) * 100, 1)

        # Any tickers that didn't have enough data got skipped above — backfill neutral
        for t in tickers:
            if t not in result:
                result[t] = 50.0

    return result


def compute_universe_ranks(data_dict: Dict) -> Dict[str, float]:
    """
    Pre-compute RS percentile for every ticker. Called ONCE per scan.
    Returns {ticker: percentile_0_to_100}.
    """
    scores     = {}
    nifty      = data_dict.get("nifty50_data", pd.DataFrame())
    stock_data = data_dict.get("stock_data", {})

    if nifty.empty or len(nifty) < 65:
        # No benchmark — rank stocks vs each other on 12-week return
        for ticker, df in stock_data.items():
            if not df.empty and len(df) >= 60:
                c   = df["Close"].squeeze().to_numpy(dtype=float)
                ret = float(c[-1] / c[-60] - 1) if c[-60] > 0 else 0.0
                scores[ticker] = ret
    else:
        nifty_c = nifty["Close"].squeeze().to_numpy(dtype=float)

        def _nret(bars):
            return float(nifty_c[-1] / nifty_c[-bars] - 1) \
                   if len(nifty_c) >= bars and nifty_c[-bars] > 0 else 0.0

        n4  = _nret(20)
        n12 = _nret(60)
        n26 = _nret(130)

        for ticker, df in stock_data.items():
            if df.empty or len(df) < 20:
                continue
            c = df["Close"].squeeze().to_numpy(dtype=float)

            def _ret(bars):
                return float(c[-1] / c[-bars] - 1) \
                       if len(c) >= bars and c[-bars] > 0 else 0.0

            s4  = _ret(20)
            s12 = _ret(60)
            s26 = _ret(130) if len(c) >= 130 else s12

            # 40% on 4w, 40% on 12w, 20% on 26w
            rs_raw = 0.40 * (s4 - n4) + 0.40 * (s12 - n12) + 0.20 * (s26 - n26)
            scores[ticker] = rs_raw

    if not scores:
        return {}

    vals     = np.array(list(scores.values()))
    sorted_v = np.sort(vals)
    result   = {}
    for ticker, raw in scores.items():
        rank = int(np.searchsorted(sorted_v, raw, side="left"))
        pct  = round(rank / max(len(sorted_v), 1) * 100, 1)
        result[ticker] = pct

    return result


def compute_rs_persistence(close: np.ndarray, nifty_close: np.ndarray,
                            weeks: int = 13) -> int:
    """
    Count weeks spent in top quartile (75th+ percentile) over last N weeks.
    Returns 0-13. Higher = more persistent leader.
    """
    if len(close) < weeks * 5 + 5 or len(nifty_close) < weeks * 5 + 5:
        return 0
    count = 0
    for w in range(weeks):
        end   = -(w * 5) if w > 0 else len(close)
        start = end - 20 if w > 0 else -20
        try:
            s_ret = close[end-1] / close[start] - 1 if close[start] > 0 else 0
            n_ret = nifty_close[end-1] / nifty_close[start] - 1 if nifty_close[start] > 0 else 0
            if s_ret - n_ret > 0.02:   # outperforming by >2% this week = top quartile proxy
                count += 1
        except (IndexError, ZeroDivisionError):
            pass
    return count


class RSAgent:
    """Per-stock RS agent. Uses pre-computed universe ranks."""

    def __init__(self, df: pd.DataFrame, nifty_df: pd.DataFrame,
                 nifty500_df: pd.DataFrame = None,
                 universe_ranks: Dict[str, float] = None,
                 sector_ranks: Dict[str, float] = None,
                 ticker: str = ""):
        self.df     = df
        self.nifty  = nifty_df
        self.ticker = ticker
        self.ranks  = universe_ranks or {}
        self.sector_pcts = sector_ranks or {}   # NEW — Mansfield sector-relative percentile
        self._pct         = 50.0
        self._sector_pct  = 50.0
        self._persistence = 0
        self._compute()

    def _compute(self):
        if self.ticker and self.ticker in self.ranks:
            self._pct = self.ranks[self.ticker]
        elif not self.df.empty and not self.nifty.empty and len(self.nifty) >= 20:
            c = self.df["Close"].squeeze().to_numpy(dtype=float)
            n = self.nifty["Close"].squeeze().to_numpy(dtype=float)

            def _ret(arr, bars):
                return float(arr[-1] / arr[-bars] - 1) \
                       if len(arr) >= bars and arr[-bars] > 0 else 0.0

            rs4  = _ret(c, 20)  - _ret(n, 20)
            rs12 = _ret(c, 60)  - _ret(n, 60)
            rs26 = _ret(c, 130) - _ret(n, 130)
            composite = 0.40 * rs4 + 0.40 * rs12 + 0.20 * rs26
            self._pct = 65.0 if composite > 0.02 else (
                55.0 if composite > 0 else (
                45.0 if composite > -0.02 else 35.0))
        else:
            self._pct = 50.0

        # RS persistence (v5)
        if not self.df.empty and not self.nifty.empty:
            c = self.df["Close"].squeeze().to_numpy(dtype=float)
            n = self.nifty["Close"].squeeze().to_numpy(dtype=float)
            self._persistence = compute_rs_persistence(c, n)

        # NEW — Mansfield sector-relative percentile (from pre-computed sector_pcts,
        # falls back to universe percentile if this ticker has no sector data —
        # e.g. sector too small, or ticker missing from universe_meta)
        self._sector_pct = self.sector_pcts.get(self.ticker, self._pct)

    def score(self) -> int:
        p = self._pct
        if p >= 90: base = 20
        elif p >= 80: base = 18
        elif p >= 70: base = 15
        elif p >= 60: base = 12
        elif p >= 50: base = 8
        elif p >= 30: base = 4   # v5: was "if p >= 40: return 4"
        else: base = 0

        # NEW — small additive bonus/penalty for sector-relative standing.
        # Deliberately modest (+/-2 max) so this REFINES the existing
        # universe-wide RS score rather than replacing it — a stock can
        # still pass on broad-universe strength alone; this just rewards
        # genuine sector leadership on top, and mildly flags a stock that's
        # only outperforming because the comparison pool is broad.
        if self._sector_pct >= 90:
            base += 2
        elif self._sector_pct >= 75:
            base += 1
        elif self._sector_pct < 25:
            base -= 1

        return max(0, min(base, 20))  # cap held at 20 — see note above on why not raised

    def get_percentile(self) -> float:
        return round(self._pct, 1)

    def get_sector_percentile(self) -> float:
        """NEW — percentile within the stock's own sector, not the full universe."""
        return round(self._sector_pct, 1)

    def get_persistence(self) -> int:
        return self._persistence

    def passes_gate(self) -> bool:
        return self._pct >= RS_GATE
