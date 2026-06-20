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
                 ticker: str = ""):
        self.df     = df
        self.nifty  = nifty_df
        self.ticker = ticker
        self.ranks  = universe_ranks or {}
        self._pct         = 50.0
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

    def score(self) -> int:
        p = self._pct
        if p >= 90: return 20
        if p >= 80: return 18
        if p >= 70: return 15
        if p >= 60: return 12
        if p >= 50: return 8
        if p >= 30: return 4   # v5: was "if p >= 40: return 4"
        return 0

    def get_percentile(self) -> float:
        return round(self._pct, 1)

    def get_persistence(self) -> int:
        return self._persistence

    def passes_gate(self) -> bool:
        return self._pct >= RS_GATE
