"""
NSE Momentum v4.3 — Market Breadth Agent
A/D ratio + % above 50 EMA + 52-week highs/lows → breadth score 0-10.
Runs FIRST each scan. Result injected into MarketAgent for regime.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class MarketBreadthAgent:
    def __init__(self, stock_data: dict, nifty_df: pd.DataFrame = None):
        self.stock_data = stock_data
        self.nifty_df   = nifty_df if nifty_df is not None else pd.DataFrame()
        self._score     = 5
        self._ad_ratio  = 1.0
        self._above_50  = 50.0
        self._new_highs = 0
        self._new_lows  = 0

    def compute(self) -> dict:
        advances = 0; declines = 0; above50 = 0; new_highs = 0; new_lows = 0
        total = 0

        for ticker, df in self.stock_data.items():
            if df.empty or len(df) < 50:
                continue
            total += 1
            c = df["Close"].squeeze().to_numpy(dtype=float)
            h = df["High"].squeeze().to_numpy(dtype=float)
            l = df["Low"].squeeze().to_numpy(dtype=float)

            # Advance/Decline
            if len(c) >= 2:
                if c[-1] > c[-2]: advances += 1
                elif c[-1] < c[-2]: declines += 1

            # Above 50 EMA
            ema50 = self._ema50(c)
            if c[-1] > ema50: above50 += 1

            # 52-week highs/lows
            w52 = 252
            if len(h) >= w52:
                if c[-1] >= max(h[-w52:]): new_highs += 1
                if c[-1] <= min(l[-w52:]): new_lows += 1

        if total == 0:
            self._score = 5
            return self._result(5)

        ad_ratio    = advances / (declines + 1)
        above50_pct = above50 / total * 100
        nh_nl       = new_highs - new_lows

        self._ad_ratio = round(ad_ratio, 2)
        self._above_50 = round(above50_pct, 1)
        self._new_highs = new_highs
        self._new_lows  = new_lows

        # Score components
        pts = 0
        if ad_ratio >= 2.0:    pts += 3
        elif ad_ratio >= 1.5:  pts += 2
        elif ad_ratio >= 1.0:  pts += 1
        elif ad_ratio < 0.7:   pts -= 1

        if above50_pct >= 70:  pts += 4
        elif above50_pct >= 55: pts += 3
        elif above50_pct >= 45: pts += 2
        elif above50_pct >= 35: pts += 1
        else:                   pts -= 1

        if nh_nl >= 40:   pts += 3
        elif nh_nl >= 15: pts += 2
        elif nh_nl >= 0:  pts += 1
        else:             pts -= 1

        self._score = max(0, min(10, pts))
        return self._result(self._score)

    def _result(self, score: int) -> dict:
        return {
            "breadth_score": score,
            "ad_ratio": self._ad_ratio,
            "above_50_pct": self._above_50,
            "new_highs": self._new_highs,
            "new_lows": self._new_lows,
        }

    @staticmethod
    def _ema50(values: np.ndarray) -> float:
        if len(values) < 2:
            return values[-1]
        alpha = 2 / 51
        e = values[0]
        for v in values[1:]:
            e = alpha * v + (1 - alpha) * e
        return e
