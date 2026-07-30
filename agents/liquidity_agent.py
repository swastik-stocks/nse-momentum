"""
NSE Momentum v4.3 — Liquidity Agent
ADT + participation rate + market cap. Universe-aware thresholds.
Acts as gate — stocks below threshold are rejected.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class LiquidityAgent:
    def __init__(self, df: pd.DataFrame, universe: str = "LARGE",
                 mcap_cr: float = 0.0, bars_per_day: float = 1.0):
        self.df      = df
        self.universe= universe
        self.mcap_cr = mcap_cr
        # [NEW] bars_per_day=1 (default) is exactly the original daily
        # behavior — the live scanner never passes this, so it is
        # unaffected. The hourly replay passes bars_per_day=7 (see
        # hourly_scaling.py) so the "20-day" ADT/participation window
        # below becomes a genuine ~20-trading-day (140-hourly-bar) window
        # instead of silently becoming a 20-HOUR (~3-day) window.
        self._window = max(1, round(20 * bars_per_day))
        self._adt_cr    = 0.0
        self._part_rate = 0.0
        self._passes    = False
        self._reason    = ""
        self._liq_score = 0
        self._mcap_tier = ""
        self._compute()

    def _compute(self):
        df = self.df
        w = self._window
        if df.empty or len(df) < w:
            self._reason = "Insufficient data"
            return

        # ADT in crores
        close = df["Close"].squeeze().to_numpy(dtype=float)
        vol   = df["Volume"].squeeze().to_numpy(dtype=float)
        daily_turnover = close * vol / 1e7  # crores
        self._adt_cr = float(np.mean(daily_turnover[-w:]))

        # Participation rate: % of days with above-average volume
        avg20v = float(np.mean(vol[-w:])) if np.mean(vol[-w:]) > 0 else 1
        self._part_rate = float(np.sum(vol[-w:] > avg20v) / w * 100)

        # Market cap tier
        if self.mcap_cr >= 20000:    self._mcap_tier = "LargeCap"
        elif self.mcap_cr >= 5000:   self._mcap_tier = "MidCap"
        elif self.mcap_cr >= 500:    self._mcap_tier = "SmallCap"
        elif self.mcap_cr > 0:       self._mcap_tier = "MicroCap"
        else:                        self._mcap_tier = "Unknown"

        # Universe thresholds
        thresholds = {
            "LARGE": {"min_adt": 50.0,  "min_part": 40.0},
            "MID":   {"min_adt": 10.0,  "min_part": 35.0},
            "SMALL": {"min_adt":  3.0,  "min_part": 30.0},
        }
        cfg = thresholds.get(self.universe, thresholds["LARGE"])

        if self._adt_cr < cfg["min_adt"]:
            self._reason = f"ADT Rs.{self._adt_cr:.1f}Cr below min Rs.{cfg['min_adt']}Cr"
            return

        # Participation rate is stored for reference but NOT used as a gate
        # (mathematically, stocks with institutional buying show LOW participation
        #  since institutions buy in concentrated bursts, not every day)
        self._passes = True
        # Liquidity score 0-10
        pts = 0
        if self._adt_cr >= 500:  pts += 5
        elif self._adt_cr >= 100: pts += 4
        elif self._adt_cr >= 50:  pts += 3
        elif self._adt_cr >= 20:  pts += 2
        else:                     pts += 1

        if self._part_rate >= 65: pts += 3
        elif self._part_rate >= 55: pts += 2
        elif self._part_rate >= 45: pts += 1

        if self.mcap_cr >= 50000: pts += 2
        elif self.mcap_cr >= 10000: pts += 1

        self._liq_score = min(pts, 10)

    def passes(self) -> bool:        return self._passes
    def reject_reason(self) -> str:  return self._reason
    def score(self) -> int:          return self._liq_score
    def get_adt(self) -> float:      return round(self._adt_cr, 1)
    def get_part_rate(self) -> float: return round(self._part_rate, 1)
    def get_mcap_tier(self) -> str:  return self._mcap_tier
