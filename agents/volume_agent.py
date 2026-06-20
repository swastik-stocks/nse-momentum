"""
NSE Momentum v4.3 — Volume Agent
RVOL ratio + volume dry-up detection + delivery %. Universe-aware multiplier.
Score: 0-12 pts.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class VolumeAgent:
    def __init__(self, df: pd.DataFrame, delivery_pct: float = 0.0,
                 universe: str = "LARGE"):
        self.df           = df
        self.delivery_pct = delivery_pct
        self.universe     = universe
        self._score       = 0
        self._rvol        = 0.0
        self._compute()

    def _compute(self):
        df = self.df
        if len(df) < 20:
            return
        vol = df["Volume"].squeeze().to_numpy(dtype=float)
        avg20 = float(np.mean(vol[-20:])) if np.mean(vol[-20:]) > 0 else 1
        self._rvol = float(vol[-1]) / avg20

        pts = 0

        # RVOL scoring
        if self._rvol >= 3.0:   pts += 5
        elif self._rvol >= 2.0: pts += 4
        elif self._rvol >= 1.5: pts += 3
        elif self._rvol >= 1.2: pts += 2
        elif self._rvol >= 0.8: pts += 1

        # Volume dry-up (accumulation signal) — 5 bars below 80% avg
        vol_dry = float(np.mean(vol[-5:])) < 0.8 * avg20
        if vol_dry:
            pts += 2

        # Up-volume quality — last bar is an up bar on high volume
        if len(df) >= 2:
            last_close = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2])
            if last_close > prev_close and self._rvol >= 1.5:
                pts += 1

        # Delivery %
        if self.delivery_pct >= 60:   pts += 4
        elif self.delivery_pct >= 45: pts += 3
        elif self.delivery_pct >= 30: pts += 2
        elif self.delivery_pct >= 15: pts += 1

        # Universe multiplier
        mult = {"LARGE": 1.0, "MID": 1.1, "SMALL": 1.2}.get(self.universe, 1.0)
        self._score = min(int(pts * mult), 12)

    def score(self) -> int:
        return self._score

    def get_rvol(self) -> float:
        return round(self._rvol, 2)
