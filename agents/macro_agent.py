"""
NSE Momentum v5.0 - MacroAgent
Market-level agent. Produces daily macro_state from free data.
States: SUPPORTIVE / MIXED / HOSTILE
Sources: India VIX (yfinance), FII/DII (NSE bhavcopy summary), breadth
"""

import logging
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


class MacroAgent:
    def __init__(self, vix: float, breadth_score: int,
                 fii_flow: float = 0.0, adv_dec_ratio: float = 1.0):
        self.vix          = vix
        self.breadth      = breadth_score
        self.fii_flow     = fii_flow        # positive = buying, negative = selling
        self.adv_dec      = adv_dec_ratio
        self._state       = "MIXED"
        self._score       = 3
        self._compute()

    def _compute(self):
        score = 0

        # VIX contribution
        if self.vix < 13:
            score += 3
        elif self.vix < 16:
            score += 2
        elif self.vix < 20:
            score += 1
        else:
            score -= 1   # VIX > 20 = fear

        # Breadth contribution (0-10 scale)
        if self.breadth >= 7:
            score += 2
        elif self.breadth >= 5:
            score += 1
        elif self.breadth <= 3:
            score -= 1

        # FII flow contribution
        if self.fii_flow > 500:    # > 500 Cr buying
            score += 1
        elif self.fii_flow < -500: # > 500 Cr selling
            score -= 1

        # A/D ratio
        if self.adv_dec >= 1.5:
            score += 1
        elif self.adv_dec < 0.8:
            score -= 1

        # State assignment
        if score >= 5:
            self._state = "SUPPORTIVE"
            self._score = 6
        elif score >= 2:
            self._state = "MIXED"
            self._score = 3
        else:
            self._state = "HOSTILE"
            self._score = 0

    def get_state(self) -> str:
        return self._state

    def get_score(self) -> int:
        """Returns 0, 3, or 6 pts for scoring."""
        return self._score

    def get_t1_cap(self, regime: str) -> int:
        """Max T1 picks based on regime x macro."""
        caps = {
            "A": {"SUPPORTIVE": 15, "MIXED": 10, "HOSTILE": 5},
            "B": {"SUPPORTIVE": 12, "MIXED": 8,  "HOSTILE": 4},
            "C": {"SUPPORTIVE": 8,  "MIXED": 5,  "HOSTILE": 3},
            "D": {"SUPPORTIVE": 5,  "MIXED": 3,  "HOSTILE": 0},
            "E": {"SUPPORTIVE": 3,  "MIXED": 0,  "HOSTILE": 0},
        }
        return caps.get(regime, caps["B"]).get(self._state, 5)
