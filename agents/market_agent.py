"""
NSE Momentum v4.3 — Market Regime Agent
5-regime A-E. Breadth-first, primary signal.
Regime is driven by: breadth score + price vs EMAs + VIX
Breadth is given equal weight to price action — avoids 200 EMA lag problem.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REGIME_CONFIG = {
    "A": {"name": "Strong Bull",  "score": 5,  "penalty": 0,   "color": "#00E676"},
    "B": {"name": "Bull",         "score": 4,  "penalty": 0,   "color": "#00D4AA"},
    "C": {"name": "Range Bound",  "score": 2,  "penalty": -5,  "color": "#FFB300"},
    "D": {"name": "Correction",   "score": 0,  "penalty": -12, "color": "#FF8C00"},
    "E": {"name": "Bear Market",  "score": 0,  "penalty": -25, "color": "#FF5252"},
}


class MarketAgent:
    def __init__(self, data_dict: dict):
        self.data         = data_dict
        self.regime       = "C"
        self.regime_name  = "Range Bound"
        self.market_score = 2
        self.penalty      = -5
        self._compute()

    def _compute(self):
        nifty   = self.data.get("nifty50_data",  pd.DataFrame())
        bank    = self.data.get("banknifty_data", pd.DataFrame())
        vix     = self.data.get("vix", 15.0)
        breadth = self.data.get("breadth_score", 5)

        def _ema(v, span):
            alpha = 2 / (span + 1)
            e = np.zeros(len(v)); e[0] = v[0]
            for i in range(1, len(v)):
                e[i] = alpha * v[i] + (1 - alpha) * e[i - 1]
            return e

        # ── Primary signal: Breadth ──────────────────────────────────────
        # A/D 4.93, 63% above 50EMA, breadth 7/10 = Bull conditions
        # Use breadth as the primary regime signal, price EMA as confirmation
        if breadth >= 8:
            breadth_signal = "A"   # Strong Bull internals
        elif breadth >= 6:
            breadth_signal = "B"   # Bull internals
        elif breadth >= 4:
            breadth_signal = "C"   # Neutral internals
        elif breadth >= 2:
            breadth_signal = "D"   # Weak internals
        else:
            breadth_signal = "E"   # Bear internals

        # ── Secondary signal: Price vs 50 EMA ───────────────────────────
        price_signal = "C"   # default
        if not nifty.empty and len(nifty) >= 60:
            c = nifty["Close"].squeeze().to_numpy(dtype=float)
            ema50  = _ema(c, 50)[-1]
            price  = c[-1]
            above_50 = price > ema50

            # BankNifty confirmation
            bank_ok = True   # default to true if data unavailable
            if not bank.empty and len(bank) >= 50:
                bc = bank["Close"].squeeze().to_numpy(dtype=float)
                b50 = _ema(bc, 50)[-1]
                bank_ok = float(bc[-1]) > b50

            if above_50 and bank_ok and vix < 15:
                price_signal = "A"
            elif above_50 and vix < 20:
                price_signal = "B"
            elif above_50:
                price_signal = "C"
            else:
                price_signal = "D"

        # ── Combine: take the BETTER of breadth and price signal ─────────
        # This prevents the 200 EMA lag from over-penalising a recovering market
        regime_order = ["A", "B", "C", "D", "E"]
        b_idx = regime_order.index(breadth_signal)
        p_idx = regime_order.index(price_signal)
        # Take the better (lower index = better) of the two signals
        combined_idx = min(b_idx, p_idx)
        # But cap improvement: can't be more than 1 level better than breadth alone
        base_idx = max(b_idx - 1, combined_idx)

        self.regime      = regime_order[base_idx]
        cfg              = REGIME_CONFIG[self.regime]
        self.regime_name = cfg["name"]
        self.market_score= cfg["score"]
        self.penalty     = cfg["penalty"]

        log.debug(f"Regime: breadth_signal={breadth_signal}({breadth}/10) "
                  f"price_signal={price_signal} VIX={vix:.1f} "
                  f"final={self.regime}")

    def score(self) -> int:
        return self.market_score

    def get_penalty(self) -> int:
        return self.penalty

    def get_regime(self) -> str:
        return self.regime

    def get_regime_name(self) -> str:
        return self.regime_name
