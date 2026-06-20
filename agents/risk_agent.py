"""
NSE Momentum v5.1 - Risk Agent
Entry: breakout zone.
Stop:  Tightest of (EMA21 | 10-day low | ATR), capped at universe max.
T1:    Nearest resistance, minimum reward floor per universe.
T2:    1.618x Fibonacci extension — always enforced > T1 * 1.05.

Fixes in v5.1:
  FIX 1: T2 now enforced to always be > T1 * 1.05 (was sometimes < T1)
  FIX 2: T2 must be > entry (sanity check)
  FIX 3: entry_low / entry_high always sorted correctly

Stop/reward caps aligned with AsymmetryGate thresholds:
  LARGE: stop cap 3%,  T1 floor 6%
  MID:   stop cap 4%,  T1 floor 8%
  SMALL: stop cap 5%,  T1 floor 10%
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

STOP_CAP     = {"LARGE": 0.03, "MID": 0.04, "SMALL": 0.05}
REWARD_FLOOR = {"LARGE": 1.06, "MID": 1.08, "SMALL": 1.10}
MIN_RRR      = {"LARGE": 1.5,  "MID": 1.8,  "SMALL": 2.0}


class RiskAgent:
    def __init__(self, df: pd.DataFrame, breakout_level: float,
                 entry_low: float = 0.0, entry_high: float = 0.0,
                 universe: str = "LARGE"):
        self.df         = df
        self.bo         = breakout_level
        self.universe   = universe
        # FIX 3: always sort entry_low < entry_high regardless of what PatternAgent passes
        raw_lo = min(entry_low, entry_high) if (entry_low > 0 and entry_high > 0) else entry_low
        raw_hi = max(entry_low, entry_high) if (entry_low > 0 and entry_high > 0) else entry_high
        self.entry_low  = raw_lo
        self.entry_high = raw_hi

        self.entry    = 0.0
        self.stop     = 0.0
        self.target1  = 0.0
        self.target2  = 0.0
        self.rrr      = 0.0
        self.stop_pct = 0.0
        self.gain_pct = 0.0
        self._passes  = False
        self._reason  = ""
        self._compute()

    def _compute(self):
        df = self.df
        if df.empty or len(df) < 14 or self.bo <= 0:
            self._reason = "No valid breakout level"
            return

        high  = df["High"].squeeze().to_numpy(dtype=float)
        low   = df["Low"].squeeze().to_numpy(dtype=float)
        close = df["Close"].squeeze().to_numpy(dtype=float)
        price = close[-1]

        atr = self._atr(high, low, close, 14)
        if atr <= 0:
            self._reason = "ATR calculation failed"
            return

        # Entry: midpoint of sorted band
        if self.entry_high > 0 and self.entry_low > 0:
            entry = (self.entry_low + self.entry_high) / 2
        elif self.entry_high > 0:
            entry = self.entry_high
        else:
            entry = self.bo * 1.002
        if entry <= 0:
            entry = price

        # Stop: three candidates, pick tightest valid one
        ema21    = float(pd.Series(close).ewm(span=21, adjust=False).mean().iloc[-1])
        stop_ema = ema21 * 0.993

        lookback_stop = min(len(low), 10)
        stop_10d = float(np.min(low[-lookback_stop:])) * 0.997

        atr_mult = {"LARGE": 1.5, "MID": 1.8, "SMALL": 2.0}.get(self.universe, 1.5)
        stop_atr = entry - atr_mult * atr

        candidates = [s for s in [stop_ema, stop_10d, stop_atr] if 0 < s < entry]
        if not candidates:
            self._reason = "No valid stop level below entry"
            return
        stop = max(candidates)

        # Cap stop width
        cap  = STOP_CAP.get(self.universe, 0.03)
        stop = min(stop, entry * (1 - cap))
        stop = max(stop, 0.01)

        # Targets
        lookback   = min(len(low), 60)
        base_low   = float(np.min(low[-lookback:]))
        pat_height = max(self.bo - base_low, atr * 2.5)

        res_3m  = float(np.max(high[-65:]))
        res_6m  = float(np.max(high[-130:])) if len(high) >= 130 else res_3m
        res_12m = float(np.max(high[-252:])) if len(high) >= 252 else res_6m

        raw_t1 = entry + pat_height
        resistance_levels = sorted(
            [r for r in [res_3m, res_6m, res_12m] if r > entry * 1.03]
        )

        floor = REWARD_FLOOR.get(self.universe, 1.06)
        if resistance_levels:
            target1 = min(raw_t1, resistance_levels[0] * 0.99)
            target1 = max(target1, entry * floor)
        else:
            target1 = max(raw_t1, entry * floor)

        # T2: Fibonacci extension
        target2 = entry + pat_height * 1.618

        # FIX 1: T2 must always be meaningfully above T1
        # If resistance cap pulled T1 up close to T2, push T2 further
        target2 = max(target2, target1 * 1.10)   # T2 >= T1 + 10%

        # FIX 2: Cap T2 at 12-month high * 1.05, but never below T1
        t2_cap  = res_12m * 1.05
        if t2_cap > target1 * 1.05:              # only cap if cap is above T1
            target2 = min(target2, t2_cap)

        # Final sanity: T2 must be > entry
        target2 = max(target2, entry * 1.15)

        # R:R gate
        risk    = max(entry - stop, 0.01)
        reward  = target1 - entry
        rrr     = round(reward / risk, 2) if risk > 0 else 0.0
        min_rrr = MIN_RRR.get(self.universe, 1.5)

        if rrr < min_rrr:
            rrr_t2 = round((target2 - entry) / risk, 2)
            if rrr_t2 >= min_rrr:
                target1 = round(entry + pat_height * 0.75, 2)
                target2 = max(round(entry + pat_height * 1.618, 2), target1 * 1.10)
                rrr = round((target1 - entry) / risk, 2)
                if rrr < min_rrr:
                    self._reason = f"R:R {rrr:.2f}x below min {min_rrr}x"
                    return
            else:
                self._reason = f"R:R {rrr:.2f}x below min {min_rrr}x"
                return

        self.entry    = round(entry, 2)
        self.stop     = round(stop, 2)
        self.target1  = round(target1, 2)
        self.target2  = round(target2, 2)
        self.rrr      = rrr
        self.stop_pct = round((entry - stop) / entry * 100, 1)
        self.gain_pct = round((target1 - entry) / entry * 100, 1)
        self._passes  = True

    def passes(self) -> bool:       return self._passes
    def reject_reason(self) -> str: return self._reason

    @staticmethod
    def _atr(high, low, close, period=14):
        if len(high) < period + 1:
            return float(np.mean(high - low))
        trs = []
        for i in range(1, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            trs.append(tr)
        return float(np.mean(trs[-period:]))
