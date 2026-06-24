"""
NSE Momentum v5.2 — VCPContractionGate (complete replacement)

This file did not exist in v5.1 as a standalone (the logic was in orchestrator).
Now it is a proper gate module imported by orchestrator.py via:
    from agents.vcp_gate import VCPContractionGate

WHAT THIS GATE DOES:
  - Computes W4: the width (high-low range as %) of the FINAL contraction leg
    (most recent 5-bar window). This is the tightest squeeze of the VCP.
  - Hard-rejects any stock where W4 > W4_HARD_REJECT (8%). At 8%+ width
    the stock is not in a VCP — it's just noisy consolidation.
  - Applies a score penalty for W4 between W4_PENALTY_THRESHOLD (4%) and
    the hard reject, proportional to how loose the contraction is.
  - Checks that each successive contraction leg is tighter than the previous
    (the "contracting" property of a real VCP).
  - All thresholds are configurable constants at module level.

ORCHESTRATOR CALL (unchanged from v5.0):
    vcpg = VCPContractionGate(df=df)
    vcp  = vcpg.check()
    r.vcp_w4_pct    = vcp["w4_pct"]
    r.vcp_contracting = vcp["contracting"]
    r.vcp_penalty   = vcp["penalty"]
    if vcp["hard_reject"]:
        r.rejected = True; r.reject_reason = vcp["fail_reason"]
        return r
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
W4_HARD_REJECT       = 8.0   # % — hard gate; no passes above this
W4_PENALTY_THRESHOLD = 4.0   # % — above this, score penalty starts
MAX_PENALTY          = 8     # score points deducted at the worst end of range
# ─────────────────────────────────────────────────────────────────────────────


class VCPContractionGate:
    """
    Volatility Contraction Pattern gate.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV dataframe. Expects columns: High, Low, Close, Volume.
    windows : list[int]
        Bar lengths for W1, W2, W3, W4 (outermost to innermost).
        Default [20, 10, 5] produces three windows; W4 is the last window.
    """

    def __init__(self, df: pd.DataFrame,
                 windows: list = None):
        self.df      = df.copy()
        self.windows = windows or [20, 10, 5]

    def _range_pct(self, bars: int) -> float:
        """Return (high - low) / high for the last `bars` bars, as %."""
        df = self.df
        if len(df) < bars:
            return 100.0   # can't compute — treat as very wide
        seg_h = float(df["High"].squeeze().iloc[-bars:].max())
        seg_l = float(df["Low"].squeeze().iloc[-bars:].min())
        if seg_h <= 0:
            return 100.0
        return (seg_h - seg_l) / seg_h * 100.0

    def check(self) -> dict:
        """
        Returns
        -------
        dict with keys:
            w4_pct      float   — width of innermost leg (%)
            w_ranges    list    — [w1%, w2%, w3%] outermost → innermost
            contracting bool    — True if each leg is tighter than previous
            hard_reject bool    — True if w4_pct > W4_HARD_REJECT
            penalty     int     — score deduction (0 if w4 ≤ W4_PENALTY_THRESHOLD)
            fail_reason str     — human-readable reason if hard_reject
        """
        result = {
            "w4_pct":      0.0,
            "w_ranges":    [],
            "contracting": False,
            "hard_reject": False,
            "penalty":     0,
            "fail_reason": "",
        }

        if len(self.df) < max(self.windows):
            # Not enough data — pass silently (liquidity gate already checked len)
            result["contracting"] = True
            return result

        # Compute range % for each window (outermost → innermost)
        ranges = [self._range_pct(w) for w in self.windows]
        result["w_ranges"] = [round(r, 2) for r in ranges]

        # W4 is the innermost (tightest expected) window
        w4_pct = ranges[-1]
        result["w4_pct"] = round(w4_pct, 2)

        # Contracting check: each leg should be tighter than the previous
        result["contracting"] = all(
            ranges[i] > ranges[i + 1]
            for i in range(len(ranges) - 1)
        )

        # Hard reject
        if w4_pct > W4_HARD_REJECT:
            result["hard_reject"] = True
            result["fail_reason"] = (
                f"VCPGate: W4 {w4_pct:.1f}% > {W4_HARD_REJECT}% hard reject "
                f"— base not contracting to VCP standard "
                f"(contracting={result['contracting']})"
            )
            log.debug(result["fail_reason"])
            return result

        # Penalty (proportional between threshold and hard reject)
        if w4_pct > W4_PENALTY_THRESHOLD:
            ratio = (w4_pct - W4_PENALTY_THRESHOLD) / (W4_HARD_REJECT - W4_PENALTY_THRESHOLD)
            result["penalty"] = int(round(ratio * MAX_PENALTY))

        # Warn if not truly contracting (but don't hard reject — could be a
        # tight flat base rather than a textbook VCP)
        if not result["contracting"]:
            log.debug(
                "VCPGate: ranges not fully contracting %s — minor penalty added",
                result["w_ranges"]
            )
            result["penalty"] = min(result["penalty"] + 2, MAX_PENALTY)

        return result

    def summary(self, result: dict) -> str:
        if result["hard_reject"]:
            return f"REJECT W4={result['w4_pct']:.1f}%  {result['fail_reason']}"
        status = "PASS" if result["contracting"] else "PASS(non-contracting)"
        return (
            f"{status} W4={result['w4_pct']:.1f}%  "
            f"ranges={result['w_ranges']}  penalty={result['penalty']}"
        )
