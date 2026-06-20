"""
NSE Momentum v5 â€” VCPContractionGate
======================================
Validates that the current price structure is tight enough
to support a <=2% stop-loss entry.

Logic (modelled after Minervini VCP):
  - Identify up to 4 consolidation patches (W1..W4)
  - Each patch = a local high-to-low swing within recent price action
  - Rules:
      W1 > W2 > W3 (if exists)  â€” volatility contracting
      W4 (most recent) width <= 4%  â€” fully compressed
  - If W4 is 4-8%: penalty (-5 pts) but not hard reject
  - If W4 > 8%:    hard reject (structure too loose for 2% stop)

Returns a dict consumed by orchestrator to either reject or penalise.

Usage in orchestrator.py:
    from agents.vcp_gate import VCPContractionGate
    vcpg = VCPContractionGate(df=df)
    vcp_result = vcpg.check()
    if vcp_result["hard_reject"]:
        # reject before scoring
    else:
        score_penalty += vcp_result["penalty"]
        vcp_w4 = vcp_result["w4_pct"]
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)

# â”€â”€ Thresholds â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
W4_PASS_PCT        = 4.0   # W4 <= 4% â†’ clean, no penalty
W4_PENALTY_PCT     = 10.0   # 4% < W4 <= 8% â†’ -5 pts penalty
W4_HARD_REJECT_PCT = 15.0   # W4 > 8% â†’ hard reject
SCORE_PENALTY      = 5     # Penalty points if 4% < W4 <= 8%

# Look back this many bars to find the 4 contraction patches
LOOKBACK_BARS = 60


class VCPContractionGate:
    """
    Checks whether the stock's recent price action shows
    volatility contraction consistent with a tight-stop entry.

    Parameters
    ----------
    df : pd.DataFrame â€” OHLCV with columns High, Low, Close
                        (at least 40 bars recommended)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # â”€â”€ Public interface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def check(self) -> dict:
        """
        Returns:
          hard_reject     : bool   â€” True = stop processing immediately
          penalty         : int    â€” score penalty (0 or 5)
          w4_pct          : float  â€” width of most recent contraction patch
          contracting     : bool   â€” True if W1 > W2 > W3 (where measurable)
          fail_reason     : str    â€” populated on hard_reject
          note            : str    â€” descriptive for logging/email
        """
        result = {
            "hard_reject": False,
            "penalty":     0,
            "w4_pct":      0.0,
            "contracting": False,
            "fail_reason": "",
            "note":        "",
        }

        df = self.df
        if df is None or len(df) < 20:
            result["note"] = "VCPGate: insufficient data â€” skipping check"
            return result

        # â”€â”€ Extract recent price window â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        window = df.tail(LOOKBACK_BARS).copy()
        highs  = window["High"].values
        lows   = window["Low"].values

        # â”€â”€ Identify contraction patches â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        patches = self._find_patches(highs, lows)

        if not patches:
            result["note"] = "VCPGate: no contraction patches found â€” skipping"
            return result

        # â”€â”€ Assess most recent patch (W4 or whatever is last) â”€â”€â”€â”€â”€â”€â”€â”€
        w4_pct = patches[-1]
        result["w4_pct"] = round(w4_pct, 2)

        # â”€â”€ Check contraction sequence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        result["contracting"] = self._is_contracting(patches)

        # â”€â”€ Apply rules â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if w4_pct > W4_HARD_REJECT_PCT:
            result["hard_reject"] = True
            result["fail_reason"] = (
                f"VCPGate: W4={w4_pct:.1f}% > hard limit {W4_HARD_REJECT_PCT}% "
                f"â€” structure too loose for <=2% stop"
            )
            result["note"] = result["fail_reason"]
            return result

        if w4_pct > W4_PASS_PCT:
            result["penalty"] = SCORE_PENALTY
            result["note"] = (
                f"VCPGate: W4={w4_pct:.1f}% (4â€“8% range) "
                f"â€” penalty -{SCORE_PENALTY} pts applied"
            )
            return result

        # W4 <= 4% â€” clean pass
        contracting_str = "contracting" if result["contracting"] else "flat"
        result["note"] = (
            f"VCPGate: PASS | W4={w4_pct:.1f}% | "
            f"sequence {contracting_str} | "
            f"patches={[round(p,1) for p in patches]}"
        )
        return result

    # â”€â”€ Internal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _find_patches(self, highs: np.ndarray, lows: np.ndarray) -> list:
        """
        Divide the lookback window into up to 4 equal segments.
        Each segment's width = (max_high - min_low) / min_low * 100.
        Returns list of up to 4 width percentages, oldest â†’ newest.
        """
        n = len(highs)
        if n < 8:
            return []

        # Split into up to 4 segments of roughly equal size
        num_segments = min(4, n // 8)
        segment_size = n // num_segments

        patches = []
        for i in range(num_segments):
            start = i * segment_size
            end   = start + segment_size if i < num_segments - 1 else n
            seg_highs = highs[start:end]
            seg_lows  = lows[start:end]
            if len(seg_highs) == 0:
                continue
            seg_max = float(np.max(seg_highs))
            seg_min = float(np.min(seg_lows))
            if seg_min <= 0:
                continue
            width_pct = (seg_max - seg_min) / seg_min * 100.0
            patches.append(width_pct)

        return patches

    def _is_contracting(self, patches: list) -> bool:
        """
        Returns True if the patch widths are broadly decreasing.
        Allows one exception (non-monotone step) to handle noise.
        """
        if len(patches) < 2:
            return True   # insufficient data â€” assume OK
        violations = 0
        for i in range(1, len(patches)):
            if patches[i] >= patches[i - 1]:
                violations += 1
        return violations <= 1

