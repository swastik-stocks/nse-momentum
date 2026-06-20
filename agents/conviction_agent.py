"""
NSE Momentum v4.3 — Conviction Agent
Aggregates all agent scores, applies universe-aware regime penalty,
calibrates confidence % from seed win rates.
"""

import logging

log = logging.getLogger(__name__)

# Seed win rates from 264 training examples (211 StockEdge + 53 PDF reports)
SEED_WIN_RATES = {
    "High Tight Flag":      0.78,
    "Swing High Breakout":  0.72,
    "VCP":                  0.68,
    "3-Weeks-Tight":        0.67,
    "High Base":            0.65,
    "Base Breakout":        0.64,
    "Rounded Base":         0.63,
    "Double Bottom":        0.62,
    "Cup & Handle":         0.61,
    "Ascending Triangle":   0.60,
    "IPO Base":             0.60,
    "Flat Base":            0.58,
    "Volume Expansion":     0.57,
    "52W Momentum":         0.56,
    "Bull Flag":            0.55,
    "Diamond Bottom":       0.54,
    "Symmetrical Triangle": 0.53,
    "Descending Wedge":     0.52,
    "Falling Wedge":        0.50,
}


class ConvictionAgent:
    def __init__(self):
        pass

    def calibrate_confidence(self, pattern: str, total_score: int,
                              universe: str = "LARGE") -> float:
        """
        Confidence % based on seed win rate + score.
        Adjusted downward for MID/SMALL universe.
        """
        base_wr = SEED_WIN_RATES.get(pattern, 0.55)
        # Scale from score: gate (78/80/82) = base_wr, 100 = base_wr + 10%
        gate = {"LARGE": 78, "MID": 80, "SMALL": 82}.get(universe, 78)
        score_bonus = max(0, (total_score - gate) / (100 - gate)) * 0.10

        # Universe liquidity discount
        univ_disc = {"LARGE": 0.0, "MID": -0.03, "SMALL": -0.06}.get(universe, 0.0)

        confidence = (base_wr + score_bonus + univ_disc) * 100
        return round(max(0, min(99, confidence)), 1)
