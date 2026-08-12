"""
NSE Momentum vNext - 52-Week High Proximity Agent   [PROPOSED -- NOT WIRED
INTO orchestrator.py]

STATUS: prototype only. Per the same evidence-first discipline as every
other prototype in this codebase (agents/weekly_trend_agent.py,
agents/randomness_agent.py) -- this must NOT be added to live scoring
until it clears the same validation chain (pipeline-style replay ->
permutation significance test -> out-of-sample half-split check) that
factor-library item 1 (volatility-adjusted momentum, now live in
agents/rs_agent.py) already passed.

WHY THIS EXISTS
    Factor-library item 3 (Raju 2023, "52-Week High Effect" + Anonymous
    2024, mid/small-cap segmentation). George & Hwang's anchoring effect
    is one of the most repeatedly replicated cross-sectional anomalies in
    the literature: stocks trading near their own 52-week high tend to
    keep outperforming, because investors underreact to good news once a
    stock nears a salient reference point (the 52-week high itself acts
    as an anchor). This is a DIFFERENT signal from what already exists in
    this codebase:
      - agents/near_breakout.py measures distance to a PATTERN breakout
        level (from PatternAgent's own structural analysis), not the
        literal 52-week high.
      - agents/rs_agent.py measures return relative to the Nifty
        benchmark, not proximity to a stock's own price ceiling.
    Raju's own NSE backtest found this ranking signal to be the highest
    standalone-alpha item in the factor library reviewed for this project
    (see FACTOR_LIBRARY_IMPLEMENTATION_PLAN.md, Tier 2).

DESIGN CHOICE: absolute proximity, not cross-sectional rank
    Raju's original paper builds a cross-sectional rank (price/52wk-high
    percentile across the universe) for portfolio construction. This
    codebase's per-stock agents (WeeklyTrendAgent, InstitutionalProxyAgent)
    instead compute a self-contained, stock-specific signal that doesn't
    need a universe-wide precompute step -- "how close is THIS stock to
    ITS OWN 52-week high" is inherently a single-stock question, unlike RS
    (which is relative to a benchmark by construction). This mirrors the
    existing architecture rather than introducing a second cross-sectional
    ranking pipeline alongside RSAgent's. The validation harness should
    confirm this design choice holds up, not just the underlying anomaly.

CAP-TIER WEIGHTING (Anonymous 2024)
    That paper found the anchoring effect concentrated in mid/small-cap
    names, weaker in large-cap where fundamentals dominate more of the
    return. TIER_MULTIPLIER below encodes that as a starting hypothesis
    for the validation run to confirm or recalibrate -- not tuned against
    real outcomes yet, same caveat this file's sibling prototypes carry
    for their own starting thresholds.

INTERFACE
    Mirrors WeeklyTrendAgent / RandomnessAgent: plain __init__(df,
    universe), internal _compute(), dual .passes_gate() / .score_bonus()
    interface, fails open on insufficient history.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Trading days in a year -- the literal "52-week" window from the source
# papers, not this repo's shorter 4w/12w/26w swing-trading windows (RS
# agent) -- proximity-to-high is a full-year reference point by definition.
LOOKBACK_DAYS = 252

# Minimum bars required before trusting a 52-week-high measurement at all.
# Deliberately short of the full 252-day window (a stock with 180 days of
# listed history still HAS a meaningful "highest price it's traded at" --
# just over a shorter window than a full year) -- MIN_BARS_FOR_SIGNAL is
# about not trusting a handful of days, not about requiring the full year.
MIN_BARS_FOR_SIGNAL = 150

# Gate threshold: within 15% of the 52-week high. This specific cutoff is
# a common industry heuristic (e.g. IBD-style screens) roughly consistent
# with the anchoring literature's finding that the effect is strongest
# very near the high and fades further below it -- a starting point for
# the validation run, not a backtested threshold yet.
GATE_PROXIMITY_PCT = 85.0

# Anonymous (2024): anchoring effect concentrated in mid/small-cap.
# Starting hypothesis, not yet tuned against real outcomes.
TIER_MULTIPLIER = {"LARGE": 1.0, "MID": 1.3, "SMALL": 1.5}


class FiftyTwoWeekHighAgent:
    """
    Answers: how close is this stock trading to its own 52-week high right
    now, and (per Anonymous 2024) is that proximity worth more in this
    stock's cap tier?
    """

    def __init__(self, daily_df: pd.DataFrame, universe: str = "LARGE"):
        self.daily_df = daily_df
        self.universe = universe

        self._bars_available = 0
        self._fifty_two_wk_high = 0.0
        self._last_close = 0.0
        self._proximity_pct = 0.0   # last_close / 52wk_high * 100

        self._compute()

    def _compute(self):
        if self.daily_df is None or self.daily_df.empty:
            return
        if "Close" not in self.daily_df.columns or "High" not in self.daily_df.columns:
            return

        close = self.daily_df["Close"].squeeze().to_numpy(dtype=float)
        high  = self.daily_df["High"].squeeze().to_numpy(dtype=float)
        self._bars_available = len(close)

        if self._bars_available < MIN_BARS_FOR_SIGNAL:
            return

        # 52-week high computed from the window EXCLUDING today's bar, so a
        # fresh breakout to a new high today correctly scores as proximity
        # > 100%, not clipped to exactly 100% by definition -- a stock
        # making a brand-new high is the strongest form of this signal, not
        # a boundary case.
        window = high[-min(LOOKBACK_DAYS, self._bars_available):-1]
        if len(window) == 0:
            return

        self._fifty_two_wk_high = float(np.max(window))
        self._last_close = float(close[-1])

        if self._fifty_two_wk_high > 0:
            self._proximity_pct = round(
                self._last_close / self._fifty_two_wk_high * 100, 2
            )

    # ── public interface ────────────────────────────────────────────────

    def passes_gate(self) -> bool:
        """
        [VALIDATION-DIAGNOSTIC -- integration style not yet confirmed.]

        True (pass) when trading within GATE_PROXIMITY_PCT of the 52-week
        high, OR when there isn't enough history to test at all -- fails
        open, same convention as every other prototype gate in this repo.
        """
        if self._bars_available < MIN_BARS_FOR_SIGNAL:
            return True
        return self._proximity_pct >= GATE_PROXIMITY_PCT

    def score_bonus(self) -> float:
        """
        Continuous alternative to the hard gate, for the validation run to
        compare against passes_gate() on real historical outcomes.
        Deliberately continuous (not discrete buckets) around the 85%
        proximity reference point, then scaled by TIER_MULTIPLIER per
        Anonymous (2024)'s cap-tier finding. Clamped to the same +/-2..+3
        range every other prototype bonus in this repo uses.
        """
        if self._bars_available < MIN_BARS_FOR_SIGNAL or self._fifty_two_wk_high <= 0:
            return 0.0

        raw = (self._proximity_pct - GATE_PROXIMITY_PCT) * 0.15
        tier_mult = TIER_MULTIPLIER.get(self.universe, 1.0)
        return round(float(np.clip(raw * tier_mult, -2.0, 3.0)), 2)

    def get_proximity_pct(self) -> float:
        return self._proximity_pct

    def get_fifty_two_week_high(self) -> float:
        return round(self._fifty_two_wk_high, 2)

    def get_bars_available(self) -> int:
        return self._bars_available
