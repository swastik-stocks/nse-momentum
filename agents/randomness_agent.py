"""
NSE Momentum vNext - Randomness Agent   [PROPOSED -- NOT WIRED INTO orchestrator.py]

STATUS: prototype only. Per the same evidence-first discipline documented in
pattern_agent.py's DEFAULT_WEIGHTS history and agents/weekly_trend_agent.py's
header -- this must NOT be added to the live gate chain (G1-G7) or pattern/
momentum scoring until it has been run through the same validation chain
(pipeline_replay.py -> monte_carlo_significance.py -> ideally
split_period_significance.py) that every other live-scoring input has
already passed. This file defines the agent's INTERFACE and computation only.

WHY THIS EXISTS
    Factor-library item 2 (Desai 2014, "Quantifying Randomness"). Nothing in
    the current gate chain distinguishes a stock whose recent trend is
    statistically distinguishable from a random walk from one whose "trend"
    is a run of noise that happens to look directional over a short window.
    Desai's technique is the Wald-Wolfowitz runs test -- decades-old,
    standard, not itself in question -- applied here to daily
    up/down-close sequences, computed from the SAME daily OHLCV every other
    agent already receives. No new data source needed.

    Framed in the implementation plan as a PRE-FILTER ahead of pattern/
    momentum scoring, not an additive score component: the question this
    answers is "does this stock's apparent trend have a statistical basis
    at all," which is a different kind of judgment than "how good is the
    trend" (that's RSAgent/PatternAgent's job). A gate is the natural fit
    for that question, but see passes_gate()'s docstring for why
    score_bonus() ALSO exists -- same dual-interface rationale as
    WeeklyTrendAgent, so the validation run can test hard-gate vs.
    soft-bonus integration on real historical outcomes rather than
    committing to one before there's evidence either way.

RUNS TEST (Wald-Wolfowitz), applied to a sequence of daily price directions
    Given n1 up-days and n2 down-days in the lookback window, and R the
    actual number of runs (maximal streaks of the same direction):
        E[R]   = 2*n1*n2 / (n1+n2) + 1
        Var[R] = 2*n1*n2*(2*n1*n2 - n1 - n2) / ((n1+n2)^2 * (n1+n2-1))
        Z      = (R - E[R]) / sqrt(Var[R])
    Z < 0 (fewer runs than a random walk would produce) means price moves
    are clustering into longer directional streaks than chance predicts --
    i.e. genuine trending behavior. Z > 0 (more runs than expected) means
    the series is choppier/more mean-reverting than a random walk -- NOT
    the kind of "trend" a momentum/breakout signal should be trusted on.
    Z near 0 is statistically indistinguishable from a random walk either
    way.

    run_ratio = R / E[R] is the same information as Z but scale-free and
    easier to eyeball (1.0 = exactly random-walk-like, <1.0 = trending,
    >1.0 = choppier than random) -- exposed alongside Z for that reason.

INTERFACE
    Deliberately mirrors WeeklyTrendAgent: plain __init__(df), internal
    _compute(), and BOTH a binary gate (.passes_gate()) and a small
    graduated bonus (.score_bonus()), fails open (True / 0.0) when there
    isn't enough history to trust the test.

USAGE (once validated and wired in):
    ra = RandomnessAgent(df)
    if ra.get_bars_available() >= MIN_BARS_FOR_SIGNAL:
        # as a pre-filter (the plan's stated integration style):
        if not ra.passes_gate():
            continue   # apparent trend has no statistical basis -- skip
        # OR as a graduated refinement, if validation favors that instead:
        r.bonus_score += ra.score_bonus()
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Minimum daily bars required before trusting the runs test at all -- a
# runs test on a handful of days has almost no statistical power. 60 bars
# (~3 months) mirrors this repo's other agents' preference for multi-month
# minimums (see weekly_trend_agent.py's MIN_WEEKS_FOR_SIGNAL) scaled to
# daily-bar terms.
MIN_BARS_FOR_SIGNAL = 60

# Z-score threshold for passes_gate(): more negative than this means the
# run-clustering is unlikely to be random-walk noise (roughly a one-sided
# ~84th percentile cutoff at Z=-1.0). Not tuned against outcomes yet --
# this is a starting point for the validation run to confirm or replace.
GATE_Z_THRESHOLD = -1.0


class RandomnessAgent:
    """
    Runs a Wald-Wolfowitz runs test on the sign of daily closing-price
    changes over `lookback` bars, and answers: does this stock's recent
    directional persistence look like a real trend, or like noise that
    happens to have drifted?
    """

    def __init__(self, daily_df: pd.DataFrame, lookback: int = 60):
        self.daily_df = daily_df
        self.lookback = lookback

        self._bars_available = 0
        self._z              = 0.0
        self._run_ratio       = 1.0
        self._runs            = 0
        self._expected_runs   = 0.0

        self._compute()

    # ── computation ─────────────────────────────────────────────────────

    def _compute(self):
        if self.daily_df is None or self.daily_df.empty:
            return
        if "Close" not in self.daily_df.columns:
            return

        close = self.daily_df["Close"].squeeze().to_numpy(dtype=float)
        self._bars_available = len(close)

        if self._bars_available < max(MIN_BARS_FOR_SIGNAL, self.lookback + 1):
            # Not enough history to trust a runs test -- stay neutral
            # (run_ratio=1.0, z=0.0) rather than testing off too few bars.
            return

        window = close[-(self.lookback + 1):]
        diffs  = np.diff(window)
        # Zero-change days carry no directional information for a runs
        # test -- dropping them (rather than assigning an arbitrary sign)
        # is the standard treatment for ties in the Wald-Wolfowitz test.
        signs = np.sign(diffs)
        signs = signs[signs != 0]

        n1 = int(np.sum(signs > 0))   # up days
        n2 = int(np.sum(signs < 0))   # down days
        n  = n1 + n2

        if n1 == 0 or n2 == 0 or n < 10:
            # All-one-direction or too few non-tied days -- runs test is
            # undefined/meaningless here. Stay neutral.
            return

        runs = 1 + int(np.sum(signs[1:] != signs[:-1]))

        expected = 2.0 * n1 * n2 / n + 1.0
        variance = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n)) / (n * n * (n - 1))

        self._runs          = runs
        self._expected_runs = expected
        self._run_ratio      = round(runs / expected, 3) if expected > 0 else 1.0

        if variance > 0:
            self._z = round((runs - expected) / np.sqrt(variance), 3)

    # ── public interface ────────────────────────────────────────────────

    def passes_gate(self) -> bool:
        """
        [VALIDATION-DIAGNOSTIC -- integration style not yet confirmed.]

        True (pass) when the trend looks statistically real (Z below
        GATE_Z_THRESHOLD) OR when there isn't enough history to test at
        all -- a data-quality gap should not silently reject a candidate
        that every other agent is still willing to score. Fails open, same
        convention as WeeklyTrendAgent.passes_gate().
        """
        if self._bars_available < MIN_BARS_FOR_SIGNAL:
            return True
        return bool(self._z <= GATE_Z_THRESHOLD)

    def score_bonus(self) -> float:
        """
        Continuous alternative to the hard gate above, for the validation
        run to compare against passes_gate() on real historical outcomes.
        More negative Z (more trend-like clustering) maps to a larger
        positive bonus; positive Z (choppier than random) maps to a small
        penalty. Clamped to the same +/-2..+3 range WeeklyTrendAgent's
        score_bonus() uses, for the same reason -- this should refine the
        total score, not dominate it.
        """
        if self._bars_available < MIN_BARS_FOR_SIGNAL:
            return 0.0
        raw = -1.0 * self._z
        return round(float(np.clip(raw, -2.0, 3.0)), 2)

    def get_z_score(self) -> float:
        return self._z

    def get_run_ratio(self) -> float:
        return self._run_ratio

    def get_bars_available(self) -> int:
        return self._bars_available
