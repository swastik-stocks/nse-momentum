"""
NSE Momentum vNext - Intraday Volatility-Adjusted Move Agent   [PROPOSED --
NOT WIRED INTO confirm_picks.py or orchestrator.py]

STATUS: prototype only. Same evidence-first discipline as every other
prototype in this codebase — must clear a permutation-significance
validation run (validation/intraday_vol_adjusted_validate.py) before it's
added to the 10:05/12:30/3:15 BTST checkpoint emails.

WHY THIS EXISTS
    classify_btst() (confirm_picks.py) already checks RVOL (elapsed-time
    volume vs. trailing average) as a "is this move backed by real
    activity" signal. RVOL answers "how much volume," not "was today's
    price move statistically large, and was it clean or choppy." Those
    are exactly the two questions raised by a "volume choppy today"
    session: a stock can have high RVOL from violent back-and-forth
    churn, which looks identical to RVOL from a genuine one-directional
    continuation. This agent adds two new, independent reads:

    1. VOL-ADJUSTED MAGNITUDE (Z) — direct intraday analog of
       factor-library item 1 (Shenoy & Vijaykumar 2020, now live in
       agents/rs_agent.py::RSAgent.score()), which validated cleanly
       (p=0.0004) on the principle "excess move / realized volatility
       beats raw move as a momentum signal." Applied here to today's
       move-so-far instead of a multi-week return.

    2. PATH EFFICIENCY (E) — Kaufman's Efficiency Ratio concept (net
       progress / total distance traveled), NOT a reuse of factor-library
       item 2's Randomness/Run-Ratio test, which already failed
       validation in this codebase (see FACTOR_LIBRARY_IMPLEMENTATION_PLAN.md).
       Different math, different question: item 2 asked "is the DAILY
       closing-price sequence over N days distinguishable from a random
       walk"; this asks "of the distance this stock traveled INTRADAY
       today, how much was net directional progress vs. back-and-forth."

    Combined, Z and E separate four cases RVOL alone cannot:
      high Z + high E  -> clean, sizeable continuation (genuine strength)
      high Z + low  E  -> big range, no net progress (the "choppy volume"
                          case this agent exists to catch)
      low  Z + high E  -> quiet orderly grind (mildly supportive)
      low  Z + low  E  -> directionless (neutral)

DATA
    Uses price_history_hourly (7 bars/session: 09:00-15:00 IST) for
    backtesting. A LIVE call at 10:05/12:30/3:15 should be fed whatever
    intraday bars confirm_picks.py's existing get_intraday_high_low() /
    get_rvol() machinery already has access to -- this agent does not
    fetch data itself, it only computes from bars handed to it, same
    separation of concerns as every other agent in this repo.

INTERFACE
    Mirrors WeeklyTrendAgent / RandomnessAgent / FiftyTwoWeekHighAgent:
    dual .passes_gate() / .score_bonus() interface, fails open on
    insufficient bars.
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Need at least 2 bars (day-open + 1 hourly close) to compute a path at
# all -- this is the earliest possible read, matching the 10:05 checkpoint
# (which has the 09:00 and 10:00 bars available).
MIN_BARS_FOR_SIGNAL = 2

# Gate thresholds -- starting hypotheses for the validation harness to
# confirm or recalibrate, not backtested yet. |Z| >= 1.0 means today's
# move-so-far is at least one "typical day's" worth of volatility;
# E >= 0.6 means at least 60% of today's total travel was net progress.
GATE_MIN_ABS_Z = 1.0
GATE_MIN_EFFICIENCY = 0.6


class IntradayVolAdjustedAgent:
    """
    Given today's hourly bars up to some checkpoint time and a trailing
    daily-volatility estimate, answers two questions: is today's move
    (so far) unusually large for this stock, and was that move clean
    (trending) or choppy (whipsaw)?
    """

    def __init__(self, hourly_bars_today: pd.DataFrame, day_open: float,
                 trailing_daily_vol_pct: float):
        """
        hourly_bars_today: DataFrame of today's hourly bars UP TO AND
            INCLUDING the checkpoint bar, sorted by time ascending, with a
            'Close' column at minimum (High/Low/Open unused here but
            commonly present from the same fetch).
        day_open: today's opening price (first bar's Open, or actual
            session open if available).
        trailing_daily_vol_pct: stddev of daily returns (as a fraction,
            e.g. 0.02 for 2%) over a trailing window (20 trading days is
            this repo's usual convention -- see rs_agent.py's ann_vol calc)
            -- deliberately NOT annualized here, since we're comparing to
            a single day's move, not a multi-week return.
        """
        self.hourly_bars_today = hourly_bars_today
        self.day_open = day_open
        self.trailing_daily_vol_pct = trailing_daily_vol_pct

        self._bars_available = 0
        self._price_now = 0.0
        self._z = 0.0
        self._efficiency = 0.0

        self._compute()

    def _compute(self):
        if self.hourly_bars_today is None or self.hourly_bars_today.empty:
            return
        if "Close" not in self.hourly_bars_today.columns:
            return
        if self.day_open <= 0:
            return

        closes = self.hourly_bars_today["Close"].to_numpy(dtype=float)
        self._bars_available = len(closes)
        if self._bars_available < MIN_BARS_FOR_SIGNAL:
            return

        self._price_now = float(closes[-1])
        move = self._price_now - self.day_open

        # Path = day_open -> bar1 -> bar2 -> ... -> bar_now, so a stock
        # that opened, spiked, and round-tripped back shows a long path
        # relative to its small net move (low efficiency) -- exactly the
        # "choppy" case this agent is meant to catch.
        path_points = np.concatenate(([self.day_open], closes))
        total_path = float(np.sum(np.abs(np.diff(path_points))))
        self._efficiency = round(abs(move) / total_path, 3) if total_path > 0 else 0.0

        if self.trailing_daily_vol_pct > 0:
            vol_in_price = self.trailing_daily_vol_pct * self.day_open
            self._z = round(move / vol_in_price, 3) if vol_in_price > 0 else 0.0

    # ── public interface ────────────────────────────────────────────────

    def passes_gate(self) -> bool:
        """
        [VALIDATION-DIAGNOSTIC -- integration style not yet confirmed.]
        True (pass) when today's move is both sizeable (|Z| >= 1.0) AND
        clean (efficiency >= 0.6), OR when there aren't enough bars yet --
        fails open, same convention as every other prototype gate here.
        """
        if self._bars_available < MIN_BARS_FOR_SIGNAL:
            return True
        return abs(self._z) >= GATE_MIN_ABS_Z and self._efficiency >= GATE_MIN_EFFICIENCY

    def score_bonus(self) -> float:
        """
        Continuous alternative to the hard gate, for the validation run to
        compare against passes_gate() on real historical outcomes. Blends
        |Z| and efficiency multiplicatively (a big move that's also clean
        scores much higher than either alone -- deliberately not additive,
        since a huge choppy move and a huge clean move should NOT land at
        similar scores the way an additive blend would let them).
        Clamped to the same +/-2..+3 range every other prototype bonus in
        this repo uses.
        """
        if self._bars_available < MIN_BARS_FOR_SIGNAL:
            return 0.0
        raw = abs(self._z) * self._efficiency * 1.5
        signed = raw if self._z >= 0 else -raw
        return round(float(np.clip(signed, -2.0, 3.0)), 2)

    def get_z(self) -> float:
        return self._z

    def get_efficiency(self) -> float:
        return self._efficiency

    def get_bars_available(self) -> int:
        return self._bars_available
