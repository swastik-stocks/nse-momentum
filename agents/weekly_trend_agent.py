"""
NSE Momentum vNext - Weekly Trend Agent   [PROPOSED -- NOT WIRED INTO orchestrator.py]

STATUS: prototype only. Per the same evidence-first discipline documented in
pattern_agent.py's DEFAULT_WEIGHTS history (A7) -- this must NOT be added to
live scoring until it has been run through the same validation chain
(monte_carlo_significance.py bootstrap/permutation test, ideally
split_period_significance.py for out-of-sample consistency) that every
other live-scoring input has already passed. This file defines the agent's
INTERFACE and computation only.

WHY THIS EXISTS
    A well-documented real edge in swing/momentum trading is timeframe
    alignment: a daily breakout is more reliable when the weekly trend
    agrees. Nothing in the current gate chain (G1-G7) checks the weekly
    picture at all. This agent computes that missing signal from the SAME
    daily OHLCV already being fetched -- no new data source needed for
    live scanning, since weekly bars are just a resample of daily.

DATA NOTE (from the 2026-07-29 Kaggle upload used to prototype this):
    The daily CSV (nifty500_1d.csv) has real, usable depth -- coverage
    varies by ticker/listing date, but large-caps like TCS run back to
    Dec 2005 (~20 years, ~5,000 daily bars). This is more than enough for
    weekly resampling and a multi-year evidence-chain validation. The
    supplied 15m/1m intraday files, by contrast, only cover ~10 months and
    ~5 weeks respectively -- see the accompanying chat message for why
    that matters for the SEPARATE hourly-evidence-chain item.

INTERFACE
    Deliberately mirrors RSAgent (agents/rs_agent.py): plain __init__(df),
    internal _compute(), and BOTH a binary gate (.passes_gate()) and a
    small graduated bonus (.score_bonus(), capped like RSAgent's own
    sector-relative modifier) so the validation run can test which
    integration style (hard gate vs. soft bonus) actually holds up,
    rather than committing to one before there's evidence either way.

USAGE (once validated and wired into orchestrator.py):
    wta = WeeklyTrendAgent(df)  # df = the same daily OHLCV RiskAgent/
                                  # PatternAgent already receive
    if wta.get_weeks_available() >= MIN_WEEKS:
        # binary candidate:
        weekly_ok = wta.passes_gate()
        # OR graduated candidate:
        r.bonus_score += wta.score_bonus()
"""

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Minimum weekly bars required before trusting this agent's output at all --
# below this, _compute() leaves everything at neutral defaults (trend_up=
# False, slope=0.0) rather than guessing off a handful of weekly bars.
MIN_WEEKS_FOR_SIGNAL = 30   # ~7 months of weekly bars minimum


class WeeklyTrendAgent:
    """
    Resamples daily OHLCV to weekly bars (week ending Friday, matching NSE's
    trading week) and answers: is the weekly trend constructive right now?

    Two independent signals, both derived from the weekly EMA:
      - trend_up:  last weekly close > weekly EMA(ema_period)
      - slope_pct: % change in the weekly EMA over the last slope_lookback
                   weeks -- positive means the EMA itself is rising, not
                   just that price is above a flat/falling one.
    Both must be true for passes_gate() -- price merely being above a
    declining EMA (a possible short-term bounce) does not count as trend
    alignment.
    """

    def __init__(self, daily_df: pd.DataFrame,
                 ema_period: int = 10, slope_lookback: int = 3):
        self.daily_df       = daily_df
        self.ema_period     = ema_period
        self.slope_lookback = slope_lookback

        self.weekly_df       = pd.DataFrame()
        self._weeks_available = 0
        self._trend_up        = False
        self._ema_val         = 0.0
        self._slope_pct       = 0.0
        self._last_close      = 0.0

        self._compute()

    # ── resampling ──────────────────────────────────────────────────────

    def _get_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a copy of df with a proper DatetimeIndex, needed for
        .resample(). Handles both the case where df already has one
        (expected in live scanning, per data_fetcher.py) and the case
        where a 'Datetime'/'Date' column needs to be promoted (the shape
        raw Kaggle CSV rows come in before any per-ticker preprocessing).
        """
        if isinstance(df.index, pd.DatetimeIndex):
            return df
        for col in ("Datetime", "Date", "date", "datetime"):
            if col in df.columns:
                out = df.copy()
                out[col] = pd.to_datetime(out[col])
                out = out.set_index(col).sort_index()
                return out
        raise ValueError(
            "WeeklyTrendAgent requires a DatetimeIndex or a "
            "'Datetime'/'Date' column to resample to weekly bars."
        )

    def _resample_weekly(self) -> pd.DataFrame:
        df = self._get_datetime_index(self.daily_df)
        agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            agg["Volume"] = "sum"
        weekly = df.resample("W-FRI").agg(agg).dropna(subset=["Close"])
        return weekly

    # ── computation ─────────────────────────────────────────────────────

    def _compute(self):
        if self.daily_df is None or self.daily_df.empty:
            return

        try:
            weekly = self._resample_weekly()
        except ValueError as e:
            log.debug(f"WeeklyTrendAgent: {e}")
            return

        self.weekly_df        = weekly
        self._weeks_available = len(weekly)

        if self._weeks_available < max(MIN_WEEKS_FOR_SIGNAL,
                                        self.ema_period + self.slope_lookback):
            # Not enough weekly history to trust a signal -- stay neutral
            # rather than computing an EMA off a handful of bars.
            return

        close = weekly["Close"].to_numpy(dtype=float)
        ema   = pd.Series(close).ewm(span=self.ema_period, adjust=False).mean().to_numpy()

        self._ema_val    = float(ema[-1])
        self._last_close = float(close[-1])
        self._trend_up   = self._last_close > self._ema_val

        lb = self.slope_lookback
        if len(ema) > lb and ema[-1 - lb] > 0:
            self._slope_pct = round((ema[-1] / ema[-1 - lb] - 1) * 100, 2)

    # ── public interface ────────────────────────────────────────────────

    def passes_gate(self) -> bool:
        """
        [VALIDATION-DIAGNOSTIC ONLY -- not intended for live wiring.]

        Momentum alignment is ordinal, not categorical: a stock with a
        weekly EMA rising 2.1% isn't "aligned" while one rising 1.9% is
        "not aligned" -- they're both mildly-to-moderately supportive,
        just by degree. Hard-gating on an ordinal signal creates arbitrary
        cliffs that bite exactly when regimes compress or expand
        volatility. This method exists ONLY so the evidence-chain
        validation can compare a hard-gate integration against the
        continuous score_bonus() below on real historical outcomes --
        whichever the data actually supports wins, but the prior going in
        is soft-bonus, consistent with how RSAgent already treats its own
        primary signal (continuous score, with a hard floor reserved only
        for the extreme laggard tail, not the whole signal).
        """
        if self._weeks_available < MIN_WEEKS_FOR_SIGNAL:
            return True   # insufficient history -> don't gate on it, fail open
        return self._trend_up and self._slope_pct > 0

    def score_bonus(self) -> float:
        """
        Continuous, capped bonus -- the live-wiring candidate. Deliberately
        avoids discrete step thresholds (an earlier draft of this method
        had if slope > 2: +3 elif slope > 0: +1 -- a hard gate wearing a
        soft-bonus costume, since a stock at +1.9% and one at +2.1% would
        be treated as categorically different). Instead this blends two
        continuous measures -- the weekly EMA's slope, and how far the
        last weekly close sits above/below that EMA -- into one smoothly
        scaled figure, clamped to the same +/-2..+3 range RSAgent already
        uses for its own small sector-relative modifier, so this stays a
        refinement of the total score, not a dominant factor.

        Insufficient weekly history returns 0.0 (neutral) rather than
        penalizing -- a data-quality gap is not a momentum judgment.
        """
        if self._weeks_available < MIN_WEEKS_FOR_SIGNAL or self._ema_val <= 0:
            return 0.0

        pct_above_ema = (self._last_close - self._ema_val) / self._ema_val * 100

        raw = 0.30 * self._slope_pct + 0.20 * pct_above_ema
        return round(float(np.clip(raw, -2.0, 3.0)), 2)

    def get_weekly_ema(self) -> float:
        return round(self._ema_val, 2)

    def get_slope_pct(self) -> float:
        return self._slope_pct

    def get_weeks_available(self) -> int:
        return self._weeks_available

    def get_trend_up(self) -> bool:
        return self._trend_up
