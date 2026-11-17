"""
Shared, dependency-free RVOL window + baseline helpers.

Used by dhan_rvol.compute_rvol() and confirm_picks.py's tvDatafeed / yfinance
fallbacks so all three RVOL sources measure the SAME window the SAME way.
Kept free of env/network imports (dhan_rvol reads DHAN_* at import time) so
it is trivially unit-testable.

[2026-11-17 RCA] Why "completed bars only":
Intraday candles are stamped with their START time. The old window summed
every bar whose start was <= the run time. Today the newest of those bars is
still forming (e.g. at 10:05:46 the 10:05 bar holds ~46 seconds of volume),
but on every historical day that same bar is complete (a full 5 minutes).
So "today" was systematically short by up to one bar versus its own
baseline: ~50% of the window at the 09:20 checkpoint, ~8% at 10:05.
Verified live on 17 Nov: YATHARTH logged 0.98x at 09:35 (BREAKOUT_NO_VOLUME)
where the completed-bar reading was 1.22x (CONFIRMED_LOW_VOL).

Fix: both today and history are cut at the end of the last COMPLETED bar,
and additionally at the end of the last bar the feed has actually published
for today (Dhan can lag a bar) -- so the two sides always cover identical
minutes.
"""

from __future__ import annotations

import statistics
from datetime import datetime, time, timedelta
from typing import Iterable, Optional

MARKET_OPEN = time(9, 15)


def completed_cutoff(as_of: datetime, interval_min: int,
                     today_bar_starts: Iterable[datetime] = ()) -> Optional[time]:
    """
    Time-of-day (exclusive upper bound on bar START times) for an
    elapsed-matched window made of completed bars only.

    as_of: naive IST wall-clock run time.
    interval_min: candle size in minutes (5 for Dhan, 15 tvDatafeed, 1 yfinance).
    today_bar_starts: naive IST start times of today's bars actually returned
        by the feed. If the last completed bar has not been published yet,
        the window is pulled back to the last bar that HAS been, so today is
        never compared against more minutes of history than it has data for.

    Returns None when not even one complete bar exists yet.
    A bar starting at S is included iff MARKET_OPEN <= S < cutoff.
    """
    open_dt = datetime.combine(as_of.date(), MARKET_OPEN)
    elapsed_min = (as_of - open_dt).total_seconds() / 60
    n_complete = int(elapsed_min // interval_min)
    if n_complete < 1:
        return None
    cutoff_dt = open_dt + timedelta(minutes=n_complete * interval_min)

    published = [s for s in today_bar_starts
                 if s.date() == as_of.date() and open_dt <= s < cutoff_dt]
    if published:
        last_end = max(published) + timedelta(minutes=interval_min)
        cutoff_dt = min(cutoff_dt, last_end)
    return cutoff_dt.time()


def window_minutes(cutoff: Optional[time]) -> Optional[float]:
    """Minutes actually measured by a window ending at `cutoff`."""
    if cutoff is None:
        return None
    return (datetime.combine(datetime.min, cutoff)
            - datetime.combine(datetime.min, MARKET_OPEN)).total_seconds() / 60


def in_window(bar_start_time: time, cutoff: time) -> bool:
    return MARKET_OPEN <= bar_start_time < cutoff


def rvol_mean_median(today_volume: float, hist_volumes: list) -> tuple:
    """
    Returns (rvol_mean, rvol_median), each rounded to 2dp or None.

    rvol_mean is the LIVE gate value (unchanged definition). rvol_median is
    SHADOW only [2026-11-17, fix 3]: a single event day (e.g. KOTAKBANK
    12 Nov, 7.0M first-hour shares vs a normal 1.5-3.5M) drags the mean
    baseline up and every later reading down. Logged side-by-side to Turso
    so a switch can be decided on evidence, not wired in blind.
    """
    hv = [v for v in hist_volumes if v and v > 0]
    if not hv:
        return None, None
    mean_b, med_b = sum(hv) / len(hv), statistics.median(hv)
    rm = round(today_volume / mean_b, 2) if mean_b > 0 else None
    rmed = round(today_volume / med_b, 2) if med_b > 0 else None
    return rm, rmed
