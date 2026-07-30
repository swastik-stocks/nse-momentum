"""
Shared scaling constant for running any agent against hourly bars instead
of daily bars. NSE's trading session (09:15-15:30) resamples into ~7
clock-hour bins/day (build_hourly_from_1min.py's convention: clock-hour
boundaries, so the first bin is a partial 09:15-10:00 hour and the last
is a partial 15:00-15:30 hour, but both still count as one bin each).

Every agent below defaults its own bars_per_day param to 1 (unchanged
daily behavior — the live scanner never passes this, so it is provably
unaffected by any of this work). Only the hourly replay script passes
BARS_PER_DAY explicitly.
"""

BARS_PER_DAY = 7
