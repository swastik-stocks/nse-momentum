"""
Run with: python3 -m pytest test_staleness_check.py -v
(or just: python3 test_staleness_check.py -- it also runs standalone)

These tests reproduce your actual bug scenario first, then check the
edge cases that would otherwise bite you next (Monday after a Friday
scan, and the day after an NSE holiday).
"""

from datetime import date
from staleness_check import (
    check_staleness,
    previous_trading_day,
    is_trading_day,
    StaleDataError,
)


def test_the_actual_bug_that_happened():
    """21 Jul (Tue) evening scan -> 22 Jul (Wed) 10am confirm.
    This should NOT raise -- it's the correct, expected state."""
    picks_date = date(2026, 7, 21)
    today = date(2026, 7, 22)
    check_staleness(picks_date, today)  # should not raise
    print("PASS: Jul 21 picks confirmed on Jul 22 -- no false alarm")


def test_monday_after_friday_scan():
    """Fri evening scan -> Mon 10am confirm. Weekend gap is normal."""
    friday = date(2026, 7, 24)
    monday = date(2026, 7, 27)
    check_staleness(friday, monday)  # should not raise
    print("PASS: Friday picks confirmed on Monday -- no false alarm")


def test_day_after_holiday():
    """Scan before a holiday -> confirm on the day after the holiday."""
    before_holiday = date(2026, 1, 23)   # Friday before Republic Day
    after_holiday = date(2026, 1, 27)    # Tuesday after Mon 26 Jan holiday
    check_staleness(before_holiday, after_holiday)  # should not raise
    print("PASS: pre-holiday picks confirmed after holiday -- no false alarm")


def test_genuinely_stale_data_still_caught():
    """Scan is 3+ days old with no holiday excuse -- SHOULD raise."""
    old_picks = date(2026, 7, 15)
    today = date(2026, 7, 22)
    try:
        check_staleness(old_picks, today)
        raise AssertionError("Should have raised StaleDataError!")
    except StaleDataError as e:
        print(f"PASS: genuinely stale data correctly caught -> {e}")


def test_previous_trading_day_basic():
    assert previous_trading_day(date(2026, 7, 22)) == date(2026, 7, 21)
    assert previous_trading_day(date(2026, 7, 27)) == date(2026, 7, 24)  # Mon->Fri
    print("PASS: previous_trading_day basic cases correct")


if __name__ == "__main__":
    test_the_actual_bug_that_happened()
    test_monday_after_friday_scan()
    test_day_after_holiday()
    test_genuinely_stale_data_still_caught()
    test_previous_trading_day_basic()
    print("\nAll tests passed.")
