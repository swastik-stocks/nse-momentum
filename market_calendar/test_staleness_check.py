"""
Run with: python3 -m pytest test_staleness_check.py -v
(or just: python3 test_staleness_check.py -- it also runs standalone)

These tests reproduce your actual bug scenario first, then check the
edge cases that would otherwise bite you next (Monday after a Friday
scan, and the day after an NSE holiday).
"""

from datetime import date, datetime
from staleness_check import (
    check_staleness,
    previous_trading_day,
    is_trading_day,
    verify_generation_consistency,
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


def test_generation_consistency_holiday_rerun():
    """2026-09-15 incident: cron-job.org's evening-scan trigger fires Mon-Fri
    with no NSE-holiday awareness, so it fired on Mon 14 Sep (Ganesh
    Chaturthi). scanner.py correctly re-verified/re-used Fri 11 Sep's real
    Bhavcopy and produced a fresh artifact generated Monday evening --
    generated_at's date (14 Sep) legitimately differs from
    bhavcopy_trading_date (11 Sep). Must NOT be flagged as stale/replayed."""
    prov = verify_generation_consistency("2026-09-11", datetime(2026, 9, 14, 20, 55))
    assert prov.ok, f"legitimate holiday carry-forward wrongly rejected: {prov.reason}"
    print("PASS: holiday re-run carry-forward correctly accepted")


def test_generation_consistency_normal_day():
    """Normal case: scan runs the same evening as the trading day it covers."""
    prov = verify_generation_consistency("2026-07-21", datetime(2026, 7, 21, 19, 45))
    assert prov.ok, f"same-day generation wrongly rejected: {prov.reason}"
    print("PASS: normal same-day generation correctly accepted")


def test_generation_consistency_catches_real_mismatch():
    """A bhavcopy_trading_date that ISN'T the correct last-trading-day answer
    for generated_at's own date is still a genuine problem and must still
    be caught -- the fix must not turn this into a blanket pass-through."""
    prov = verify_generation_consistency("2026-08-01", datetime(2026, 9, 14, 20, 55))
    assert not prov.ok, "genuinely stale/replayed artifact was NOT caught"
    print(f"PASS: genuine mismatch correctly caught -> {prov.reason}")


if __name__ == "__main__":
    test_the_actual_bug_that_happened()
    test_monday_after_friday_scan()
    test_day_after_holiday()
    test_genuinely_stale_data_still_caught()
    test_previous_trading_day_basic()
    test_generation_consistency_holiday_rerun()
    test_generation_consistency_normal_day()
    test_generation_consistency_catches_real_mismatch()
    print("\nAll tests passed.")


# ── Special sessions (2026-11-07: Sunday Muhurat) ───────────────────────────
def test_sunday_muhurat_is_trading_day():
    from market_calendar.staleness_check import is_trading_day, special_session_label
    assert is_trading_day(date(2026, 11, 8))            # Sunday Muhurat
    assert not is_trading_day(date(2026, 11, 7))        # ordinary Saturday
    assert not is_trading_day(date(2026, 11, 15))       # ordinary Sunday
    assert "Muhurat" in special_session_label(date(2026, 11, 8))
    assert special_session_label(date(2026, 11, 9)) is None


def test_monday_after_muhurat_expects_sunday_picks():
    from market_calendar.staleness_check import previous_trading_day
    assert previous_trading_day(date(2026, 11, 9)) == date(2026, 11, 8)


def test_last_trading_day_on_muhurat_evening():
    from data_fetcher import get_last_trading_day
    from fetch_mcap import _get_last_trading_day
    assert get_last_trading_day(date(2026, 11, 8)) == date(2026, 11, 8)
    assert _get_last_trading_day(date(2026, 11, 8)) == date(2026, 11, 8)
