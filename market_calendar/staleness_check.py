"""
Fix for the confirm_picks.py staleness-check bug.

BUG: The 10am confirmation was comparing picks_latest.json's embedded date
against TODAY's date, and flagging a mismatch as "stale". But by design,
the evening scan runs the NIGHT BEFORE and produces picks meant to be
confirmed the NEXT trading morning -- so picks_latest.json being dated
"yesterday" is the CORRECT, expected state, not an error.

FIX: Compare against the previous TRADING day (skipping weekends AND
NSE holidays), not literally "today".

Drop this into your repo (e.g. as nse_momentum/market_calendar.py) and
import previous_trading_day() + check_staleness() into confirm_picks.py.
"""

from datetime import date, timedelta

# Official NSE 2026 trading holidays (equity segment), sourced from
# Zerodha's published holiday calendar. Update this list each December
# when NSE publishes the following year's calendar.
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 15),   # Municipal Corporation Elections in Maharashtra
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 3),    # Holi
    date(2026, 3, 26),   # Shri Ram Navami
    date(2026, 3, 31),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Eid
    date(2026, 6, 26),   # Moharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali-Balipratipada
    date(2026, 11, 24),  # Prakash Gurpurb Sri Guru Nanak Dev
    date(2026, 12, 25),  # Christmas
}


def is_trading_day(d: date, holidays: set = NSE_HOLIDAYS_2026) -> bool:
    """NSE is closed on weekends and the holidays listed above."""
    if d.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    if d in holidays:
        return False
    return True


def previous_trading_day(ref_date: date, holidays: set = NSE_HOLIDAYS_2026) -> date:
    """Returns the most recent trading day strictly before ref_date."""
    d = ref_date - timedelta(days=1)
    while not is_trading_day(d, holidays):
        d -= timedelta(days=1)
    return d


def check_staleness(picks_date: date, today: date = None) -> None:
    """
    Raises StaleDataError only if picks_date does NOT match the last
    trading day before `today` -- i.e. only a REAL gap (evening scan
    genuinely missed running) triggers the alert.

    Usage in confirm_picks.py:
        picks_date = date.fromisoformat(picks_json["scan_date"])
        check_staleness(picks_date)
    """
    if today is None:
        today = date.today()

    expected_scan_date = previous_trading_day(today)

    if picks_date != expected_scan_date:
        raise StaleDataError(
            f"picks_latest.json is dated {picks_date} but the last "
            f"trading day before {today} was {expected_scan_date}. "
            f"Evening scan appears to have genuinely missed a run."
        )


class StaleDataError(Exception):
    pass
